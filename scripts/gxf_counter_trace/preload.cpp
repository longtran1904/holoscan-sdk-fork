/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "counter_reader.hpp"
#include "gxf/core/gxf.h"
#include "trace_build.hpp"

#include <dlfcn.h>
#include <elf.h>
#include <link.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
constexpr char kDeinitialize[] = "_ZN6nvidia3gxf19EventBasedScheduler12deinitializeEv";
using Clock = std::chrono::steady_clock;

uint64_t monotonic_ns() {
  timespec t{};
  clock_gettime(CLOCK_MONOTONIC, &t);
  return uint64_t(t.tv_sec) * 1000000000 + t.tv_nsec;
}

template <typename Function>
Function next(const char* symbol) {
  auto pointer = reinterpret_cast<Function>(dlsym(RTLD_NEXT, symbol));
  if (!pointer) {
    std::fprintf(stderr, "[ebs-trace] Cannot resolve %s: %s\n", symbol, dlerror());
    // Calling a missing original entry point cannot safely preserve application behavior.
    std::abort();
  }
  return pointer;
}

std::string quote(const std::string& value) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : value) {
    if (c == '"' || c == '\\')
      out << '\\' << c;
    else if (c < 32)
      out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c);
    else
      out << c;
  }
  out << '"';
  return out.str();
}

int positive_env(const char* name, int fallback, int maximum) {
  const char* value = std::getenv(name);
  if (!value)
    return fallback;
  char* end = nullptr;
  long result = std::strtol(value, &end, 10);
  if (!*value || *end || result < 1 || result > maximum)
    throw std::runtime_error(std::string("Invalid ") + name);
  return static_cast<int>(result);
}

struct Library {
  uintptr_t base = 0;
  std::string path;
  std::string build_id;
};

int find_build_id(dl_phdr_info* info, size_t, void* opaque) {
  auto& library = *static_cast<Library*>(opaque);
  if (info->dlpi_addr != library.base)
    return 0;
  for (int i = 0; i < info->dlpi_phnum; ++i) {
    const auto& ph = info->dlpi_phdr[i];
    if (ph.p_type != PT_NOTE)
      continue;
    const auto* begin = reinterpret_cast<const unsigned char*>(info->dlpi_addr + ph.p_vaddr);
    const auto* end = begin + ph.p_memsz;
    while (size_t(end - begin) >= sizeof(ElfW(Nhdr))) {
      const auto* note = reinterpret_cast<const ElfW(Nhdr)*>(begin);
      const size_t names = (size_t(note->n_namesz) + 3) & ~size_t(3);
      const size_t descs = (size_t(note->n_descsz) + 3) & ~size_t(3);
      begin += sizeof(*note);
      if (names > size_t(end - begin) || descs > size_t(end - begin) - names)
        break;
      if (note->n_type == NT_GNU_BUILD_ID && note->n_namesz == 4 &&
          std::string(reinterpret_cast<const char*>(begin), 4) == std::string("GNU\0", 4)) {
        std::ostringstream id;
        for (size_t j = 0; j < note->n_descsz; ++j)
          id << std::hex << std::setw(2) << std::setfill('0') << unsigned(begin[names + j]);
        library.build_id = id.str();
      }
      begin += names + descs;
    }
  }
  return 1;
}

Library loaded_library() {
  Dl_info info{};
  void* function = dlsym(RTLD_NEXT, kDeinitialize);
  if (!function || !dladdr(function, &info))
    throw std::runtime_error("GXF symbol not found");
  Library library{reinterpret_cast<uintptr_t>(info.dli_fbase), info.dli_fname, {}};
  dl_iterate_phdr(find_build_id, &library);
  if (library.build_id != EBS_TRACE_BUILD_ID)
    throw std::runtime_error("GXF ELF build ID mismatch: " + library.build_id);
  return library;
}

struct Sample {
  uint64_t ns;
  uint64_t end_ns;
  const char* phase;
  CounterSnapshot counters;
};

struct Session {
  gxf_context_t context;
  void* scheduler;
  std::string prefix;
  Library library;
  int interval_ms;
  size_t capacity;
  std::vector<Sample> samples;
  bool truncated = false;
  bool stop_requested = false;
  bool sampler_ready = false;
  int sampler_policy_error = 0;
  std::mutex mutex;
  std::condition_variable wake;
  std::thread thread;

  Session(gxf_context_t c, void* s, std::string p, Library lib, int interval, size_t cap)
      : context(c),
        scheduler(s),
        prefix(std::move(p)),
        library(std::move(lib)),
        interval_ms(interval),
        capacity(cap) {
    samples.reserve(capacity);
  }

  void capture(const char* phase, bool final = false) {
    if (samples.size() >= capacity - (final ? 0 : 1)) {
      truncated = true;
      return;  // Always reserve a slot for the final snapshot.
    }
    const uint64_t begin = monotonic_ns();
    const auto counters = read_counters(scheduler);
    samples.push_back({begin, monotonic_ns(), phase, counters});
  }

  void metadata(const std::string& status, bool stable) {
    const std::string path = prefix + ".json";
    std::ofstream out(path + ".tmp");
    out.exceptions(std::ios::failbit | std::ios::badbit);
    out << "{\n  \"status\": " << quote(status)
        << ",\n  \"final_stable\": " << (stable ? "true" : "false")
        << ",\n  \"truncated\": " << (truncated ? "true" : "false")
        << ",\n  \"interval_ms\": " << interval_ms << ",\n  \"capacity\": " << capacity
        << ",\n  \"sample_count\": " << samples.size()
        << ",\n  \"clock\": \"CLOCK_MONOTONIC\",\n  \"pid\": " << getpid()
        << ",\n  \"library_path\": " << quote(library.path)
        << ",\n  \"library_build_id\": " << quote(library.build_id)
        << ",\n  \"library_sha256_at_build\": " << quote(EBS_TRACE_LIBRARY_SHA256)
        << ",\n  \"header_sha256\": " << quote(EBS_TRACE_HEADER_SHA256)
        << ",\n  \"compiler\": " << quote(__VERSION__)
        << ",\n  \"scheduler_size\": " << scheduler_size()
        << ",\n  \"counters_offset\": " << counters_offset(scheduler)
        << ",\n  \"running_threads_offset\": " << running_threads_offset(scheduler)
        << ",\n  \"sampler_policy_error\": " << sampler_policy_error << "\n}\n";
    out.close();
    std::filesystem::rename(path + ".tmp", path);
  }

  void start() {
    capture("initial");
    metadata("recording", false);  // Survives abrupt termination: absence of 'complete' is visible.
    thread = std::thread([this] {
      sched_param param{};
      const int policy_error = pthread_setschedparam(pthread_self(), SCHED_OTHER, &param);
      std::unique_lock lock(mutex);
      sampler_policy_error = policy_error;
      sampler_ready = true;
      wake.notify_all();
      auto deadline = Clock::now() + std::chrono::milliseconds(interval_ms);
      while (!wake.wait_until(lock, deadline, [this] { return stop_requested; })) {
        capture("periodic");
        // Skip missed deadlines; do not burst samples after descheduling.
        deadline += std::chrono::milliseconds(interval_ms);
        if (deadline <= Clock::now())
          deadline = Clock::now() + std::chrono::milliseconds(interval_ms);
      }
    });
    std::unique_lock lock(mutex);
    wake.wait(lock, [this] { return sampler_ready; });
  }

  void stop() {
    {
      std::lock_guard lock(mutex);
      stop_requested = true;
    }
    wake.notify_all();
    if (thread.joinable())
      thread.join();
  }

  void finish(const char* reason, bool stable) {
    stop();
    capture(stable ? "final" : "incomplete", true);
    std::ofstream out(prefix + ".csv");
    out.exceptions(std::ios::failbit | std::ios::badbit);
    out << "monotonic_ns,read_end_ns,phase";
#define HEADER(name) out << "," #name;
    EBS_TRACE_COUNTERS(HEADER)
#undef HEADER
    out << ",running_threads\n";
    for (const auto& sample : samples) {
      out << sample.ns << ',' << sample.end_ns << ',' << sample.phase;
#define VALUE(name) out << ',' << sample.counters.name;
      EBS_TRACE_COUNTERS(VALUE)
#undef VALUE
      out << ',' << sample.counters.running_threads << '\n';
    }
    out.close();
    metadata(reason, stable);
  }

  ~Session() { stop(); }
};

struct State {
  std::mutex mutex;
  std::unique_ptr<Session> active;
  uint64_t sequence = 0;
};

State& state() {
  // GXF can call through us during static destruction. Avoid destruction-order dependencies.
  static State* value = new State;
  return *value;
}

template <typename Action>
void guarded(Action action) noexcept {
  try {
    action();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[ebs-trace] %s\n", e.what());
  } catch (...) {
    std::fprintf(stderr, "[ebs-trace] Unknown instrumentation error\n");
  }
}

void* find_scheduler(gxf_context_t context) {
  gxf_tid_t tid{};
  if (GxfComponentTypeId(context, "nvidia::gxf::EventBasedScheduler", &tid) != GXF_SUCCESS)
    throw std::runtime_error("No EventBasedScheduler type registered");
  uint64_t count = 0;
  auto result = GxfEntityFindAll(context, &count, nullptr);
  if (result != GXF_SUCCESS && result != GXF_QUERY_NOT_ENOUGH_CAPACITY)
    throw std::runtime_error("Cannot enumerate GXF entities");
  std::vector<gxf_uid_t> entities(count);
  if (GxfEntityFindAll(context, &count, entities.data()) != GXF_SUCCESS)
    throw std::runtime_error("Cannot enumerate GXF entities");
  void* scheduler = nullptr;
  for (auto eid : entities) {
    int32_t offset = 0;
    gxf_uid_t cid = 0;
    while (GxfComponentFind(context, eid, tid, nullptr, &offset, &cid) == GXF_SUCCESS) {
      if (scheduler)
        throw std::runtime_error("Multiple EventBasedSchedulers are unsupported");
      if (GxfComponentPointer(context, cid, tid, &scheduler) != GXF_SUCCESS || !scheduler)
        throw std::runtime_error("Cannot get scheduler pointer");
      ++offset;
    }
  }
  if (!scheduler)
    throw std::runtime_error("No EventBasedScheduler component found");
  return scheduler;
}

void begin(gxf_context_t context) {
  const char* directory = std::getenv("GXF_EBS_TRACE_DIR");
  if (!directory || !*directory)
    return;
  auto& registry = state();
  std::lock_guard lock(registry.mutex);
  const std::string prefix = std::string(directory) + "/ebs-" + std::to_string(getpid()) + "-" +
                             std::to_string(++registry.sequence) + "-" +
                             std::to_string(monotonic_ns());
  try {
    if (registry.active) {
      registry.active->finish("unsupported_concurrent_graph", false);
      registry.active.reset();
      throw std::runtime_error("Concurrent graphs are unsupported");
    }
    auto library = loaded_library();  // Reject a different binary before any private reads.
    void* scheduler = find_scheduler(context);
    const int interval = positive_env("GXF_EBS_TRACE_INTERVAL_MS", 10, 60000);
    const int capacity = positive_env("GXF_EBS_TRACE_MAX_SAMPLES", 60000 / interval + 2, 600002);
    if (capacity < 2)
      throw std::runtime_error("Trace capacity must be at least 2");
    std::filesystem::create_directories(directory);
    auto session =
        std::make_unique<Session>(context, scheduler, prefix, library, interval, capacity);
    session->start();
    registry.active = std::move(session);
  } catch (const std::exception& e) {
    std::filesystem::create_directories(directory);
    std::ofstream out(prefix + ".json");
    out << "{\"status\":\"rejected\",\"error\":" << quote(e.what()) << "}\n";
    throw;
  }
}

void finish(gxf_context_t context, void* scheduler, const char* reason, bool stable) {
  auto& registry = state();
  std::lock_guard lock(registry.mutex);
  if (registry.active &&
      (registry.active->context == context || registry.active->scheduler == scheduler)) {
    // Detach first: even an I/O error must never leave a sampler reading a destroyed object.
    auto session = std::move(registry.active);
    session->finish(reason, stable);
  }
}
}  // namespace

extern "C" gxf_result_t GxfGraphRunAsync(gxf_context_t context) {
  static const auto original = next<decltype(&GxfGraphRunAsync)>("GxfGraphRunAsync");
  guarded([&] { begin(context); });
  const auto result = original(context);
  if (result != GXF_SUCCESS)
    guarded([&] { finish(context, nullptr, "run_failed", false); });
  return result;
}

extern "C" gxf_result_t GxfGraphWait(gxf_context_t context) {
  static const auto original = next<decltype(&GxfGraphWait)>("GxfGraphWait");
  const auto result = original(context);
  guarded([&] {
    finish(context,
           nullptr,
           result == GXF_SUCCESS ? "complete" : "wait_failed",
           result == GXF_SUCCESS);
  });
  return result;
}

extern "C" gxf_result_t GxfGraphDeactivate(gxf_context_t context) {
  static const auto original = next<decltype(&GxfGraphDeactivate)>("GxfGraphDeactivate");
  guarded([&] { finish(context, nullptr, "deactivated_without_wait", false); });
  return original(context);
}

extern "C" gxf_result_t GxfContextDestroy(gxf_context_t context) {
  static const auto original = next<decltype(&GxfContextDestroy)>("GxfContextDestroy");
  guarded([&] { finish(context, nullptr, "destroyed_without_wait", false); });
  return original(context);
}

extern "C" gxf_result_t trace_deinitialize(void*) asm(
    "_ZN6nvidia3gxf19EventBasedScheduler12deinitializeEv");
extern "C" gxf_result_t trace_deinitialize(void* scheduler) {
  static const auto original = next<gxf_result_t (*)(void*)>(kDeinitialize);
  guarded([&] { finish(nullptr, scheduler, "deinitialized_without_wait", false); });
  return original(scheduler);
}
