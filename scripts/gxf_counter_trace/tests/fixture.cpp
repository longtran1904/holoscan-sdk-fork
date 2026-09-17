/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Synthetic GXF lifecycle for testing preload behavior, not the private-member ABI.
// The latter is validated against the real GXF final log by run.py.
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <thread>
#include "counter_reader.hpp"
#include "gxf/core/gxf.h"

struct FakeScheduler {
  std::atomic<bool> alive{true};
  std::atomic<uint64_t> ticks{0};
};
struct FakeContext {
  FakeScheduler scheduler;
};

extern "C" gxf_result_t GxfContextCreate(gxf_context_t* context) {
  *context = new FakeContext;
  return GXF_SUCCESS;
}
extern "C" gxf_result_t GxfContextDestroy(gxf_context_t context) {
  auto* object = static_cast<FakeContext*>(context);
  object->scheduler.alive = false;
  std::this_thread::sleep_for(std::chrono::milliseconds(15));
  delete object;
  return GXF_SUCCESS;
}
extern "C" gxf_result_t GxfComponentTypeId(gxf_context_t, const char*, gxf_tid_t* tid) {
  *tid = {1, 2};
  return GXF_SUCCESS;
}
extern "C" gxf_result_t GxfEntityFindAll(gxf_context_t, uint64_t* count, gxf_uid_t* entities) {
  if (!entities) {
    *count = 1;
    return GXF_QUERY_NOT_ENOUGH_CAPACITY;
  }
  *count = 1;
  entities[0] = 1;
  return GXF_SUCCESS;
}
extern "C" gxf_result_t GxfComponentFind(gxf_context_t, gxf_uid_t, gxf_tid_t, const char*,
                                         int32_t* offset, gxf_uid_t* cid) {
  const int components = std::getenv("FIXTURE_MULTIPLE") ? 2 : 1;
  if (*offset >= components)
    return GXF_ENTITY_NOT_FOUND;
  *cid = *offset + 1;
  return GXF_SUCCESS;
}
extern "C" gxf_result_t GxfComponentPointer(gxf_context_t c, gxf_uid_t, gxf_tid_t, void** pointer) {
  *pointer = &static_cast<FakeContext*>(c)->scheduler;
  return GXF_SUCCESS;
}
extern "C" gxf_result_t GxfGraphRunAsync(gxf_context_t) {
  return std::getenv("FIXTURE_RUN_FAILURE") ? GXF_FAILURE : GXF_SUCCESS;
}
extern "C" gxf_result_t fixture_deinitialize(void*) asm(
    "_ZN6nvidia3gxf19EventBasedScheduler12deinitializeEv");
extern "C" gxf_result_t fixture_deinitialize(void* scheduler) {
  static_cast<FakeScheduler*>(scheduler)->alive = false;
  std::this_thread::sleep_for(std::chrono::milliseconds(15));
  return GXF_SUCCESS;
}
extern "C" gxf_result_t GxfGraphWait(gxf_context_t context) {
  auto* object = static_cast<FakeContext*>(context);
  for (int i = 0; i < 15; ++i) {
    ++object->scheduler.ticks;
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  if (std::getenv("FIXTURE_WAIT_FAILURE")) {
    // Model internal deinitialization before an unsuccessful wait returns to the caller.
    fixture_deinitialize(&object->scheduler);
    return GXF_FAILURE;
  }
  return GXF_SUCCESS;
}
extern "C" gxf_result_t GxfGraphDeactivate(gxf_context_t context) {
  return fixture_deinitialize(&static_cast<FakeContext*>(context)->scheduler);
}

// Test reader linked instead of counter_reader.o. Abort on any read after teardown.
CounterSnapshot read_counters(void* pointer) {
  auto* scheduler = static_cast<FakeScheduler*>(pointer);
  if (!scheduler->alive.load())
    std::abort();
  CounterSnapshot snapshot{};
  snapshot.steal_attempts = scheduler->ticks.load();
  return snapshot;
}
size_t scheduler_size() {
  return sizeof(FakeScheduler);
}
size_t counters_offset(void*) {
  return 0;
}
size_t running_threads_offset(void*) {
  return 0;
}
