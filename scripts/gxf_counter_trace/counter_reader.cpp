/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// This is the ONLY translation unit compiled with GCC -fno-access-control.
// The class declaration is unchanged; the compiler calculates the member addresses.
#include "counter_reader.hpp"

#include <atomic>
#include "gxf/std/event_based_scheduler.hpp"

using Scheduler = nvidia::gxf::EventBasedScheduler;
static_assert(std::atomic<uint64_t>::is_always_lock_free);
static_assert(std::atomic<uint8_t>::is_always_lock_free);

CounterSnapshot read_counters(void* scheduler) {
  const auto* object = static_cast<const Scheduler*>(scheduler);
  CounterSnapshot snapshot{};
#define LOAD(name) snapshot.name = object->perf_counters_.name.load(std::memory_order_relaxed);
  EBS_TRACE_COUNTERS(LOAD)
#undef LOAD
  snapshot.running_threads = object->running_threads_.load(std::memory_order_relaxed);
  return snapshot;
}

size_t scheduler_size() {
  return sizeof(Scheduler);
}
size_t counters_offset(void* scheduler) {
  return reinterpret_cast<char*>(&static_cast<Scheduler*>(scheduler)->perf_counters_) -
         static_cast<char*>(scheduler);
}
size_t running_threads_offset(void* scheduler) {
  return reinterpret_cast<char*>(&static_cast<Scheduler*>(scheduler)->running_threads_) -
         static_cast<char*>(scheduler);
}
