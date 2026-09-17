/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <cstdint>

// Names intentionally match the installed GXF header (and its final statistics log).
#define EBS_TRACE_COUNTERS(X)        \
  X(steal_attempts)                  \
  X(steal_successes)                 \
  X(worker_waitforjob_calls)         \
  X(worker_waitforjob_total_us)      \
  X(worker_execute_calls)            \
  X(worker_execute_total_us)         \
  X(worker_postcheck_fastpath_ready) \
  X(dispatcher_events_dispatched)

struct CounterSnapshot {
#define FIELD(name) uint64_t name;
  EBS_TRACE_COUNTERS(FIELD)
#undef FIELD
  uint64_t running_threads;
};

CounterSnapshot read_counters(void* scheduler);
size_t scheduler_size();
size_t counters_offset(void* scheduler);
size_t running_threads_offset(void* scheduler);
