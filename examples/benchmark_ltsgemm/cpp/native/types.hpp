/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef EXAMPLES_BENCHMARK_LTSGEMM_CPP_NATIVE_TYPES_HPP
#define EXAMPLES_BENCHMARK_LTSGEMM_CPP_NATIVE_TYPES_HPP

#include <exception>
#include <mutex>

namespace ltsgemm::native {

constexpr int kCases = 1;
constexpr int kRepeats = 1000;

struct RunOptions {
  int gemms_per_tick = 1;
  int operator_count = 1;
};

struct DeviceInfo {
  int l2_cache_bytes;
};

struct TimingStatistics {
  float average_us;
  float stddev_us;
  float min_us;
  float median_us;
  float p99_us;
  double total_gpu_us;
};

struct CaseResult {
  int m;
  bool flush_l2;
  TimingStatistics statistics;
  double wall_ms;
};

struct RunResult {
  std::mutex counter_mutex;
  int completed_cases = 0;
  int timed_gemms = 0;
  int compute_calls = 0;
  int return_code = 1;
  double app_run_wall_ms = 0;
  std::exception_ptr error;
};

}  // namespace ltsgemm::native

#endif  // EXAMPLES_BENCHMARK_LTSGEMM_CPP_NATIVE_TYPES_HPP
