/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef EXAMPLES_BENCHMARK_LTSGEMM_CPP_NATIVE_TIMING_HPP
#define EXAMPLES_BENCHMARK_LTSGEMM_CPP_NATIVE_TIMING_HPP

#include <chrono>

namespace ltsgemm::native {

using Clock = std::chrono::steady_clock;

inline double elapsed_ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

}  // namespace ltsgemm::native

#endif  // EXAMPLES_BENCHMARK_LTSGEMM_CPP_NATIVE_TIMING_HPP
