/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef EXAMPLES_BENCHMARK_LTSGEMM_CPP_NATIVE_EXECUTION_HPP
#define EXAMPLES_BENCHMARK_LTSGEMM_CPP_NATIVE_EXECUTION_HPP

#include <cuda_runtime_api.h>
#include <memory>

#include "timing.hpp"
#include "types.hpp"

namespace ltsgemm::native {

DeviceInfo query_device_info();

class BenchmarkCase {
 public:
  BenchmarkCase(int m, bool flush_l2, cudaStream_t stream);
  ~BenchmarkCase();
  BenchmarkCase(const BenchmarkCase&) = delete;
  BenchmarkCase& operator=(const BenchmarkCase&) = delete;

  void launch();
  bool complete() const;
  CaseResult finish();

 private:
  struct State;
  std::unique_ptr<State> state_;
};

}  // namespace ltsgemm::native

#endif  // EXAMPLES_BENCHMARK_LTSGEMM_CPP_NATIVE_EXECUTION_HPP
