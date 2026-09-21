/*
 * SPDX-FileCopyrightText: Copyright (c) 2020 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "sample_cublasLt_LtSgemm.h"
#include "helpers.h"

#include <chrono>
#include <cstdio>
#include <limits>

int main() {
  constexpr int n = 128;
  constexpr int k = 128;
  constexpr int maxM = 1 << 16;
  constexpr int nRepeats = 1000;

  int device = 0;
  cudaDeviceProp properties{};
  checkCudaStatus(cudaGetDevice(&device));
  checkCudaStatus(cudaGetDeviceProperties(&properties, device));
  std::printf("L2 cache size: %d bytes (%.2f MiB)\n",
              properties.l2CacheSize,
              properties.l2CacheSize / (1024.0 * 1024.0));

  const auto runBenchmark = [](int m, int n, int k, bool flushL2Cache) {
    // Include setup, copies, warmup, cache flushing, statistics, and resource cleanup.
    // TestBench::run synchronizes the output copies before returning.
    const auto start = std::chrono::steady_clock::now();
    LtSgemmBenchResult result{};
    {
      TestBench<float> props(CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, 2.0f, 0.0f);

      props.run([&props, &result, flushL2Cache] {
        result = LtSgemmBench(props.ltHandle,
                              props.transa,
                              props.transb,
                              props.m,
                              props.n,
                              props.k,
                              &props.alpha,
                              props.Adev,
                              props.lda,
                              props.Bdev,
                              props.ldb,
                              &props.beta,
                              props.Cdev,
                              props.ldc,
                              props.workspace,
                              props.workspaceSize,
                              nRepeats,
                              flushL2Cache);
      });
    }
    const double totalUs =
        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
    const double computePercent = totalUs > 0.0 ? 100.0 * result.totalGpuUs / totalUs
                                                : std::numeric_limits<double>::quiet_NaN();

    // Two FLOPs per multiply-add; averageUs is in microseconds.
    const double tflops = (2.0 * m * n * k) / (result.averageUs * 1e6);
    const double cvPercent = result.averageUs > 0.0f ? 100.0 * result.stddevUs / result.averageUs
                                                     : std::numeric_limits<double>::quiet_NaN();
    std::printf(
        "GEMM (M=%d, N=%d, K=%d, L2 flush=%s): %.3f ± %.3f us, "
        "min=%.3f us, median=%.3f us, P99=%.3f us, CV=%.3f%%, %.3f TFLOP/s\n",
        m,
        n,
        k,
        flushL2Cache ? "on" : "off",
        result.averageUs,
        result.stddevUs,
        result.minUs,
        result.medianUs,
        result.p99Us,
        cvPercent,
        tflops);
    std::printf("  Timed GEMM total: %.3f ms, wall time: %.3f ms, GEMM/wall: %.3f%%\n",
                result.totalGpuUs / 1000.0,
                totalUs / 1000.0,
                computePercent);
  };

  // Only one loop for Nsight Systems profiling:
  // runBenchmark(65536, n, k, true);

  // Benchmarking Loop: setting (1) matrix size, (2) flush L2 cache or not
  for (bool flushL2Cache : {false, true}) {
    for (int m = 128; m <= maxM; m *= 2) {
      runBenchmark(m, n, k, flushL2Cache);
    }
  }

  // for (bool flushL2Cache : {true}) {
  //   for (int size : {512, 1024, 2048, 4096, 8192}) {
  //     runBenchmark(size, size, size, flushL2Cache);
  //   }
  // }

  return 0;
}
