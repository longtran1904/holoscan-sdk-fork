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

#include <cublasLt.h>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <memory>
#include <vector>

#include "sample_cublasLt_LtSgemm.h"
#include "helpers.h"

namespace {

// Own scratch resources, including partially initialized resources on errors.
struct BenchResources {
    void *cache = nullptr;
    cudaStream_t stream = nullptr;
    cublasLtHandle_t handle = nullptr;
    std::vector<cudaEvent_t> starts, ends;

    ~BenchResources() {
        for (auto event : starts) if (event) cudaEventDestroy(event);
        for (auto event : ends) if (event) cudaEventDestroy(event);
        if (cache) cudaFree(cache);
        if (handle) cublasLtDestroy(handle);
        if (stream) cudaStreamDestroy(stream);
    }
};

// f must enqueue its work on the shared stream. Returns microseconds.
template <typename F, typename Factory>
LtSgemmBenchResult doBenchCuda(F f, Factory makeBench, int nRepeats, bool flushL2Cache,
                                cudaStream_t stream) {
  if (nRepeats <= 0) {
    throw std::invalid_argument("repeats must be positive");
  }

  // Warm up the GPU with a large GEMM before collecting any timings.
  {
    TestBench<float> warmup(CUBLAS_OP_N, CUBLAS_OP_N, 4096, 4096, 4096,
                            1.0f, 0.0f, 4 * 1024 * 1024, 1, false, false, false, stream);
    warmup.run([&] {
      // LtSgemm uses stream 0; explicitly order it with the shared copy stream.
      warmup.streamSynchronize();
      LtSgemm(warmup.ltHandle,
              warmup.transa,
              warmup.transb,
              warmup.m,
              warmup.n,
              warmup.k,
              &warmup.alpha,
              warmup.Adev,
              warmup.lda,
              warmup.Bdev,
              warmup.ldb,
              &warmup.beta,
              warmup.Cdev,
              warmup.ldc,
              warmup.workspace,
              warmup.workspaceSize);
      checkCudaStatus(cudaDeviceSynchronize());
    });
    checkCudaStatus(cudaDeviceSynchronize());
  }

  BenchResources resources;
  size_t cacheSize = 0;
  if (flushL2Cache) {
    int device = 0;
    cudaDeviceProp properties{};
    checkCudaStatus(cudaGetDevice(&device));
    checkCudaStatus(cudaGetDeviceProperties(&properties, device));
    if (properties.l2CacheSize <= 0) {
      throw std::runtime_error("device does not report a positive L2 cache size");
    }
    cacheSize = static_cast<size_t>(properties.l2CacheSize);
    checkCudaStatus(cudaMalloc(&resources.cache, cacheSize));
  }

  auto flushCache = [&] {
    // Cache eviction attempt, not an architectural guarantee of a cold L2.
    checkCudaStatus(cudaMemsetAsync(resources.cache, 0, cacheSize, stream));
    checkCudaStatus(cudaDeviceSynchronize());
  };

  resources.starts.resize(nRepeats, nullptr);
  resources.ends.resize(nRepeats, nullptr);
  for (int i = 0; i < nRepeats; ++i) {
    checkCudaStatus(cudaEventCreate(&resources.starts[i]));
    checkCudaStatus(cudaEventCreate(&resources.ends[i]));
  }
  checkCudaStatus(cudaDeviceSynchronize());

  for (int i = 0; i < nRepeats; ++i) {
    // Construction allocates fresh host/device buffers and a cuBLASLt handle.
    // run() copies inputs and outputs; destruction frees buffers but keeps the shared stream.
    auto props = makeBench();
    props->run([&] {
      // Complete input copies before timing GEMM on the same shared stream.
      props->streamSynchronize();
      if (flushL2Cache)
        flushCache();
      checkCudaStatus(cudaEventRecord(resources.starts[i], stream));
      f(*props);
      checkCudaStatus(cudaEventRecord(resources.ends[i], stream));
      checkCudaStatus(cudaEventSynchronize(resources.ends[i]));
    });
  }
  checkCudaStatus(cudaDeviceSynchronize());

  double meanUs = 0.0;
  double totalGpuUs = 0.0;
  double squaredDeviations = 0.0;
  std::vector<double> timingsUs;
  timingsUs.reserve(nRepeats);
  for (int i = 0; i < nRepeats; ++i) {
    float elapsedMs = 0.0f;
    checkCudaStatus(cudaEventElapsedTime(&elapsedMs, resources.starts[i], resources.ends[i]));
    // Welford's algorithm avoids subtracting nearly equal squared values.
    const double elapsedUs = 1000.0 * elapsedMs;
    totalGpuUs += elapsedUs;
    timingsUs.push_back(elapsedUs);
    const double delta = elapsedUs - meanUs;
    meanUs += delta / (i + 1);
    squaredDeviations += delta * (elapsedUs - meanUs);
  }
  std::sort(timingsUs.begin(), timingsUs.end());
  const auto percentile = [&timingsUs](double p) {
    const double index = p * (timingsUs.size() - 1);
    const size_t lower = static_cast<size_t>(std::floor(index));
    const size_t upper = static_cast<size_t>(std::ceil(index));
    return static_cast<float>(timingsUs[lower] +
                              (timingsUs[upper] - timingsUs[lower]) * (index - lower));
  };
  return {static_cast<float>(meanUs),
          static_cast<float>(std::sqrt(squaredDeviations / nRepeats)),
          static_cast<float>(timingsUs.front()),
          percentile(0.50),
          percentile(0.99),
          totalGpuUs};
}

}  // namespace

// Fresh TestBench inputs (including C) are initialized on every iteration.
// Event statistics exclude allocation, copies, descriptor setup, and cleanup.
LtSgemmBenchResult LtSgemmBenchFreshAlloc(cublasOperation_t transa,
                                         cublasOperation_t transb,
                                         int m, int n, int k,
                                         float alpha, float beta,
                                         size_t workspaceSize,
                                         int nRepeats, bool flushL2Cache) {
  if (m <= 0 || n <= 0 || k <= 0) {
    throw std::invalid_argument("matrix dimensions must be positive");
  }
  const int lda = transa == CUBLAS_OP_N ? m : k;
  const int ldb = transb == CUBLAS_OP_N ? k : n;
  const int ldc = m;
  BenchResources setup;
  if (nRepeats <= 0) {
    throw std::invalid_argument("repeats must be positive");
  }
  checkCudaStatus(cudaStreamCreate(&setup.stream));
  checkCublasStatus(cublasLtCreate(&setup.handle));
  cublasLtMatmulDesc_t operationDesc = NULL;
  cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
  cublasLtMatmulPreference_t preference = NULL;

  int returnedResults = 0;
  cublasLtMatmulHeuristicResult_t heuristicResult = {};

  // create operation descriptor; see cublasLtMatmulDescAttributes_t for details about defaults;
  // here we just need to set the transforms for A and B
  checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  checkCublasStatus(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

  // create matrix descriptors, we are good with the details here so no need to set any extra
  // attributes
  checkCublasStatus(cublasLtMatrixLayoutCreate(
      &Adesc, CUDA_R_32F, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
  checkCublasStatus(cublasLtMatrixLayoutCreate(
      &Bdesc, CUDA_R_32F, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, m, n, ldc));

  // create preference handle; here we could use extra attributes to disable tensor ops or to make
  // sure algo selected will work with badly aligned A, B, C; here for simplicity we just assume
  // A,B,C are always well aligned (e.g. directly come from cudaMalloc)
  checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
  checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

  // we just need the best available heuristic to try and run matmul. There is no guarantee this
  // will work, e.g. if A is badly aligned, you can request more (e.g. 32) algos and try to run them
  // one by one until something works
  checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(setup.handle,
                                                   operationDesc,
                                                   Adesc,
                                                   Bdesc,
                                                   Cdesc,
                                                   Cdesc,
                                                   preference,
                                                   1,
                                                   &heuristicResult,
                                                   &returnedResults));

  if (returnedResults == 0) {
    checkCublasStatus(CUBLAS_STATUS_NOT_SUPPORTED);
  }

  const LtSgemmBenchResult result = doBenchCuda(
      [&](TestBench<float>& props) {
        checkCublasStatus(cublasLtMatmul(props.ltHandle,
                                         operationDesc,
                                         &props.alpha,
                                         props.Adev,
                                         Adesc,
                                         props.Bdev,
                                         Bdesc,
                                         &props.beta,
                                         props.Cdev,
                                         Cdesc,
                                         props.Cdev,
                                         Cdesc,
                                         &heuristicResult.algo,
                                         props.workspace,
                                         workspaceSize,
                                         props.stream));
      },
      [&] {
        return std::unique_ptr<TestBench<float>>(new TestBench<float>(
            transa, transb, m, n, k, alpha, beta, workspaceSize,
            1, false, false, false, setup.stream));
      },
      nRepeats,
      flushL2Cache,
      setup.stream);

  // descriptors are no longer needed as all GPU work was already enqueued
  if (preference)
    checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
  if (Cdesc)
    checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
  if (Bdesc)
    checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
  if (Adesc)
    checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
  if (operationDesc)
    checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));
  return result;
}

// Standalone counterpart to main.cpp, using the same profiling workload.
int main() {
  const auto start = std::chrono::steady_clock::now();
  const auto result = LtSgemmBenchFreshAlloc(
      CUBLAS_OP_N, CUBLAS_OP_N, 65536, 128, 128,
      2.0f, 0.0f, 4 * 1024 * 1024, 1000, false);
  const double totalUs = std::chrono::duration<double, std::micro>(
      std::chrono::steady_clock::now() - start).count();
  std::printf("Fresh TestBench GEMM (M=65536, N=128, K=128, L2 flush=on): "
              "%.3f +/- %.3f us, min=%.3f us, median=%.3f us, P99=%.3f us\n",
              result.averageUs, result.stddevUs, result.minUs,
              result.medianUs, result.p99Us);
  std::printf("  Timed GEMM total: %.3f ms, wall time: %.3f ms, GEMM/wall: %.3f%%\n",
              result.totalGpuUs / 1000.0, totalUs / 1000.0,
              100.0 * result.totalGpuUs / totalUs);
  return 0;
}
