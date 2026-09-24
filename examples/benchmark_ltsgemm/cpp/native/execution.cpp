/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "execution.hpp"

#include <array>
#include <cstddef>
#include <cstdio>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "helpers.h"
#include "sample_cublasLt_LtSgemm.h"
#include "support.hpp"

namespace ltsgemm::native {
namespace {

constexpr size_t kWorkspaceBytes = 4 * 1024 * 1024;

void check_status(cudaError_t status) {
  checkCudaStatus(status);
}
void check_status(cublasStatus_t status) {
  checkCublasStatus(status);
}

// Members acquire resources individually, so constructor failures unwind safely.
// Normal cleanup is checked; exception cleanup must not throw a second exception.
template <typename T, auto Destroy>
struct Resource {
  T value = nullptr;
  Resource() = default;
  Resource(const Resource&) = delete;
  Resource& operator=(const Resource&) = delete;
  ~Resource() {
    if (value)
      Destroy(value);
  }
  void close() {
    if (value) {
      check_status(Destroy(value));
      value = nullptr;
    }
  }
};
using Buffer = Resource<void*, cudaFree>;
using Event = Resource<cudaEvent_t, cudaEventDestroy>;
using Matrices = TestBench<float>;

double timed_device_synchronize() {
  const auto start = Clock::now();
  checkCudaStatus(cudaDeviceSynchronize());
  return elapsed_ms(start);
}

struct Matmul {
  Resource<cublasLtMatmulDesc_t, cublasLtMatmulDescDestroy> operation;
  Resource<cublasLtMatrixLayout_t, cublasLtMatrixLayoutDestroy> a, b, c;
  Resource<cublasLtMatmulPreference_t, cublasLtMatmulPreferenceDestroy> preference;
  cublasLtMatmulHeuristicResult_t heuristic{};

  void initialize(const Matrices& matrices) {
    checkCublasStatus(cublasLtMatmulDescCreate(&operation.value, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(
        operation.value, CUBLASLT_MATMUL_DESC_TRANSA, &matrices.transa, sizeof(matrices.transa)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(
        operation.value, CUBLASLT_MATMUL_DESC_TRANSB, &matrices.transb, sizeof(matrices.transb)));
    checkCublasStatus(
        cublasLtMatrixLayoutCreate(&a.value, CUDA_R_32F, matrices.m, matrices.k, matrices.lda));
    checkCublasStatus(
        cublasLtMatrixLayoutCreate(&b.value, CUDA_R_32F, matrices.k, matrices.n, matrices.ldb));
    checkCublasStatus(
        cublasLtMatrixLayoutCreate(&c.value, CUDA_R_32F, matrices.m, matrices.n, matrices.ldc));
    checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference.value));
    checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference.value,
                                                           CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                           &matrices.workspaceSize,
                                                           sizeof(matrices.workspaceSize)));
    int returned = 0;
    checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(matrices.ltHandle,
                                                     operation.value,
                                                     a.value,
                                                     b.value,
                                                     c.value,
                                                     c.value,
                                                     preference.value,
                                                     1,
                                                     &heuristic,
                                                     &returned));
    if (returned == 0)
      checkCublasStatus(CUBLAS_STATUS_NOT_SUPPORTED);
  }

  void launch(const Matrices& matrices) {
    checkCublasStatus(cublasLtMatmul(matrices.ltHandle,
                                     operation.value,
                                     &matrices.alpha,
                                     matrices.Adev,
                                     a.value,
                                     matrices.Bdev,
                                     b.value,
                                     &matrices.beta,
                                     matrices.Cdev,
                                     c.value,
                                     matrices.Cdev,
                                     c.value,
                                     &heuristic.algo,
                                     matrices.workspace,
                                     matrices.workspaceSize,
                                     matrices.stream));
  }

  void close() {
    preference.close();
    c.close();
    b.close();
    a.close();
    operation.close();
  }
};

double warmup() {
  Matrices matrices(CUBLAS_OP_N, CUBLAS_OP_N, 4096, 4096, 4096, 1.0f, 0.0f, kWorkspaceBytes);
  // Reuse the reference's single warmup GEMM, never its benchmark entrypoint or loop.
  matrices.run([&] {
    LtSgemm(matrices.ltHandle,
            matrices.transa,
            matrices.transb,
            matrices.m,
            matrices.n,
            matrices.k,
            &matrices.alpha,
            matrices.Adev,
            matrices.lda,
            matrices.Bdev,
            matrices.ldb,
            &matrices.beta,
            matrices.Cdev,
            matrices.ldc,
            matrices.workspace,
            matrices.workspaceSize);
  });
  return timed_device_synchronize();
}

}  // namespace

struct BenchmarkCase::State {
  const Clock::time_point start = Clock::now();
  const int m;
  const bool flush;
  const cudaStream_t stream;
  std::optional<Matrices> matrices;
  Matmul matmul;
  Buffer cache;
  size_t cache_size = 0;
  std::array<Event, kRepeats> starts, ends;
  int repetitions = 0;
  double warmup_sync_ms = 0;
  double setup_sync_ms = 0;
  double flush_sync_total_ms = 0;
  double midpoint_sync_ms = 0;

  State(int rows, bool flush_l2, cudaStream_t cuda_stream)
      : m(rows),
        flush(flush_l2),
        stream(cuda_stream),
        matrices(std::in_place, CUBLAS_OP_N, CUBLAS_OP_N, rows, 128, 128, 2.0f, 0.0f,
                 kWorkspaceBytes, false, false, false, cuda_stream) {
    matrices->copyDataToDevice();
    matmul.initialize(*matrices);
    warmup_sync_ms = warmup();
    if (flush) {
      int device = 0;
      cudaDeviceProp properties{};
      checkCudaStatus(cudaGetDevice(&device));
      checkCudaStatus(cudaGetDeviceProperties(&properties, device));
      if (properties.l2CacheSize <= 0)
        throw std::runtime_error("device does not report a positive L2 cache size");
      cache_size = static_cast<size_t>(properties.l2CacheSize);
      checkCudaStatus(cudaMalloc(&cache.value, cache_size));
    }
    for (int i = 0; i < kRepeats; ++i) {
      checkCudaStatus(cudaEventCreate(&starts[i].value));
      checkCudaStatus(cudaEventCreate(&ends[i].value));
    }
    setup_sync_ms = timed_device_synchronize();
  }

  void launch() {
    if (flush) {
      checkCudaStatus(cudaMemsetAsync(cache.value, 0, cache_size, 0));
      flush_sync_total_ms += timed_device_synchronize();
    }
    checkCudaStatus(cudaEventRecord(starts[repetitions].value, 0));
    matmul.launch(*matrices);
    checkCudaStatus(cudaEventRecord(ends[repetitions].value, 0));
    ++repetitions;
  }

  CaseResult finish() {
    const double final_sync_ms = timed_device_synchronize();
    std::vector<double> timings;
    timings.reserve(kRepeats);
    for (int i = 0; i < kRepeats; ++i) {
      float ms = 0;
      checkCudaStatus(cudaEventElapsedTime(&ms, starts[i].value, ends[i].value));
      timings.push_back(1000.0 * ms);
    }
    for (auto& event : starts)
      event.close();
    for (auto& event : ends)
      event.close();
    cache.close();
    matmul.close();
    matrices->copyDataFromDevice();
    matrices->streamSynchronize();
    matrices.reset();
    // Keep the sample allocation alive through cleanup, as in the original finish().
    // Aggregation and sample destruction both remain inside the case wall timer.
    const auto statistics = calculate_statistics(std::move(timings));
    const double wall_ms = elapsed_ms(start);
    const int flush_sync_count = flush ? repetitions : 0;
    std::printf(
        "CUDA sync CPU wait (M=%d, L2 flush=%s): warmup=%.3f ms, setup=%.3f ms, "
        "pre-GEMM flush sync=%d calls/%.3f ms total (%.3f ms avg), midpoint=%.3f ms, "
        "final=%.3f ms\n",
        m,
        flush ? "on" : "off",
        warmup_sync_ms,
        setup_sync_ms,
        flush_sync_count,
        flush_sync_total_ms,
        flush_sync_count ? flush_sync_total_ms / flush_sync_count : 0.0,
        midpoint_sync_ms,
        final_sync_ms);
    return {m, flush, statistics, wall_ms};
  }
};

DeviceInfo query_device_info() {
  int device = 0;
  cudaDeviceProp properties{};
  checkCudaStatus(cudaGetDevice(&device));
  checkCudaStatus(cudaGetDeviceProperties(&properties, device));
  return {properties.l2CacheSize};
}

BenchmarkCase::BenchmarkCase(int m, bool flush_l2, cudaStream_t stream)
    : state_(std::make_unique<State>(m, flush_l2, stream)) {}

BenchmarkCase::~BenchmarkCase() = default;

void BenchmarkCase::launch() {
  state_->launch();
}

bool BenchmarkCase::complete() const {
  return state_->repetitions == kRepeats;
}

CaseResult BenchmarkCase::finish() {
  return state_->finish();
}

}  // namespace ltsgemm::native
