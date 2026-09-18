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

#include "sample_cublasLt_LtSgemm.h"
#include "helpers.h"

namespace {

// Own scratch resources, including partially initialized resources on errors.
struct BenchResources {
    void *cache = nullptr;
    std::vector<cudaEvent_t> starts, ends;

    ~BenchResources() {
        for (auto event : starts) if (event) cudaEventDestroy(event);
        for (auto event : ends) if (event) cudaEventDestroy(event);
        if (cache) cudaFree(cache);
    }
};

// f must enqueue its work on the default stream. Returns microseconds.
template <typename F>
float doBenchCuda(F f, int nRepeats) {
    if (nRepeats <= 0) {
        throw std::invalid_argument("repeats must be positive");
    }

    BenchResources resources;
    int device = 0;
    cudaDeviceProp properties{};
    checkCudaStatus(cudaGetDevice(&device));
    checkCudaStatus(cudaGetDeviceProperties(&properties, device));
    if (properties.l2CacheSize <= 0) {
        throw std::runtime_error("device does not report a positive L2 cache size");
    }
    const size_t cacheSize = static_cast<size_t>(properties.l2CacheSize);
    checkCudaStatus(cudaMalloc(&resources.cache, cacheSize));

    auto flushCache = [&] {
        // Cache eviction attempt, not an architectural guarantee of a cold L2.
        checkCudaStatus(cudaMemsetAsync(resources.cache, 0, cacheSize, 0));
    };

    resources.starts.resize(nRepeats, nullptr);
    resources.ends.resize(nRepeats, nullptr);
    for (int i = 0; i < nRepeats; ++i) {
        checkCudaStatus(cudaEventCreate(&resources.starts[i]));
        checkCudaStatus(cudaEventCreate(&resources.ends[i]));
    }
    checkCudaStatus(cudaDeviceSynchronize());

    for (int i = 0; i < nRepeats; ++i) {
        flushCache();
        checkCudaStatus(cudaEventRecord(resources.starts[i], 0));
        f();
        checkCudaStatus(cudaEventRecord(resources.ends[i], 0));
    }
    checkCudaStatus(cudaDeviceSynchronize());

    double totalMs = 0.0;
    for (int i = 0; i < nRepeats; ++i) {
        float elapsedMs = 0.0f;
        checkCudaStatus(cudaEventElapsedTime(&elapsedMs, resources.starts[i], resources.ends[i]));
        totalMs += elapsedMs;
    }
    // CUDA events report milliseconds; convert the average to microseconds.
    return static_cast<float>(1000.0 * totalMs / nRepeats);
}

}  // namespace

/// Sample wrapper executing single precision gemm with cublasLtMatmul, nearly a drop-in replacement for cublasSgemm,
/// with addition of the workspace to support split-K algorithms
///
/// pointer mode is always host, to change it configure the appropriate matmul descriptor attribute
/// matmul is not using cublas handle's configuration of math mode, here tensor ops are implicitly allowed; to change
/// this configure appropriate attribute in the preference handle
// Returns average GPU time in microseconds; descriptor/algorithm setup is excluded.
// Like the Python callback, repeated calls mutate C when beta is nonzero.
float LtSgemmBench(cublasLtHandle_t ltHandle,
             cublasOperation_t transa,
             cublasOperation_t transb,
             int m,
             int n,
             int k,
             const float *alpha, /* host pointer */
             const float *A,
             int lda,
             const float *B,
             int ldb,
             const float *beta, /* host pointer */
             float *C,
             int ldc,
             void *workspace,
             size_t workspaceSize,
             int nRepeats) {
    if (nRepeats <= 0) {
        throw std::invalid_argument("repeats must be positive");
    }
    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    cublasLtMatmulPreference_t preference = NULL;

    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};

    // create operation descriptor; see cublasLtMatmulDescAttributes_t for details about defaults; here we just need to
    // set the transforms for A and B
    checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    checkCublasStatus(
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    checkCublasStatus(
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    // create matrix descriptors, we are good with the details here so no need to set any extra attributes
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, transa == CUBLAS_OP_N ? m : k,
                                                 transa == CUBLAS_OP_N ? k : m, lda));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, transb == CUBLAS_OP_N ? k : n,
                                                 transb == CUBLAS_OP_N ? n : k, ldb));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, m, n, ldc));

    // create preference handle; here we could use extra attributes to disable tensor ops or to make sure algo selected
    // will work with badly aligned A, B, C; here for simplicity we just assume A,B,C are always well aligned (e.g.
    // directly come from cudaMalloc)
    checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
    checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                           &workspaceSize, sizeof(workspaceSize)));

    // we just need the best available heuristic to try and run matmul. There is no guarantee this will work, e.g. if A
    // is badly aligned, you can request more (e.g. 32) algos and try to run them one by one until something works
    checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1,
                                                     &heuristicResult, &returnedResults));

    if (returnedResults == 0) {
        checkCublasStatus(CUBLAS_STATUS_NOT_SUPPORTED);
    }

    const float averageUs = doBenchCuda([&] {
        checkCublasStatus(cublasLtMatmul(ltHandle, operationDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, C, Cdesc,
                                         &heuristicResult.algo, workspace, workspaceSize, 0));
    }, nRepeats);

    // descriptors are no longer needed as all GPU work was already enqueued
    if (preference) checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
    if (Cdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
    if (operationDesc) checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));
    return averageUs;
}
