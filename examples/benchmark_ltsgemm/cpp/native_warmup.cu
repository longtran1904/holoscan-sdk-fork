/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>

#include "helpers.h"
#include "sample_cublasLt_LtSgemm.h"

// Reuse the warmup source verbatim, with scoped ownership around its descriptor API
// calls. The original function only destroys descriptors on its successful path.
// These forwarding callables also reclaim them if a later API call throws.
// Callable objects suppress argument-dependent lookup of the global cuBLAS APIs.
namespace native_warmup {
struct Descriptors {
  cublasLtMatmulDesc_t operation = nullptr;
  std::array<cublasLtMatrixLayout_t, 3> layouts{};
  cublasLtMatmulPreference_t preference = nullptr;
  size_t next_layout = 0;

  ~Descriptors() {
    if (preference)
      ::cublasLtMatmulPreferenceDestroy(preference);
    for (auto layout : layouts)
      if (layout)
        ::cublasLtMatrixLayoutDestroy(layout);
    if (operation)
      ::cublasLtMatmulDescDestroy(operation);
  }
};
thread_local Descriptors* active = nullptr;

const auto cublasLtMatmulDescCreate = [](cublasLtMatmulDesc_t* output, cublasComputeType_t compute,
                                         cudaDataType_t scale) {
  const auto status = ::cublasLtMatmulDescCreate(output, compute, scale);
  if (status == CUBLAS_STATUS_SUCCESS)
    active->operation = *output;
  return status;
};
const auto cublasLtMatmulDescDestroy = [](cublasLtMatmulDesc_t descriptor) {
  const auto status = ::cublasLtMatmulDescDestroy(descriptor);
  if (status == CUBLAS_STATUS_SUCCESS)
    active->operation = nullptr;
  return status;
};
const auto cublasLtMatrixLayoutCreate = [](cublasLtMatrixLayout_t* output, cudaDataType type,
                                           uint64_t rows, uint64_t cols, int64_t ld) {
  auto& owner = active->layouts.at(active->next_layout++);
  const auto status = ::cublasLtMatrixLayoutCreate(output, type, rows, cols, ld);
  if (status == CUBLAS_STATUS_SUCCESS)
    owner = *output;
  return status;
};
const auto cublasLtMatrixLayoutDestroy = [](cublasLtMatrixLayout_t descriptor) {
  const auto status = ::cublasLtMatrixLayoutDestroy(descriptor);
  if (status == CUBLAS_STATUS_SUCCESS) {
    for (auto& layout : active->layouts)
      if (layout == descriptor)
        layout = nullptr;
  }
  return status;
};
const auto cublasLtMatmulPreferenceCreate = [](cublasLtMatmulPreference_t* output) {
  const auto status = ::cublasLtMatmulPreferenceCreate(output);
  if (status == CUBLAS_STATUS_SUCCESS)
    active->preference = *output;
  return status;
};
const auto cublasLtMatmulPreferenceDestroy = [](cublasLtMatmulPreference_t descriptor) {
  const auto status = ::cublasLtMatmulPreferenceDestroy(descriptor);
  if (status == CUBLAS_STATUS_SUCCESS)
    active->preference = nullptr;
  return status;
};

// Headers were included above, outside this namespace. Only the function body is
// brought into this namespace, where its create/destroy calls use the owners above.
#include "sample_cublasLt_LtSgemm.cu"
}  // namespace native_warmup

void LtSgemm(cublasLtHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m,
             int n, int k, const float* alpha, const float* a, int lda, const float* b, int ldb,
             const float* beta, float* c, int ldc, void* workspace, size_t workspace_size) {
  native_warmup::Descriptors descriptors;
  struct Activation {
    native_warmup::Descriptors* previous = native_warmup::active;
    explicit Activation(native_warmup::Descriptors* current) { native_warmup::active = current; }
    ~Activation() { native_warmup::active = previous; }
  } activation(&descriptors);
  native_warmup::LtSgemm(handle,
                         transa,
                         transb,
                         m,
                         n,
                         k,
                         alpha,
                         a,
                         lda,
                         b,
                         ldb,
                         beta,
                         c,
                         ldc,
                         workspace,
                         workspace_size);
}
