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

#include <cstdio>

int main() {
  constexpr int n = 128;
  constexpr int k = 128;
  constexpr int maxM = 1 << 16;
  constexpr int nRepeats = 1000;

  for (int m = 128; m <= maxM; m *= 2) {
    TestBench<float> props(CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, 2.0f, 0.0f);

    float averageUs = 0.0f;
    props.run([&props, &averageUs] {
      averageUs = LtSgemmBench(props.ltHandle,
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
                               nRepeats);
    });

    // Two FLOPs per multiply-add; averageUs is in microseconds.
    const double tflops = (2.0 * m * n * k) / (averageUs * 1e6);
    std::printf("GEMM (M=%d, N=%d, K=%d): %.3f us, %.3f TFLOP/s\n",
                m, n, k, averageUs, tflops);
  }

  return 0;
}
