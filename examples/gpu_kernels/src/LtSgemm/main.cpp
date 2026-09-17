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
    TestBench<float> props(CUBLAS_OP_N, CUBLAS_OP_N, 4, 4, 4, 2.0f, 0.0f);

    props.Ahost = {
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1
    };
     
    props.Bhost = {
      1, 2, 3, 4,
      5, 6, 7, 8,
      9, 10, 11, 12,
      13, 14, 15, 16
    };

    props.run([&props] {
        LtSgemm(props.ltHandle, props.transa, props.transb, props.m, props.n, props.k, &props.alpha, props.Adev,
                props.lda, props.Bdev, props.ldb, &props.beta, props.Cdev, props.ldc, props.workspace,
                props.workspaceSize);
    });

    // run() copies the result to the host and waits for completion.
    // cuBLAS stores matrices in column-major order.
    std::printf("Result C (%d x %d):\n", props.m, props.n);
    for (int row = 0; row < props.m; ++row) {
        for (int col = 0; col < props.n; ++col) {
            std::printf("%10.2f ", props.Chost[row + col * props.ldc]);
        }
        std::printf("\n");
    }

    return 0;
}
