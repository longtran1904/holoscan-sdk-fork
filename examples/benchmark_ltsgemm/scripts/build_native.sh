#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
example_dir="$(cd "${script_dir}/.." && pwd)"
cmake_bin="${example_dir}/benchmark-env/bin/cmake"
build_dir="${example_dir}/build-ltsgemm-holoscan"

if [[ ! -x "$cmake_bin" ]]; then
  echo "CMake not found at $cmake_bin" >&2
  echo "Set up examples/benchmark_ltsgemm/benchmark-env before building." >&2
  exit 1
fi

"$cmake_bin" -S "$example_dir" -B "$build_dir" \
  "-DCMAKE_PREFIX_PATH=${HOLOSCAN_PREFIX_PATH:-/opt/nvidia/holoscan}" \
  -DCMAKE_BUILD_TYPE=Release \
  "-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG -march=native -ftree-vectorize" \
  "-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG" \
  "-DCMAKE_CUDA_COMPILER=${CUDACXX:-/usr/local/cuda/bin/nvcc}" \
  "-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES:-native}" \
  -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

"$cmake_bin" --build "$build_dir" --target benchmark_ltsgemm_native -j "${BUILD_JOBS:-3}"
echo "Built: ${build_dir}/cpp/benchmark_ltsgemm_native"
