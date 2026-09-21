# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

# This experiment needs the SDK source tree, including the unchanged LtSgemm sources.
# It can be built by the SDK or independently against an installed Holoscan SDK.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(benchmark_ltsgemm LANGUAGES CXX CUDA)
else()
  enable_language(CUDA)
endif()

if(NOT TARGET holoscan::core)
  find_package(holoscan REQUIRED CONFIG)
endif()
find_package(CUDAToolkit REQUIRED)

add_subdirectory(cpp)
