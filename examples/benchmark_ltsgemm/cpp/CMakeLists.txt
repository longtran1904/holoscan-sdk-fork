# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set(ltsgemm_reference_dir "${CMAKE_CURRENT_SOURCE_DIR}/../../gpu_kernels/src/LtSgemm")

# Keep the rename local to the reference entrypoint and retain its original C++11 setting.
# In particular, none of holoscan::core's compile requirements reach these reference objects.
add_library(ltsgemm_reference_main OBJECT "${ltsgemm_reference_dir}/main.cpp")
target_compile_definitions(ltsgemm_reference_main PRIVATE
  main=run_original_ltsgemm_benchmark
)
set_target_properties(ltsgemm_reference_main PROPERTIES
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS ON
)

add_library(ltsgemm_warmup_cuda OBJECT
  "${ltsgemm_reference_dir}/sample_cublasLt_LtSgemm.cu"
)
add_library(ltsgemm_reference_cuda OBJECT
  "${ltsgemm_reference_dir}/benchmark_cublasLt_LtSgemm.cu"
)
add_library(ltsgemm_native_warmup_cuda OBJECT native_warmup.cu)
target_include_directories(ltsgemm_native_warmup_cuda PRIVATE "${ltsgemm_reference_dir}")
# The original mixed CXX/CUDA target derives CUDA's standard from its C++11 setting
# unless an explicit CMAKE_CUDA_STANDARD is supplied. Splitting it into object targets
# otherwise selects the compiler's newer default and changes the comparison.
if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set_target_properties(ltsgemm_reference_cuda ltsgemm_warmup_cuda ltsgemm_native_warmup_cuda
    PROPERTIES CUDA_STANDARD 11)
endif()

foreach(reference_target IN ITEMS
    ltsgemm_reference_main ltsgemm_reference_cuda ltsgemm_warmup_cuda ltsgemm_native_warmup_cuda)
  target_include_directories(${reference_target} PRIVATE
    "${ltsgemm_reference_dir}/../Common"
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  )
endforeach()

# Configure architecture, stream mode, and any explicit CUDA standard with the same
# compiler/toolkit and flags as the independently built original.
add_executable(benchmark_ltsgemm
  benchmark_ltsgemm.cpp
  $<TARGET_OBJECTS:ltsgemm_reference_main>
  $<TARGET_OBJECTS:ltsgemm_reference_cuda>
  $<TARGET_OBJECTS:ltsgemm_warmup_cuda>
)
set_target_properties(benchmark_ltsgemm PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
)
target_link_libraries(benchmark_ltsgemm PRIVATE
  holoscan::core
  CUDA::cudart
  CUDA::cublasLt
  ${CMAKE_DL_LIBS}
)

# Native scheduling owns its case state; only the warmup implementation is shared.
add_executable(benchmark_ltsgemm_native
  benchmark_ltsgemm_native.cpp
  native/application.cpp
  native/execution.cpp
  native/support.cpp
  $<TARGET_OBJECTS:ltsgemm_native_warmup_cuda>
)
set_target_properties(benchmark_ltsgemm_native PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
)
target_include_directories(benchmark_ltsgemm_native PRIVATE
  "${ltsgemm_reference_dir}"
  "${ltsgemm_reference_dir}/../Common"
)
target_link_libraries(benchmark_ltsgemm_native PRIVATE
  holoscan::core CUDA::cudart CUDA::cublasLt ${CMAKE_DL_LIBS}
)

if(HOLOSCAN_BUILD_TESTS)
  add_test(NAME EXAMPLE_CPP_BENCHMARK_LTSGEMM_HELP_TEST
    COMMAND benchmark_ltsgemm --help
  )
  set_tests_properties(EXAMPLE_CPP_BENCHMARK_LTSGEMM_HELP_TEST PROPERTIES
    PASS_REGULAR_EXPRESSION "Usage:"
    TIMEOUT 10
  )
  add_test(NAME EXAMPLE_CPP_BENCHMARK_LTSGEMM_NATIVE_HELP_TEST
    COMMAND benchmark_ltsgemm_native --help
  )
  set_tests_properties(EXAMPLE_CPP_BENCHMARK_LTSGEMM_NATIVE_HELP_TEST PROPERTIES
    PASS_REGULAR_EXPRESSION "Usage:"
    TIMEOUT 10
  )
endif()
