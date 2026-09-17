cmake_minimum_required(VERSION 3.18)
project(gpu_kernels LANGUAGES CXX)

find_package(CUDAToolkit REQUIRED)

add_executable(cublas_example src/cublas_example/cublas_example.cpp)
target_compile_features(cublas_example PRIVATE cxx_std_17)
target_link_libraries(cublas_example PRIVATE CUDA::cudart CUDA::cublas CUDA::cublasLt)

add_subdirectory(src/LtSgemm)
