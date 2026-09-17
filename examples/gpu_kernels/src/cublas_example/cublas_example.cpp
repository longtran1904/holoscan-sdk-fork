#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Helper macro for error checking
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUBLAS(call) { \
    cublasStatus_t stat = call; \
    if (stat != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS Error code: " << stat << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

int main() {
    // Matrix dimensions (M x K) * (K x N) = (M x N)
    const int M = 3;
    const int N = 4;
    const int K = 2;

    // Host matrices (Column-Major representation)
    float h_A[M * K] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}; 
    float h_B[K * N] = {1.0f, 0.0f, 2.0f, 1.0f, 3.0f, 2.0f, 4.0f, 3.0f};
    float h_C[M * N] = {0.0f};

    // Device pointers
    float *d_A, *d_B, *d_C;

    // 1. Allocate GPU memory
    CHECK_CUDA(cudaMalloc((void**)&d_A, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_B, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_C, M * N * sizeof(float)));

    // 2. Copy data from Host to Device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));

    // 3. Initialize cuBLAS context
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // Scalars for the operation: C = alpha*A*B + beta*C
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // 4. Execute Matrix Multiplication
    // Parameters: handle, operation A, operation B, M, N, K, alpha, A, leading_dim_A, B, leading_dim_B, beta, C, leading_dim_C
    CHECK_CUBLAS(cublasSgemm(handle, 
                             CUBLAS_OP_N, CUBLAS_OP_N, 
                             M, N, K, 
                             &alpha, 
                             d_A, M, 
                             d_B, K, 
                             &beta, 
                             d_C, M));

    // 5. Synchronize device and copy results back
    CHECK_CUDA(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Print a sample element to verify output
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << "C[" << i << "][" << j << "] = " << h_C[i + j * M] << std::endl;
        }
    }
    // std::cout << "Top-left result element C[0][0]: " << h_C[0] << std::endl;

    // 6. Resource Cleanup
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}
