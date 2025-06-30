// Compatibility fix for CUDA 10.2 with Visual Studio 2022
#ifdef _MSC_VER
#if _MSC_VER >= 1920
#define _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS
#endif
#endif

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <math.h>

#ifdef _WIN32
#define EXPORT_API extern "C" __declspec(dllexport)
#else
#define EXPORT_API extern "C"
#endif

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Matrix multiplication kernel (GEMM)
// Computes C = alpha * A * B + beta * C
// A: (M, K), B: (K, N), C: (M, N)
__global__ void gemm_kernel(
    const float* A, const float* B, float* C,
    const int M, const int N, const int K,
    const float alpha, const float beta
) {
    // Calculate global indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Bounds check
    if (row >= M || col >= N) return;
    
    // Compute dot product
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }
    
    // Write result
    int idx = row * N + col;
    C[idx] = alpha * sum + beta * C[idx];
}

// Batched matrix multiplication kernel
// A: (batch_size, M, K), B: (batch_size, K, N), C: (batch_size, M, N)
__global__ void batched_gemm_kernel(
    const float* A, const float* B, float* C,
    const int batch_size, const int M, const int N, const int K,
    const float alpha, const float beta
) {
    // Calculate global indices
    int batch = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Bounds check
    if (batch >= batch_size || row >= M || col >= N) return;
    
    // Compute offsets for this batch
    int A_offset = batch * M * K;
    int B_offset = batch * K * N;
    int C_offset = batch * M * N;
    
    // Compute dot product
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A[A_offset + row * K + k] * B[B_offset + k * N + col];
    }
    
    // Write result
    int idx = C_offset + row * N + col;
    C[idx] = alpha * sum + beta * C[idx];
}

// ReLU activation kernel
__global__ void relu_kernel(float* data, const int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

// ReLU gradient kernel
__global__ void relu_gradient_kernel(const float* input, float* gradient, const int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        gradient[idx] = (input[idx] > 0.0f) ? gradient[idx] : 0.0f;
    }
}

// Sigmoid activation kernel
__global__ void sigmoid_kernel(float* data, const int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = 1.0f / (1.0f + expf(-data[idx]));
    }
}

// Sigmoid gradient kernel
__global__ void sigmoid_gradient_kernel(const float* output, float* gradient, const int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float sigmoid_val = output[idx];
        gradient[idx] *= sigmoid_val * (1.0f - sigmoid_val);
    }
}

// Tanh activation kernel
__global__ void tanh_kernel(float* data, const int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = tanhf(data[idx]);
    }
}

// Tanh gradient kernel
__global__ void tanh_gradient_kernel(const float* output, float* gradient, const int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float tanh_val = output[idx];
        gradient[idx] *= (1.0f - tanh_val * tanh_val);
    }
}

// Binary cross-entropy loss kernel
__global__ void binary_cross_entropy_kernel(
    const float* predictions, const float* targets, float* loss,
    const int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float pred = fmaxf(fminf(predictions[idx], 1.0f - 1e-7f), 1e-7f);
        float target = targets[idx];
        loss[idx] = -(target * logf(pred) + (1.0f - target) * logf(1.0f - pred));
    }
}

// Binary cross-entropy gradient kernel
__global__ void binary_cross_entropy_gradient_kernel(
    const float* predictions, const float* targets, float* gradient,
    const int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float pred = fmaxf(fminf(predictions[idx], 1.0f - 1e-7f), 1e-7f);
        float target = targets[idx];
        gradient[idx] = (pred - target) / (pred * (1.0f - pred));
    }
}

// Matrix transpose kernel for gradient computation
__global__ void transpose_kernel(
    const float* input, float* output,
    const int rows, const int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < cols && idy < rows) {
        output[idx * rows + idy] = input[idy * cols + idx];
    }
}

// Adam optimizer kernel
// Updates parameters using Adam algorithm
__global__ void adam_update_kernel(
    float* params, float* gradients,
    float* m, float* v,  // First and second moment estimates
    const int size,
    const float learning_rate, const float beta1, const float beta2,
    const float epsilon, const int t  // Current timestep
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Update biased first moment estimate
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * gradients[idx];
        
        // Update biased second raw moment estimate
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * gradients[idx] * gradients[idx];
        
        // Compute bias-corrected first moment estimate
        float m_hat = m[idx] / (1.0f - powf(beta1, t));
        
        // Compute bias-corrected second raw moment estimate
        float v_hat = v[idx] / (1.0f - powf(beta2, t));
        
        // Update parameters
        params[idx] -= learning_rate * m_hat / (sqrtf(v_hat) + epsilon);
    }
}

// Utility function to get optimal block dimensions
void get_optimal_block_dims(int rows, int cols, dim3& block_dim) {
    // For matrix operations, use 2D blocks
    block_dim.x = min(32, cols);
    block_dim.y = min(32, rows);
    block_dim.z = 1;
}

// Utility function to get optimal grid dimensions
void get_optimal_grid_dims(int rows, int cols, int batch_size, dim3& grid_dim, const dim3& block_dim) {
    grid_dim.x = (cols + block_dim.x - 1) / block_dim.x;
    grid_dim.y = (rows + block_dim.y - 1) / block_dim.y;
    grid_dim.z = batch_size;
}

// C-style wrapper functions for Python integration
// Exported wrapper functions

EXPORT_API void cuda_gemm(
    const float* A, const float* B, float* C,
    const int M, const int N, const int K,
    const float alpha, const float beta
) {
    dim3 block_dim, grid_dim;
    get_optimal_block_dims(M, N, block_dim);
    get_optimal_grid_dims(M, N, 1, grid_dim, block_dim);
    
    gemm_kernel<<<grid_dim, block_dim>>>(
        A, B, C, M, N, K, alpha, beta
    );
    CUDA_CHECK(cudaGetLastError());
}

EXPORT_API void cuda_batched_gemm(
    const float* A, const float* B, float* C,
    const int batch_size, const int M, const int N, const int K,
    const float alpha, const float beta
) {
    dim3 block_dim, grid_dim;
    get_optimal_block_dims(M, N, block_dim);
    get_optimal_grid_dims(M, N, batch_size, grid_dim, block_dim);
    
    batched_gemm_kernel<<<grid_dim, block_dim>>>(
        A, B, C, batch_size, M, N, K, alpha, beta
    );
    CUDA_CHECK(cudaGetLastError());
}

EXPORT_API void cuda_relu(float* data, const int size) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    
    relu_kernel<<<grid_size, block_size>>>(data, size);
    CUDA_CHECK(cudaGetLastError());
}

EXPORT_API void cuda_relu_gradient(const float* input, float* gradient, const int size) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    
    relu_gradient_kernel<<<grid_size, block_size>>>(input, gradient, size);
    CUDA_CHECK(cudaGetLastError());
}

EXPORT_API void cuda_sigmoid(float* data, const int size) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    
    sigmoid_kernel<<<grid_size, block_size>>>(data, size);
    CUDA_CHECK(cudaGetLastError());
}

EXPORT_API void cuda_sigmoid_gradient(const float* output, float* gradient, const int size) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    
    sigmoid_gradient_kernel<<<grid_size, block_size>>>(output, gradient, size);
    CUDA_CHECK(cudaGetLastError());
}

EXPORT_API void cuda_tanh(float* data, const int size) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    
    tanh_kernel<<<grid_size, block_size>>>(data, size);
    CUDA_CHECK(cudaGetLastError());
}

EXPORT_API void cuda_tanh_gradient(const float* output, float* gradient, const int size) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    
    tanh_gradient_kernel<<<grid_size, block_size>>>(output, gradient, size);
    CUDA_CHECK(cudaGetLastError());
}

EXPORT_API void cuda_binary_cross_entropy(
    const float* predictions, const float* targets, float* loss,
    const int size
) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    
    binary_cross_entropy_kernel<<<grid_size, block_size>>>(
        predictions, targets, loss, size
    );
    CUDA_CHECK(cudaGetLastError());
}

EXPORT_API void cuda_binary_cross_entropy_gradient(
    const float* predictions, const float* targets, float* gradient,
    const int size
) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    
    binary_cross_entropy_gradient_kernel<<<grid_size, block_size>>>(
        predictions, targets, gradient, size
    );
    CUDA_CHECK(cudaGetLastError());
}

EXPORT_API void cuda_transpose(
    const float* input, float* output,
    const int rows, const int cols
) {
    dim3 block_dim(32, 32);
    dim3 grid_dim((cols + 31) / 32, (rows + 31) / 32);
    
    transpose_kernel<<<grid_dim, block_dim>>>(input, output, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

EXPORT_API void cuda_adam_update(
    float* params, float* gradients,
    float* m, float* v,
    const int size,
    const float learning_rate, const float beta1, const float beta2,
    const float epsilon, const int t
) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    
    adam_update_kernel<<<grid_size, block_size>>>(
        params, gradients, m, v, size,
        learning_rate, beta1, beta2, epsilon, t
    );
    CUDA_CHECK(cudaGetLastError());
} 