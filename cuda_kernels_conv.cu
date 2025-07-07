#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// Convolutional operations for GAN
extern "C" {

// 2D Convolution forward pass
__global__ void conv2d_forward_kernel(
    const float* input,
    const float* weights,
    float* output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_height,
    const int in_width,
    const int kernel_size,
    const int stride,
    const int padding
) {
    int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * out_channels * out_height * out_width;
    
    if (idx >= total_outputs) return;
    
    // Calculate output position
    int b = idx / (out_channels * out_height * out_width);
    int oc = (idx % (out_channels * out_height * out_width)) / (out_height * out_width);
    int oh = (idx % (out_height * out_width)) / out_width;
    int ow = idx % out_width;
    
    float sum = 0.0f;
    
    // Convolution operation
    for (int ic = 0; ic < in_channels; ic++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int ih = oh * stride + kh - padding;
                int iw = ow * stride + kw - padding;
                
                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                    int input_idx = b * in_channels * in_height * in_width + 
                                   ic * in_height * in_width + 
                                   ih * in_width + iw;
                    int weight_idx = oc * in_channels * kernel_size * kernel_size + 
                                    ic * kernel_size * kernel_size + 
                                    kh * kernel_size + kw;
                    sum += input[input_idx] * weights[weight_idx];
                }
            }
        }
    }
    
    output[idx] = sum;
}

// Transpose convolution (for generator)
__global__ void conv2d_transpose_kernel(
    const float* input,
    const float* weights,
    float* output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_height,
    const int in_width,
    const int kernel_size,
    const int stride,
    const int padding
) {
    int out_height = (in_height - 1) * stride - 2 * padding + kernel_size;
    int out_width = (in_width - 1) * stride - 2 * padding + kernel_size;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * out_channels * out_height * out_width;
    
    if (idx >= total_outputs) return;
    
    // Calculate output position
    int b = idx / (out_channels * out_height * out_width);
    int oc = (idx % (out_channels * out_height * out_width)) / (out_height * out_width);
    int oh = (idx % (out_height * out_width)) / out_width;
    int ow = idx % out_width;
    
    float sum = 0.0f;
    
    // Transpose convolution operation
    for (int ic = 0; ic < in_channels; ic++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int ih = (oh + padding - kh) / stride;
                int iw = (ow + padding - kw) / stride;
                
                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                    int input_idx = b * in_channels * in_height * in_width + 
                                   ic * in_height * in_width + 
                                   ih * in_width + iw;
                    int weight_idx = ic * out_channels * kernel_size * kernel_size + 
                                    oc * kernel_size * kernel_size + 
                                    kh * kernel_size + kw;
                    sum += input[input_idx] * weights[weight_idx];
                }
            }
        }
    }
    
    output[idx] = sum;
}

// Batch normalization
__global__ void batch_norm_kernel(
    float* data,
    const float* gamma,
    const float* beta,
    const float* running_mean,
    const float* running_var,
    const int size,
    const float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    float normalized = (data[idx] - running_mean[idx]) / sqrt(running_var[idx] + eps);
    data[idx] = gamma[idx] * normalized + beta[idx];
}

// Existing kernels (for compatibility)
__global__ void gemm_kernel(
    const float* A,
    const float* B,
    float* C,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float beta
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

__global__ void relu_kernel(float* data, const int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

__global__ void sigmoid_kernel(float* data, const int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = 1.0f / (1.0f + expf(-data[idx]));
    }
}

__global__ void tanh_kernel(float* data, const int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = tanhf(data[idx]);
    }
}

__global__ void binary_cross_entropy_kernel(
    const float* predictions,
    const float* targets,
    float* losses,
    const int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float pred = fmaxf(fminf(predictions[idx], 1.0f - 1e-7f), 1e-7f);
        losses[idx] = -(targets[idx] * logf(pred) + (1.0f - targets[idx]) * logf(1.0f - pred));
    }
}

__global__ void binary_cross_entropy_backward_kernel(
    const float* predictions,
    const float* targets,
    float* gradients,
    const int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float pred = fmaxf(fminf(predictions[idx], 1.0f - 1e-7f), 1e-7f);
        gradients[idx] = (pred - targets[idx]) / (pred * (1.0f - pred));
    }
}

__global__ void adam_update_kernel(
    float* params,
    const float* gradients,
    float* m,
    float* v,
    const int size,
    const float lr,
    const float beta1,
    const float beta2,
    const float epsilon,
    const int t
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = gradients[idx];
        
        // Update biased first moment estimate
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
        
        // Update biased second raw moment estimate
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;
        
        // Compute bias-corrected first moment estimate
        float m_hat = m[idx] / (1.0f - powf(beta1, t));
        
        // Compute bias-corrected second raw moment estimate
        float v_hat = v[idx] / (1.0f - powf(beta2, t));
        
        // Update parameters
        params[idx] -= lr * m_hat / (sqrtf(v_hat) + epsilon);
    }
}

// Wrapper functions
__declspec(dllexport) void cuda_gemm(
    const void* A,
    const void* B,
    void* C,
    int M,
    int N,
    int K,
    float alpha,
    float beta
) {
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
    
    gemm_kernel<<<gridDim, blockDim>>>(
        (const float*)A,
        (const float*)B,
        (float*)C,
        M, N, K, alpha, beta
    );
}

__declspec(dllexport) void cuda_binary_cross_entropy(
    const void* predictions,
    const void* targets,
    void* losses,
    int size
) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    binary_cross_entropy_kernel<<<gridSize, blockSize>>>(
        (const float*)predictions,
        (const float*)targets,
        (float*)losses,
        size
    );
}

__declspec(dllexport) void cuda_binary_cross_entropy_backward(
    const void* predictions,
    const void* targets,
    void* gradients,
    int size
) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    binary_cross_entropy_backward_kernel<<<gridSize, blockSize>>>(
        (const float*)predictions,
        (const float*)targets,
        (float*)gradients,
        size
    );
}

__declspec(dllexport) void cuda_adam_update(
    void* params,
    const void* gradients,
    void* m,
    void* v,
    int size,
    float lr,
    float beta1,
    float beta2,
    float epsilon,
    int t
) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    adam_update_kernel<<<gridSize, blockSize>>>(
        (float*)params,
        (const float*)gradients,
        (float*)m,
        (float*)v,
        size, lr, beta1, beta2, epsilon, t
    );
}

__declspec(dllexport) void cuda_relu(void* data, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    relu_kernel<<<gridSize, blockSize>>>((float*)data, size);
}

__declspec(dllexport) void cuda_sigmoid(void* data, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    sigmoid_kernel<<<gridSize, blockSize>>>((float*)data, size);
}

__declspec(dllexport) void cuda_tanh(void* data, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    tanh_kernel<<<gridSize, blockSize>>>((float*)data, size);
}

__declspec(dllexport) void cuda_conv2d_forward(
    const void* input,
    const void* weights,
    void* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int kernel_size,
    int stride,
    int padding
) {
    int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    int total_outputs = batch_size * out_channels * out_height * out_width;
    
    int blockSize = 256;
    int gridSize = (total_outputs + blockSize - 1) / blockSize;
    
    conv2d_forward_kernel<<<gridSize, blockSize>>>(
        (const float*)input,
        (const float*)weights,
        (float*)output,
        batch_size, in_channels, out_channels,
        in_height, in_width, kernel_size, stride, padding
    );
}

__declspec(dllexport) void cuda_conv2d_transpose(
    const void* input,
    const void* weights,
    void* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int kernel_size,
    int stride,
    int padding
) {
    int out_height = (in_height - 1) * stride - 2 * padding + kernel_size;
    int out_width = (in_width - 1) * stride - 2 * padding + kernel_size;
    int total_outputs = batch_size * out_channels * out_height * out_width;
    
    int blockSize = 256;
    int gridSize = (total_outputs + blockSize - 1) / blockSize;
    
    conv2d_transpose_kernel<<<gridSize, blockSize>>>(
        (const float*)input,
        (const float*)weights,
        (float*)output,
        batch_size, in_channels, out_channels,
        in_height, in_width, kernel_size, stride, padding
    );
}

__declspec(dllexport) void cuda_batch_norm(
    void* data,
    const void* gamma,
    const void* beta,
    const void* running_mean,
    const void* running_var,
    int size,
    float eps
) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    batch_norm_kernel<<<gridSize, blockSize>>>(
        (float*)data,
        (const float*)gamma,
        (const float*)beta,
        (const float*)running_mean,
        (const float*)running_var,
        size, eps
    );
}

} 