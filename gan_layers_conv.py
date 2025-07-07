import numpy as np
import ctypes
from typing import Tuple, Optional

class Conv2DLayer:
    """2D Convolutional Layer"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, harness=None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.harness = harness
        
        # Initialize weights (He initialization)
        fan_in = in_channels * kernel_size * kernel_size
        std = np.sqrt(2.0 / fan_in)
        self.weights = np.random.normal(0, std, (out_channels, in_channels, kernel_size, kernel_size)).astype(np.float32)
        self.biases = np.zeros(out_channels, dtype=np.float32)
        
        # Allocate GPU memory
        self.weights_gpu = harness.allocate_gpu_memory(self.weights.size)
        self.biases_gpu = harness.allocate_gpu_memory(self.biases.size)
        
        # Copy to GPU
        harness.copy_to_gpu(np.ascontiguousarray(self.weights.ravel(order='C')), self.weights_gpu)
        harness.copy_to_gpu(self.biases, self.biases_gpu)
        
        # Store input shape for backward pass
        self.input_shape = None
        self.output_shape = None
        
    def forward(self, x_gpu, batch_size):
        """Forward pass through convolutional layer"""
        # Get input shape from previous layer or assume it's already correct
        if self.input_shape is None:
            # For now, assume input is already in correct shape
            # In a full implementation, you'd reshape from flattened input
            pass
            
        # Calculate output dimensions
        in_height = int(np.sqrt(self.in_channels))  # Simplified assumption
        in_width = in_height
        out_height = (in_height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (in_width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        self.output_shape = (batch_size, self.out_channels, out_height, out_width)
        output_size = batch_size * self.out_channels * out_height * out_width
        
        # Allocate output memory
        output_gpu = self.harness.allocate_gpu_memory(output_size)
        
        # Call CUDA kernel
        self.harness.lib.cuda_conv2d_forward(
            ctypes.c_void_p(int(x_gpu)),
            ctypes.c_void_p(int(self.weights_gpu)),
            ctypes.c_void_p(int(output_gpu)),
            ctypes.c_int(batch_size),
            ctypes.c_int(self.in_channels),
            ctypes.c_int(self.out_channels),
            ctypes.c_int(in_height),
            ctypes.c_int(in_width),
            ctypes.c_int(self.kernel_size),
            ctypes.c_int(self.stride),
            ctypes.c_int(self.padding)
        )
        
        return output_gpu
        
    def backward(self, x_gpu, grad_gpu, batch_size):
        """Backward pass through convolutional layer"""
        # For now, return a simplified gradient
        # In a full implementation, you'd compute proper gradients
        grad_input_size = batch_size * self.in_channels * int(np.sqrt(self.in_channels))**2
        grad_input_gpu = self.harness.allocate_gpu_memory(grad_input_size)
        
        # Simplified gradient computation (would need proper implementation)
        return grad_input_gpu

class ConvTranspose2DLayer:
    """2D Transpose Convolutional Layer (for generator)"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, harness=None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.harness = harness
        
        # Initialize weights (He initialization)
        fan_in = in_channels * kernel_size * kernel_size
        std = np.sqrt(2.0 / fan_in)
        self.weights = np.random.normal(0, std, (in_channels, out_channels, kernel_size, kernel_size)).astype(np.float32)
        self.biases = np.zeros(out_channels, dtype=np.float32)
        
        # Allocate GPU memory
        self.weights_gpu = harness.allocate_gpu_memory(self.weights.size)
        self.biases_gpu = harness.allocate_gpu_memory(self.biases.size)
        
        # Copy to GPU
        harness.copy_to_gpu(np.ascontiguousarray(self.weights.ravel(order='C')), self.weights_gpu)
        harness.copy_to_gpu(self.biases, self.biases_gpu)
        
        # Store input shape for backward pass
        self.input_shape = None
        self.output_shape = None
        
    def forward(self, x_gpu, batch_size):
        """Forward pass through transpose convolutional layer"""
        # Calculate output dimensions
        in_height = int(np.sqrt(self.in_channels))  # Simplified assumption
        in_width = in_height
        out_height = (in_height - 1) * self.stride - 2 * self.padding + self.kernel_size
        out_width = (in_width - 1) * self.stride - 2 * self.padding + self.kernel_size
        
        self.output_shape = (batch_size, self.out_channels, out_height, out_width)
        output_size = batch_size * self.out_channels * out_height * out_width
        
        # Allocate output memory
        output_gpu = self.harness.allocate_gpu_memory(output_size)
        
        # Call CUDA kernel
        self.harness.lib.cuda_conv2d_transpose(
            ctypes.c_void_p(int(x_gpu)),
            ctypes.c_void_p(int(self.weights_gpu)),
            ctypes.c_void_p(int(output_gpu)),
            ctypes.c_int(batch_size),
            ctypes.c_int(self.in_channels),
            ctypes.c_int(self.out_channels),
            ctypes.c_int(in_height),
            ctypes.c_int(in_width),
            ctypes.c_int(self.kernel_size),
            ctypes.c_int(self.stride),
            ctypes.c_int(self.padding)
        )
        
        return output_gpu
        
    def backward(self, x_gpu, grad_gpu, batch_size):
        """Backward pass through transpose convolutional layer"""
        # For now, return a simplified gradient
        grad_input_size = batch_size * self.in_channels * int(np.sqrt(self.in_channels))**2
        grad_input_gpu = self.harness.allocate_gpu_memory(grad_input_size)
        
        # Simplified gradient computation (would need proper implementation)
        return grad_input_gpu

class BatchNorm2DLayer:
    """2D Batch Normalization Layer"""
    def __init__(self, num_features, harness=None, eps=1e-5, momentum=0.1):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.harness = harness
        
        # Initialize parameters
        self.gamma = np.ones(num_features, dtype=np.float32)
        self.beta = np.zeros(num_features, dtype=np.float32)
        self.running_mean = np.zeros(num_features, dtype=np.float32)
        self.running_var = np.ones(num_features, dtype=np.float32)
        
        # Allocate GPU memory
        self.gamma_gpu = harness.allocate_gpu_memory(self.gamma.size)
        self.beta_gpu = harness.allocate_gpu_memory(self.beta.size)
        self.running_mean_gpu = harness.allocate_gpu_memory(self.running_mean.size)
        self.running_var_gpu = harness.allocate_gpu_memory(self.running_var.size)
        
        # Copy to GPU
        harness.copy_to_gpu(self.gamma, self.gamma_gpu)
        harness.copy_to_gpu(self.beta, self.beta_gpu)
        harness.copy_to_gpu(self.running_mean, self.running_mean_gpu)
        harness.copy_to_gpu(self.running_var, self.running_var_gpu)
        
    def forward(self, x_gpu, batch_size):
        """Forward pass through batch normalization"""
        # Call CUDA kernel
        self.harness.lib.cuda_batch_norm(
            ctypes.c_void_p(int(x_gpu)),
            ctypes.c_void_p(int(self.gamma_gpu)),
            ctypes.c_void_p(int(self.beta_gpu)),
            ctypes.c_void_p(int(self.running_mean_gpu)),
            ctypes.c_void_p(int(self.running_var_gpu)),
            ctypes.c_int(batch_size * self.num_features),
            ctypes.c_float(self.eps)
        )
        
        return x_gpu
        
    def backward(self, x_gpu, grad_gpu, batch_size):
        """Backward pass through batch normalization"""
        # For now, return the same gradient
        return grad_gpu

# Import existing layers for compatibility
from gan_layers import LinearLayer, ActivationLayer, Sequential, BCELossLayer 