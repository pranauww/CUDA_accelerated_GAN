import numpy as np
import ctypes
import pycuda.driver as cuda
import pycuda.autoinit

# Import the test harness to access the loaded CUDA kernels
from test_kernels import CUDATestHarness

# Helper for activation types
ACTIVATION_KERNELS = {
    'relu': ('cuda_relu', 'cuda_relu_gradient'),
    'sigmoid': ('cuda_sigmoid', 'cuda_sigmoid_gradient'),
    'tanh': ('cuda_tanh', 'cuda_tanh_gradient'),
}

class LinearLayer:
    def __init__(self, in_features, out_features, harness: CUDATestHarness, dtype=np.float32):
        self.in_features = in_features
        self.out_features = out_features
        self.harness = harness
        self.dtype = dtype
        # Weight: (in_features, out_features), Bias: (out_features,)
        self.W = np.random.randn(in_features, out_features).astype(dtype) * np.sqrt(2. / in_features)
        self.b = np.zeros(out_features, dtype=dtype)
        # Allocate GPU memory
        self.W_gpu = harness.allocate_gpu_memory(in_features * out_features)
        self.b_gpu = harness.allocate_gpu_memory(out_features)
        # Copy to GPU
        harness.copy_to_gpu(np.ascontiguousarray(self.W.ravel(order='C')), self.W_gpu)
        harness.copy_to_gpu(np.ascontiguousarray(self.b.ravel(order='C')), self.b_gpu)
        # Gradients
        self.dW_gpu = harness.allocate_gpu_memory(in_features * out_features)
        self.db_gpu = harness.allocate_gpu_memory(out_features)

    def forward(self, x_gpu, batch_size):
        # x_gpu: (batch_size, in_features)
        # Output: (batch_size, out_features)
        out_gpu = self.harness.allocate_gpu_memory(batch_size * self.out_features)
        # GEMM: (B, O) = (B, I) x (I, O)
        self.harness.lib.cuda_gemm(
            ctypes.c_void_p(int(x_gpu)),
            ctypes.c_void_p(int(self.W_gpu)),
            ctypes.c_void_p(int(out_gpu)),
            ctypes.c_int(batch_size),
            ctypes.c_int(self.out_features),
            ctypes.c_int(self.in_features),
            ctypes.c_float(1.0),
            ctypes.c_float(0.0)
        )
        # Add bias (broadcast)
        # Launch a simple kernel for bias add (or do on CPU for now)
        # For now, copy to CPU, add bias, copy back (can optimize later)
        out_host = np.zeros((batch_size, self.out_features), dtype=self.dtype)
        self.harness.copy_from_gpu(out_gpu, out_host.ravel(order='C'))
        out_host += self.b  # broadcast
        self.harness.copy_to_gpu(np.ascontiguousarray(out_host.ravel(order='C')), out_gpu)
        return out_gpu

    def backward(self, x_gpu, grad_out_gpu, batch_size, grad_shape):
        # grad_shape: tuple, e.g. (batch_size, out_features)
        grad_out_host = np.zeros(grad_shape, dtype=self.dtype)
        self.harness.copy_from_gpu(grad_out_gpu, grad_out_host)
        db = grad_out_host.sum(axis=0)
        self.harness.copy_to_gpu(np.ascontiguousarray(db.ravel(order='C')), self.db_gpu)
        # Compute dW and grad_input as before
        x_T_gpu = self.harness.allocate_gpu_memory(self.in_features * batch_size)
        self.harness.lib.cuda_transpose(
            ctypes.c_void_p(int(x_gpu)),
            ctypes.c_void_p(int(x_T_gpu)),
            ctypes.c_int(batch_size),
            ctypes.c_int(self.in_features)
        )
        self.harness.lib.cuda_gemm(
            ctypes.c_void_p(int(x_T_gpu)),
            ctypes.c_void_p(int(grad_out_gpu)),
            ctypes.c_void_p(int(self.dW_gpu)),
            ctypes.c_int(self.in_features),
            ctypes.c_int(self.out_features),
            ctypes.c_int(batch_size),
            ctypes.c_float(1.0),
            ctypes.c_float(0.0)
        )
        W_T_gpu = self.harness.allocate_gpu_memory(self.in_features * self.out_features)
        self.harness.lib.cuda_transpose(
            ctypes.c_void_p(int(self.W_gpu)),
            ctypes.c_void_p(int(W_T_gpu)),
            ctypes.c_int(self.out_features),
            ctypes.c_int(self.in_features)
        )
        grad_input_gpu = self.harness.allocate_gpu_memory(batch_size * self.in_features)
        self.harness.lib.cuda_gemm(
            ctypes.c_void_p(int(grad_out_gpu)),
            ctypes.c_void_p(int(W_T_gpu)),
            ctypes.c_void_p(int(grad_input_gpu)),
            ctypes.c_int(batch_size),
            ctypes.c_int(self.in_features),
            ctypes.c_int(self.out_features),
            ctypes.c_float(1.0),
            ctypes.c_float(0.0)
        )
        return grad_input_gpu

    def update(self, lr, beta1, beta2, epsilon, t, mW_gpu, vW_gpu, mb_gpu, vb_gpu):
        # Update weights using Adam optimizer kernel
        size_W = self.in_features * self.out_features
        size_b = self.out_features
        self.harness.lib.cuda_adam_update(
            ctypes.c_void_p(int(self.W_gpu)),
            ctypes.c_void_p(int(self.dW_gpu)),
            ctypes.c_void_p(int(mW_gpu)),
            ctypes.c_void_p(int(vW_gpu)),
            ctypes.c_int(size_W),
            ctypes.c_float(lr),
            ctypes.c_float(beta1),
            ctypes.c_float(beta2),
            ctypes.c_float(epsilon),
            ctypes.c_int(t)
        )
        self.harness.lib.cuda_adam_update(
            ctypes.c_void_p(int(self.b_gpu)),
            ctypes.c_void_p(int(self.db_gpu)),
            ctypes.c_void_p(int(mb_gpu)),
            ctypes.c_void_p(int(vb_gpu)),
            ctypes.c_int(size_b),
            ctypes.c_float(lr),
            ctypes.c_float(beta1),
            ctypes.c_float(beta2),
            ctypes.c_float(epsilon),
            ctypes.c_int(t)
        )

class ActivationLayer:
    def __init__(self, activation, harness: CUDATestHarness):
        assert activation in ACTIVATION_KERNELS
        self.activation = activation
        self.harness = harness
        self.forward_kernel, self.backward_kernel = ACTIVATION_KERNELS[activation]

    def forward(self, x_gpu, size):
        # In-place activation
        getattr(self.harness.lib, self.forward_kernel)(
            ctypes.c_void_p(int(x_gpu)),
            ctypes.c_int(size)
        )
        return x_gpu

    def backward(self, x_gpu, grad_gpu, size):
        # In-place gradient
        getattr(self.harness.lib, self.backward_kernel)(
            ctypes.c_void_p(int(x_gpu)),
            ctypes.c_void_p(int(grad_gpu)),
            ctypes.c_int(size)
        )
        return grad_gpu

class BCELossLayer:
    def __init__(self, harness: CUDATestHarness):
        self.harness = harness

    def forward(self, pred_gpu, target_gpu, size):
        # Allocate GPU memory for loss
        loss_gpu = self.harness.allocate_gpu_memory(size)
        self.harness.lib.cuda_binary_cross_entropy(
            ctypes.c_void_p(int(pred_gpu)),
            ctypes.c_void_p(int(target_gpu)),
            ctypes.c_void_p(int(loss_gpu)),
            ctypes.c_int(size)
        )
        return loss_gpu

    def backward(self, pred_gpu, target_gpu, grad_gpu, size):
        # grad_gpu is the output gradient (usually ones for BCE)
        self.harness.lib.cuda_binary_cross_entropy_gradient(
            ctypes.c_void_p(int(pred_gpu)),
            ctypes.c_void_p(int(target_gpu)),
            ctypes.c_void_p(int(grad_gpu)),
            ctypes.c_int(size)
        )
        return grad_gpu

class Sequential:
    def __init__(self, layers):
        self.layers = layers
        self.layer_inputs = []  # Will store inputs to each layer during forward

    def forward(self, x_gpu, batch_size):
        self.layer_inputs = [x_gpu]  # Store input to the first layer
        current_features = None
        for layer in self.layers:
            if isinstance(layer, LinearLayer):
                x_gpu = layer.forward(x_gpu, batch_size)
                current_features = layer.out_features
            elif isinstance(layer, ActivationLayer):
                x_gpu = layer.forward(x_gpu, batch_size * current_features)
            self.layer_inputs.append(x_gpu)  # Store input to the next layer
        return x_gpu

    def backward(self, x_gpu, grad_gpu, batch_size):
        # Find the output features of the last LinearLayer
        current_features = None
        for layer in reversed(self.layers):
            if isinstance(layer, LinearLayer):
                current_features = layer.out_features
                break
        grad_shape = (batch_size, current_features)
        # Use stored layer inputs
        for i, layer in enumerate(reversed(self.layers)):
            layer_input = self.layer_inputs[-(i+2)]  # -1 is output, -2 is last input, etc.
            if isinstance(layer, LinearLayer):
                grad_gpu = layer.backward(layer_input, grad_gpu, batch_size, grad_shape)
                current_features = layer.in_features
                grad_shape = (batch_size, current_features)
            elif isinstance(layer, ActivationLayer):
                grad_gpu = layer.backward(layer_input, grad_gpu, grad_shape[0] * grad_shape[1])
        return grad_gpu 