import numpy as np
import ctypes
import os
import subprocess
import sys
from typing import Tuple, Optional

class CUDATestHarness:
    def __init__(self):
        self.lib = None
        self.compiled = False
        
    def compile_kernels(self) -> bool:
        """Compile the CUDA kernels into a shared library"""
        try:
            # Check if nvcc is available
            result = subprocess.run(['nvcc', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print("Error: nvcc not found. Please install CUDA toolkit.")
                return False
            
            # Compile the CUDA kernels
            compile_cmd = [
                'nvcc', '-shared', '-Xcompiler', '-fPIC',
                '-o', 'cuda_kernels.so',
                'cuda_kernels.cu'
            ]
            
            print("Compiling CUDA kernels...")
            result = subprocess.run(compile_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Compilation failed: {result.stderr}")
                return False
            
            # Load the compiled library
            self.lib = ctypes.CDLL('./cuda_kernels.so')
            self.compiled = True
            print("CUDA kernels compiled successfully!")
            return True
            
        except Exception as e:
            print(f"Error during compilation: {e}")
            return False
    
    def allocate_gpu_memory(self, size: int) -> int:
        """Allocate GPU memory and return pointer"""
        ptr = ctypes.c_void_p()
        cuda_malloc = self.lib.cudaMalloc
        cuda_malloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        cuda_malloc.restype = ctypes.c_int
        
        result = cuda_malloc(ctypes.byref(ptr), size * ctypes.sizeof(ctypes.c_float))
        if result != 0:
            raise RuntimeError(f"CUDA memory allocation failed: {result}")
        return ptr.value
    
    def copy_to_gpu(self, host_data: np.ndarray, gpu_ptr: int):
        """Copy data from host to GPU"""
        cuda_memcpy = self.lib.cudaMemcpy
        cuda_memcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, 
                               ctypes.c_size_t, ctypes.c_int]
        cuda_memcpy.restype = ctypes.c_int
        
        result = cuda_memcpy(
            gpu_ptr, 
            host_data.ctypes.data,
            host_data.nbytes,
            1  # cudaMemcpyHostToDevice
        )
        if result != 0:
            raise RuntimeError(f"CUDA copy to device failed: {result}")
    
    def copy_from_gpu(self, gpu_ptr: int, host_data: np.ndarray):
        """Copy data from GPU to host"""
        cuda_memcpy = self.lib.cudaMemcpy
        cuda_memcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, 
                               ctypes.c_size_t, ctypes.c_int]
        cuda_memcpy.restype = ctypes.c_int
        
        result = cuda_memcpy(
            host_data.ctypes.data,
            gpu_ptr,
            host_data.nbytes,
            2  # cudaMemcpyDeviceToHost
        )
        if result != 0:
            raise RuntimeError(f"CUDA copy from device failed: {result}")
    
    def free_gpu_memory(self, gpu_ptr: int):
        """Free GPU memory"""
        cuda_free = self.lib.cudaFree
        cuda_free.argtypes = [ctypes.c_void_p]
        cuda_free.restype = ctypes.c_int
        
        result = cuda_free(gpu_ptr)
        if result != 0:
            raise RuntimeError(f"CUDA memory free failed: {result}")
    
    def test_gemm(self, M: int = 64, N: int = 64, K: int = 64) -> bool:
        """Test matrix multiplication kernel"""
        print(f"\nTesting GEMM: {M}x{K} @ {K}x{N}")
        
        # Create test matrices
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        C_numpy = np.zeros((M, N), dtype=np.float32)
        
        # NumPy reference
        C_ref = np.dot(A, B)
        
        # CUDA implementation
        try:
            # Allocate GPU memory
            A_gpu = self.allocate_gpu_memory(M * K)
            B_gpu = self.allocate_gpu_memory(K * N)
            C_gpu = self.allocate_gpu_memory(M * N)
            
            # Copy data to GPU
            self.copy_to_gpu(A.flatten(), A_gpu)
            self.copy_to_gpu(B.flatten(), B_gpu)
            self.copy_to_gpu(C_numpy.flatten(), C_gpu)
            
            # Call CUDA kernel
            self.lib.cuda_gemm(
                ctypes.c_void_p(A_gpu),
                ctypes.c_void_p(B_gpu),
                ctypes.c_void_p(C_gpu),
                ctypes.c_int(M),
                ctypes.c_int(N),
                ctypes.c_int(K),
                ctypes.c_float(1.0),
                ctypes.c_float(0.0)
            )
            
            # Copy result back
            C_cuda = np.zeros((M, N), dtype=np.float32)
            self.copy_from_gpu(C_gpu, C_cuda.flatten())
            C_cuda = C_cuda.reshape(M, N)
            
            # Free GPU memory
            self.free_gpu_memory(A_gpu)
            self.free_gpu_memory(B_gpu)
            self.free_gpu_memory(C_gpu)
            
            # Compare results
            error = np.abs(C_cuda - C_ref).max()
            print(f"Max error: {error:.2e}")
            
            if error < 1e-3:
                print("✓ GEMM test passed!")
                return True
            else:
                print("✗ GEMM test failed!")
                return False
                
        except Exception as e:
            print(f"✗ GEMM test failed with error: {e}")
            return False
    
    def test_activation_functions(self, size: int = 1024) -> bool:
        """Test activation function kernels"""
        print(f"\nTesting activation functions with size {size}")
        
        # Create test data
        x = np.random.randn(size).astype(np.float32)
        
        # Test ReLU
        print("Testing ReLU...")
        try:
            x_gpu = self.allocate_gpu_memory(size)
            self.copy_to_gpu(x, x_gpu)
            
            self.lib.cuda_relu(
                ctypes.c_void_p(x_gpu),
                ctypes.c_int(size)
            )
            
            x_relu_cuda = np.zeros(size, dtype=np.float32)
            self.copy_from_gpu(x_gpu, x_relu_cuda)
            self.free_gpu_memory(x_gpu)
            
            x_relu_ref = np.maximum(0, x)
            error = np.abs(x_relu_cuda - x_relu_ref).max()
            print(f"ReLU max error: {error:.2e}")
            
            if error > 1e-6:
                print("✗ ReLU test failed!")
                return False
                
        except Exception as e:
            print(f"✗ ReLU test failed with error: {e}")
            return False
        
        # Test Sigmoid
        print("Testing Sigmoid...")
        try:
            x_gpu = self.allocate_gpu_memory(size)
            self.copy_to_gpu(x, x_gpu)
            
            self.lib.cuda_sigmoid(
                ctypes.c_void_p(x_gpu),
                ctypes.c_int(size)
            )
            
            x_sigmoid_cuda = np.zeros(size, dtype=np.float32)
            self.copy_from_gpu(x_gpu, x_sigmoid_cuda)
            self.free_gpu_memory(x_gpu)
            
            x_sigmoid_ref = 1.0 / (1.0 + np.exp(-x))
            error = np.abs(x_sigmoid_cuda - x_sigmoid_ref).max()
            print(f"Sigmoid max error: {error:.2e}")
            
            if error > 1e-5:
                print("✗ Sigmoid test failed!")
                return False
                
        except Exception as e:
            print(f"✗ Sigmoid test failed with error: {e}")
            return False
        
        # Test Tanh
        print("Testing Tanh...")
        try:
            x_gpu = self.allocate_gpu_memory(size)
            self.copy_to_gpu(x, x_gpu)
            
            self.lib.cuda_tanh(
                ctypes.c_void_p(x_gpu),
                ctypes.c_int(size)
            )
            
            x_tanh_cuda = np.zeros(size, dtype=np.float32)
            self.copy_from_gpu(x_gpu, x_tanh_cuda)
            self.free_gpu_memory(x_gpu)
            
            x_tanh_ref = np.tanh(x)
            error = np.abs(x_tanh_cuda - x_tanh_ref).max()
            print(f"Tanh max error: {error:.2e}")
            
            if error > 1e-5:
                print("✗ Tanh test failed!")
                return False
                
        except Exception as e:
            print(f"✗ Tanh test failed with error: {e}")
            return False
        
        print("✓ All activation function tests passed!")
        return True
    
    def test_binary_cross_entropy(self, size: int = 1024) -> bool:
        """Test binary cross-entropy loss and gradient"""
        print(f"\nTesting Binary Cross-Entropy with size {size}")
        
        # Create test data
        predictions = np.random.uniform(0.1, 0.9, size).astype(np.float32)
        targets = np.random.randint(0, 2, size).astype(np.float32)
        
        # Test loss computation
        print("Testing BCE loss...")
        try:
            pred_gpu = self.allocate_gpu_memory(size)
            target_gpu = self.allocate_gpu_memory(size)
            loss_gpu = self.allocate_gpu_memory(size)
            
            self.copy_to_gpu(predictions, pred_gpu)
            self.copy_to_gpu(targets, target_gpu)
            
            self.lib.cuda_binary_cross_entropy(
                ctypes.c_void_p(pred_gpu),
                ctypes.c_void_p(target_gpu),
                ctypes.c_void_p(loss_gpu),
                ctypes.c_int(size)
            )
            
            loss_cuda = np.zeros(size, dtype=np.float32)
            self.copy_from_gpu(loss_gpu, loss_cuda)
            
            # NumPy reference
            epsilon = 1e-7
            pred_clipped = np.clip(predictions, epsilon, 1.0 - epsilon)
            loss_ref = -(targets * np.log(pred_clipped) + 
                        (1.0 - targets) * np.log(1.0 - pred_clipped))
            
            error = np.abs(loss_cuda - loss_ref).max()
            print(f"BCE loss max error: {error:.2e}")
            
            if error > 1e-5:
                print("✗ BCE loss test failed!")
                return False
            
            # Test gradient computation
            print("Testing BCE gradient...")
            grad_gpu = self.allocate_gpu_memory(size)
            self.copy_to_gpu(np.ones(size, dtype=np.float32), grad_gpu)
            
            self.lib.cuda_binary_cross_entropy_gradient(
                ctypes.c_void_p(pred_gpu),
                ctypes.c_void_p(target_gpu),
                ctypes.c_void_p(grad_gpu),
                ctypes.c_int(size)
            )
            
            grad_cuda = np.zeros(size, dtype=np.float32)
            self.copy_from_gpu(grad_gpu, grad_cuda)
            
            # NumPy reference for gradient
            grad_ref = (pred_clipped - targets) / (pred_clipped * (1.0 - pred_clipped))
            
            error = np.abs(grad_cuda - grad_ref).max()
            print(f"BCE gradient max error: {error:.2e}")
            
            # Free GPU memory
            self.free_gpu_memory(pred_gpu)
            self.free_gpu_memory(target_gpu)
            self.free_gpu_memory(loss_gpu)
            self.free_gpu_memory(grad_gpu)
            
            if error > 1e-5:
                print("✗ BCE gradient test failed!")
                return False
                
        except Exception as e:
            print(f"✗ BCE test failed with error: {e}")
            return False
        
        print("✓ Binary Cross-Entropy test passed!")
        return True
    
    def test_adam_optimizer(self, size: int = 1024) -> bool:
        """Test Adam optimizer kernel"""
        print(f"\nTesting Adam optimizer with size {size}")
        
        # Create test data
        params = np.random.randn(size).astype(np.float32)
        gradients = np.random.randn(size).astype(np.float32)
        m = np.zeros(size, dtype=np.float32)
        v = np.zeros(size, dtype=np.float32)
        
        # Adam hyperparameters
        learning_rate = 0.001
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        t = 1
        
        # NumPy reference
        m_ref = beta1 * m + (1.0 - beta1) * gradients
        v_ref = beta2 * v + (1.0 - beta2) * gradients * gradients
        m_hat = m_ref / (1.0 - beta1 ** t)
        v_hat = v_ref / (1.0 - beta2 ** t)
        params_ref = params - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        
        # CUDA implementation
        try:
            params_gpu = self.allocate_gpu_memory(size)
            grad_gpu = self.allocate_gpu_memory(size)
            m_gpu = self.allocate_gpu_memory(size)
            v_gpu = self.allocate_gpu_memory(size)
            
            self.copy_to_gpu(params, params_gpu)
            self.copy_to_gpu(gradients, grad_gpu)
            self.copy_to_gpu(m, m_gpu)
            self.copy_to_gpu(v, v_gpu)
            
            self.lib.cuda_adam_update(
                ctypes.c_void_p(params_gpu),
                ctypes.c_void_p(grad_gpu),
                ctypes.c_void_p(m_gpu),
                ctypes.c_void_p(v_gpu),
                ctypes.c_int(size),
                ctypes.c_float(learning_rate),
                ctypes.c_float(beta1),
                ctypes.c_float(beta2),
                ctypes.c_float(epsilon),
                ctypes.c_int(t)
            )
            
            params_cuda = np.zeros(size, dtype=np.float32)
            self.copy_from_gpu(params_gpu, params_cuda)
            
            # Free GPU memory
            self.free_gpu_memory(params_gpu)
            self.free_gpu_memory(grad_gpu)
            self.free_gpu_memory(m_gpu)
            self.free_gpu_memory(v_gpu)
            
            # Compare results
            error = np.abs(params_cuda - params_ref).max()
            print(f"Adam max error: {error:.2e}")
            
            if error < 1e-5:
                print("✓ Adam optimizer test passed!")
                return True
            else:
                print("✗ Adam optimizer test failed!")
                return False
                
        except Exception as e:
            print(f"✗ Adam test failed with error: {e}")
            return False
    
    def run_all_tests(self) -> bool:
        """Run all kernel tests"""
        if not self.compiled:
            print("Error: CUDA kernels not compiled. Run compile_kernels() first.")
            return False
        
        print("Running CUDA kernel tests...")
        
        tests = [
            ("GEMM", lambda: self.test_gemm(64, 64, 64)),
            ("Activation Functions", lambda: self.test_activation_functions(1024)),
            ("Binary Cross-Entropy", lambda: self.test_binary_cross_entropy(1024)),
            ("Adam Optimizer", lambda: self.test_adam_optimizer(1024))
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            print(f"\n{'='*50}")
            print(f"Running {test_name} test...")
            print(f"{'='*50}")
            
            try:
                if test_func():
                    passed += 1
                else:
                    print(f"❌ {test_name} test failed!")
            except Exception as e:
                print(f"❌ {test_name} test failed with exception: {e}")
        
        print(f"\n{'='*50}")
        print(f"Test Results: {passed}/{total} tests passed")
        print(f"{'='*50}")
        
        return passed == total

def main():
    """Main test function"""
    print("CUDA GAN Kernel Test Harness")
    print("=" * 50)
    
    # Check if CUDA is available
    try:
        import torch
        if torch.cuda.is_available():
            print(f"CUDA is available. Found {torch.cuda.device_count()} device(s)")
            print(f"Current device: {torch.cuda.get_device_name()}")
        else:
            print("CUDA is not available. Tests may fail.")
    except ImportError:
        print("PyTorch not available. Continuing without CUDA availability check.")
    
    # Create test harness
    harness = CUDATestHarness()
    
    # Compile kernels
    if not harness.compile_kernels():
        print("Failed to compile CUDA kernels. Exiting.")
        return False
    
    # Run tests
    success = harness.run_all_tests()
    
    if success:
        print("\n🎉 All tests passed! CUDA kernels are working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
    
    return success

if __name__ == "__main__":
    main() 