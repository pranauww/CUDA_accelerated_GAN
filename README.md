# CUDA-Accelerated GAN from Scratch

A high-performance Generative Adversarial Network (GAN) implementation built from scratch using custom CUDA kernels, without relying on PyTorch or TensorFlow for training. This project demonstrates how to implement neural network operations directly on GPU using CUDA.

## Features

- **Custom CUDA Kernels**: Matrix multiplication (GEMM), activation functions, loss functions, and optimizers
- **High Performance**: Optimized for NVIDIA GPUs with efficient memory management
- **Cross-Platform**: Works on Windows and Linux
- **Mixed Precision Support**: Ready for float16/float32 mixed precision training
- **Complete GAN Pipeline**: Generator, Discriminator, and training loop implementation
- **Full CUDA Backward Pass**: All gradients and optimizer updates run on GPU
- **Advanced Logging & Visualization**: Loss curves, accuracy, D output histograms, and generated image grids

## Development History & Changes

### Phase 1: Core CUDA Infrastructure
- **Custom CUDA Kernels** (`cuda_kernels.cu`): Implemented matrix multiplication (GEMM), activation functions (ReLU, Sigmoid, Tanh), binary cross-entropy loss, and Adam optimizer
- **Test Harness** (`test_kernels.py`): Created comprehensive testing framework for kernel validation
- **Layer Abstractions** (`gan_layers.py`): Built CUDA-accelerated layer classes (LinearLayer, ActivationLayer, BCELossLayer, Sequential)
- **Cross-Platform Support**: Added `__declspec(dllexport)` for Windows DLL exports

### Phase 2: GAN Architecture & Training
- **Network Definitions** (`gan_networks.py`): Implemented MLP-based Generator and Discriminator architectures
- **Full Training Loop** (`train.py`): Complete end-to-end GAN training with GPU acceleration
- **Advanced Logging**: Added per-batch/epoch loss tracking, accuracy metrics, D output histograms, and generated image visualization
- **Memory Management**: Optimized PyCUDA memory handling with proper cleanup

### Phase 3: Training Stability & Diagnostics
- **Training Analysis** (`analyze_results.py`): Comprehensive analysis of training dynamics
- **GAN Diagnostics** (`diagnose_gan.py`): Mode collapse detection and training health monitoring
- **Evaluation Tools** (`evaluate_gan.py`): FID score calculation and quality assessment
- **Improved Training Scripts**: Multiple variants with different strategies:
  - `train_improved.py`: Better hyperparameters and balanced training
  - `train_advanced.py`: Advanced techniques (gradient clipping, label smoothing)
  - `train_balanced.py`: Balanced D/G training frequency
  - `train_hinge.py`: Hinge loss implementation
  - `train_wgan.py`: Wasserstein GAN approach

### Phase 4: Convolutional GAN Architecture
- **Convolutional Kernels** (`cuda_kernels_conv.cu`): Added 2D convolution and transpose convolution kernels
- **Conv Layer Classes** (`gan_layers_conv.py`): Convolutional and transpose convolutional layer implementations
- **DCGAN Networks** (`gan_networks_dcgan.py`, `gan_networks_dcgan_full.py`): Deep Convolutional GAN architectures
- **DCGAN Training** (`train_dcgan.py`): Full DCGAN training pipeline with convolutional layers

### Key Technical Improvements

#### CUDA Kernel Enhancements
- **Batched GEMM**: Support for batch matrix operations
- **Gradient Computation**: Complete backward pass implementation
- **Adam Optimizer**: Full GPU-based parameter updates with momentum
- **Error Handling**: Comprehensive CUDA error checking and debugging

#### Training Stability Fixes
- **Batch Size Tracking**: Fixed incorrect batch size calculations in layer forward passes
- **Gradient Shape Management**: Proper tracking of gradient shapes through backward pass
- **Memory Copy Errors**: Resolved GPU memory copy issues with correct buffer sizes
- **Layer Input Storage**: Store layer inputs during forward pass for correct backpropagation

#### Architecture Evolution
- **MLP → DCGAN**: Transition from simple MLP to convolutional architectures
- **Mode Collapse Mitigation**: Implemented various strategies to address GAN training instability
- **Loss Function Improvements**: Added hinge loss and Wasserstein loss variants
- **Training Strategy Refinement**: Balanced D/G training, gradient clipping, label smoothing

#### Development Environment
- **Visual Studio Integration**: Fixed Windows compilation issues with proper compiler setup
- **Dependency Management**: Added OpenCV for evaluation scripts
- **Cross-Platform Compatibility**: Ensured Windows and Linux support

## Requirements

### Hardware
- NVIDIA GPU with CUDA support (Compute Capability 7.0+)
- Tested on NVIDIA GeForce GTX 1660 Ti

### Software
- **Windows**: 
  - CUDA Toolkit 11.0+ (tested with CUDA 12.9)
  - Visual Studio 2022 Build Tools with C++ workload
  - Python 3.8+
- **Linux**:
  - CUDA Toolkit 11.0+
  - GCC compiler
  - Python 3.8+

### Python Dependencies
```bash
pip install numpy pycuda torch matplotlib torchvision
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd CUDA_project
   ```

2. **Install Python dependencies**:
   ```bash
   pip install numpy pycuda torch matplotlib torchvision
   ```

3. **Verify CUDA installation**:
   ```bash
   nvcc --version
   ```

4. **Test the setup**:
   ```bash
   python test_kernels.py
   ```

## Project Structure

```
CUDA_project/
├── cuda_kernels.cu          # Custom CUDA kernels implementation
├── test_kernels.py          # Test harness for kernel validation
├── gan_layers.py            # CUDA-accelerated layer/network abstractions
├── gan_networks.py          # Generator and Discriminator definitions
├── train.py                 # Full GAN training loop (end-to-end)
├── README.md                # This file
└── requirements.txt         # Python dependencies
```

## GAN Architecture

- **Generator**: MLP with 4 layers (ReLU/Tanh activations), outputs 28x28 images (MNIST)
- **Discriminator**: MLP with 4 layers (ReLU/Sigmoid activations), outputs probability
- **All layers**: Fully GPU-accelerated, including forward, backward, and optimizer steps

## 🔧 CUDA Kernels & Layer Abstractions

### Matrix Operations
- **GEMM**: General matrix multiplication with batched support
- **Transpose**: Matrix transpose for gradient computation

### Activation Functions
- **ReLU**: Rectified Linear Unit with gradient
- **Sigmoid**: Sigmoid activation with gradient
- **Tanh**: Hyperbolic tangent with gradient

### Loss Functions
- **Binary Cross-Entropy**: Loss and gradient computation (CUDA kernel)

### Optimizers
- **Adam**: Adaptive moment estimation optimizer (CUDA kernel)

### Layer Abstractions
- **LinearLayer**: Fully connected layer with CUDA-based forward/backward/Adam update
- **ActivationLayer**: CUDA-based activations and gradients
- **BCELossLayer**: CUDA-based binary cross-entropy loss/grad
- **Sequential**: Container for stacking layers, tracks layer inputs for correct backprop

## Training Loop

- Loads and preprocesses MNIST data ([-1, 1] range)
- Alternates D and G training steps
- All forward, backward, and optimizer steps run on GPU using custom kernels
- Uses PyCUDA for all memory management
- **Advanced logging**:
  - Per-batch and per-epoch loss for D and G
  - Discriminator accuracy on real and fake samples
  - D output histograms (saved as images)
  - Generated image grids saved every epoch
  - Loss and accuracy curves saved at the end

## Testing

Run the comprehensive test suite to validate all CUDA kernels:

```bash
python test_kernels.py
```

This will:
1. Compile the CUDA kernels into a shared library
2. Test matrix multiplication (GEMM)
3. Validate activation functions (ReLU, Sigmoid, Tanh)
4. Verify binary cross-entropy loss and gradients
5. Test Adam optimizer implementation

Expected output:
```
==================================================
Test Results: 4/4 tests passed
==================================================

🎉 All tests passed! CUDA kernels are working correctly.
```

## Usage

### Basic Kernel Usage

```python
import numpy as np
import ctypes
from test_kernels import CUDATestHarness

# Initialize test harness
harness = CUDATestHarness()
harness.compile_kernels()

# Example: Matrix multiplication
M, N, K = 64, 64, 64
A = np.random.randn(M, K).astype(np.float32)
B = np.random.randn(K, N).astype(np.float32)

# Allocate GPU memory
A_gpu = harness.allocate_gpu_memory(M * K)
B_gpu = harness.allocate_gpu_memory(K * N)
C_gpu = harness.allocate_gpu_memory(M * N)

# Copy data to GPU
harness.copy_to_gpu(np.ascontiguousarray(A.ravel(order='C')), A_gpu)
harness.copy_to_gpu(np.ascontiguousarray(B.ravel(order='C')), B_gpu)

# Perform matrix multiplication
harness.lib.cuda_gemm(
    ctypes.c_void_p(int(A_gpu)),
    ctypes.c_void_p(int(B_gpu)),
    ctypes.c_void_p(int(C_gpu)),
    ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K),
    ctypes.c_float(1.0), ctypes.c_float(0.0)
)

# Copy result back
C_cuda = np.zeros((M, N), dtype=np.float32)
harness.copy_from_gpu(C_gpu, C_cuda.ravel(order='C'))
C_cuda = C_cuda.reshape((M, N), order='C')
```

## Technical Details

### Memory Management
- Uses PyCUDA for GPU memory allocation and data transfer
- Automatic memory cleanup through PyCUDA's reference counting
- Contiguous array handling for optimal performance

### Kernel Optimization
- Shared memory tiling ready for GEMM optimization
- Efficient grid/block dimension calculation
- Error checking and debugging support

### Cross-Platform Support
- Windows: Uses `__declspec(dllexport)` for DLL exports
- Linux: Uses standard `extern "C"` declarations
- Automatic platform detection and compilation

## Troubleshooting

### Common Issues

1. **"nvcc not found"**
   - Install CUDA Toolkit and add to PATH
   - Verify with `nvcc --version`

2. **"Cannot find compiler 'cl.exe'" (Windows)**
   - Install Visual Studio 2022 Build Tools with C++ workload
   - Use Developer Command Prompt

3. **"function not found" errors**
   - Recompile kernels: `python test_kernels.py`
   - Check CUDA version compatibility

4. **Large GEMM errors**
   - Ensure arrays are contiguous and row-major
   - Check memory alignment

5. **cuMemcpyDtoH failed: invalid argument**
   - Ensure the host buffer and device buffer sizes match exactly
   - Track and pass the correct gradient shape through all layers
   - Store and use layer inputs for correct backprop

### Debug Mode
Enable detailed error reporting by modifying `cuda_kernels.cu`:
```cpp
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)
```

## Performance

### Benchmarks
- **GEMM**: ~7.6e-06 max error (float32 precision)
- **Activations**: ~1.2e-07 max error
- **Loss Functions**: ~1.2e-07 max error
- **Adam Optimizer**: ~3.7e-09 max error

### Optimization Opportunities
- Shared memory tiling for larger matrices
- Mixed precision (float16) support
- CUDA streams for overlapping operations
- Kernel fusion for better memory bandwidth

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Acknowledgments

- NVIDIA for CUDA toolkit and documentation
- PyCUDA developers for Python-CUDA integration
- The CUDA programming community for best practices

---
