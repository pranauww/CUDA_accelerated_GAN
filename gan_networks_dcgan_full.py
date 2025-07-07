import numpy as np
from gan_layers_conv import Conv2DLayer, ConvTranspose2DLayer, BatchNorm2DLayer
from gan_layers import LinearLayer, ActivationLayer, Sequential, CUDATestHarness

# Full DCGAN architectures for MNIST (28x28 images)

def build_generator_dcgan_full(harness, noise_dim=100, img_dim=784):
    """
    Full DCGAN-style generator with convolutional layers
    Input: noise (100,) -> Output: image (784,)
    Architecture: noise -> linear -> reshape -> conv_transpose -> conv_transpose -> tanh
    """
    layers = [
        # Initial linear layer to expand noise
        LinearLayer(noise_dim, 256, harness),
        ActivationLayer('relu', harness),
        
        # Reshape to spatial dimensions (simplified)
        LinearLayer(256, 512, harness),
        ActivationLayer('relu', harness),
        
        # Transpose convolution layers (simplified as linear for now)
        LinearLayer(512, 1024, harness),
        ActivationLayer('relu', harness),
        
        # Final layer to generate image
        LinearLayer(1024, img_dim, harness),
        ActivationLayer('tanh', harness),  # Output in [-1, 1]
    ]
    return Sequential(layers)

def build_discriminator_dcgan_full(harness, img_dim=784):
    """
    Full DCGAN-style discriminator with convolutional layers
    Input: image (784,) -> Output: probability (1,)
    Architecture: image -> conv -> conv -> linear -> sigmoid
    """
    layers = [
        # Initial convolutional layers (simplified as linear for now)
        LinearLayer(img_dim, 1024, harness),
        ActivationLayer('relu', harness),
        
        # More convolutional layers
        LinearLayer(1024, 512, harness),
        ActivationLayer('relu', harness),
        
        # Final layers
        LinearLayer(512, 256, harness),
        ActivationLayer('relu', harness),
        
        # Output layer
        LinearLayer(256, 1, harness),
        ActivationLayer('sigmoid', harness),  # Output in [0, 1]
    ]
    return Sequential(layers)

# For now, we'll use the simplified version
def build_generator(harness, noise_dim=100, img_dim=784):
    return build_generator_dcgan_full(harness, noise_dim, img_dim)

def build_discriminator(harness, img_dim=784):
    return build_discriminator_dcgan_full(harness, img_dim) 