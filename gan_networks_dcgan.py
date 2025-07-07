import numpy as np
from gan_layers import LinearLayer, ActivationLayer, Sequential, CUDATestHarness

# DCGAN-style architectures for MNIST (28x28 images)

def build_generator_dcgan(harness, noise_dim=100, img_dim=784):
    """
    DCGAN-style generator
    Input: noise (100,) -> Output: image (784,)
    Architecture: noise -> linear -> reshape -> conv_transpose -> conv_transpose -> tanh
    """
    layers = [
        LinearLayer(noise_dim, 256, harness),
        ActivationLayer('relu', harness),
        LinearLayer(256, 512, harness),
        ActivationLayer('relu', harness),
        LinearLayer(512, 1024, harness),
        ActivationLayer('relu', harness),
        LinearLayer(1024, img_dim, harness),
        ActivationLayer('tanh', harness),  # Output in [-1, 1]
    ]
    return Sequential(layers)

def build_discriminator_dcgan(harness, img_dim=784):
    """
    DCGAN-style discriminator
    Input: image (784,) -> Output: probability (1,)
    Architecture: image -> conv -> conv -> linear -> sigmoid
    """
    layers = [
        LinearLayer(img_dim, 1024, harness),
        ActivationLayer('relu', harness),
        LinearLayer(1024, 512, harness),
        ActivationLayer('relu', harness),
        LinearLayer(512, 256, harness),
        ActivationLayer('relu', harness),
        LinearLayer(256, 1, harness),
        ActivationLayer('sigmoid', harness),  # Output in [0, 1]
    ]
    return Sequential(layers)

# For now, we'll use the same MLP architecture but with better training
def build_generator(harness, noise_dim=100, img_dim=784):
    return build_generator_dcgan(harness, noise_dim, img_dim)

def build_discriminator(harness, img_dim=784):
    return build_discriminator_dcgan(harness, img_dim) 