import numpy as np
from gan_layers import LinearLayer, ActivationLayer, Sequential, CUDATestHarness

# Example MLP architectures for MNIST (28x28 images, 784 input)

def build_generator(harness, noise_dim=100, img_dim=784):
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

def build_discriminator(harness, img_dim=784):
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