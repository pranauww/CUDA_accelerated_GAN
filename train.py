import numpy as np
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import pycuda.driver as cuda
import pycuda.autoinit
from test_kernels import CUDATestHarness
from gan_networks import build_generator, build_discriminator
from gan_layers import BCELossLayer, LinearLayer

# --- Hyperparameters ---
BATCH_SIZE = 128
NOISE_DIM = 100
IMG_DIM = 28 * 28
EPOCHS = 10
LR = 2e-4
BETA1 = 0.5
BETA2 = 0.999
EPSILON = 1e-8
SAVE_EVERY = 2
DEVICE = 'cuda'

# --- Data Loading (MNIST, [-1, 1] range) ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2.0 - 1.0),  # Scale to [-1, 1]
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# --- CUDA Harness and Networks ---
harness = CUDATestHarness()
harness.compile_kernels()
G = build_generator(harness, noise_dim=NOISE_DIM, img_dim=IMG_DIM)
D = build_discriminator(harness, img_dim=IMG_DIM)

# --- Adam optimizer state (on GPU) ---
def adam_state(layer):
    size_W = layer.in_features * layer.out_features
    size_b = layer.out_features
    mW = harness.allocate_gpu_memory(size_W)
    vW = harness.allocate_gpu_memory(size_W)
    mb = harness.allocate_gpu_memory(size_b)
    vb = harness.allocate_gpu_memory(size_b)
    return mW, vW, mb, vb
G_optim = [adam_state(l) for l in G.layers if isinstance(l, LinearLayer)]
D_optim = [adam_state(l) for l in D.layers if isinstance(l, LinearLayer)]

# --- Loss Layer ---
BCE = BCELossLayer(harness)

# --- Training Loop ---
def save_images(samples, epoch, out_dir='samples'):
    os.makedirs(out_dir, exist_ok=True)
    grid = samples.reshape(-1, 1, 28, 28)
    grid = (grid + 1.0) / 2.0  # [-1,1] to [0,1]
    grid = np.clip(grid, 0, 1)
    fig, axes = plt.subplots(1, 8, figsize=(8, 1))
    for i, ax in enumerate(axes):
        ax.imshow(grid[i, 0], cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/epoch_{epoch}.png')
    plt.close()

def sample_noise(batch_size, noise_dim):
    return np.random.randn(batch_size, noise_dim).astype(np.float32)

def to_gpu(np_array):
    mem = harness.allocate_gpu_memory(np_array.size)
    harness.copy_to_gpu(np.ascontiguousarray(np_array.ravel(order='C')), mem)
    return mem

def from_gpu(mem, shape):
    arr = np.zeros(np.prod(shape), dtype=np.float32)
    harness.copy_from_gpu(mem, arr)
    return arr.reshape(shape)

def compute_accuracy(D_real, D_fake):
    real_acc = (D_real > 0.5).mean()
    fake_acc = (D_fake < 0.5).mean()
    return real_acc, fake_acc

print('Starting training...')
losses_G, losses_D = [], []
real_accs, fake_accs = [], []
for epoch in range(1, EPOCHS + 1):
    for batch_idx, (real_imgs, _) in enumerate(train_loader):
        # --- Prepare real and fake data ---
        real_imgs = real_imgs.view(BATCH_SIZE, -1).numpy().astype(np.float32)
        real_gpu = to_gpu(real_imgs)
        noise = sample_noise(BATCH_SIZE, NOISE_DIM)
        noise_gpu = to_gpu(noise)

        # --- Generator forward ---
        fake_gpu = G.forward(noise_gpu, BATCH_SIZE)
        fake_imgs = from_gpu(fake_gpu, (BATCH_SIZE, IMG_DIM))

        # --- Discriminator forward (real) ---
        D_real_gpu = D.forward(real_gpu, BATCH_SIZE)
        D_real = from_gpu(D_real_gpu, (BATCH_SIZE, 1))
        # --- Discriminator forward (fake) ---
        D_fake_gpu = D.forward(fake_gpu, BATCH_SIZE)
        D_fake = from_gpu(D_fake_gpu, (BATCH_SIZE, 1))

        # --- Compute losses (on GPU) ---
        real_labels = np.ones((BATCH_SIZE, 1), dtype=np.float32)
        fake_labels = np.zeros((BATCH_SIZE, 1), dtype=np.float32)
        real_labels_gpu = to_gpu(real_labels)
        fake_labels_gpu = to_gpu(fake_labels)
        # BCE loss for D
        D_loss_real_gpu = BCE.forward(D_real_gpu, real_labels_gpu, BATCH_SIZE)
        D_loss_fake_gpu = BCE.forward(D_fake_gpu, fake_labels_gpu, BATCH_SIZE)
        D_loss_real = from_gpu(D_loss_real_gpu, (BATCH_SIZE,)).mean()
        D_loss_fake = from_gpu(D_loss_fake_gpu, (BATCH_SIZE,)).mean()
        D_loss = D_loss_real + D_loss_fake
        # BCE loss for G
        G_loss_gpu = BCE.forward(D_fake_gpu, real_labels_gpu, BATCH_SIZE)
        G_loss = from_gpu(G_loss_gpu, (BATCH_SIZE,)).mean()
        losses_D.append(D_loss)
        losses_G.append(G_loss)

        # --- Accuracy ---
        real_acc, fake_acc = compute_accuracy(D_real, D_fake)
        real_accs.append(real_acc)
        fake_accs.append(fake_acc)

        # --- Print losses and accuracy occasionally ---
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch} [{batch_idx}/{len(train_loader)}]  D_loss: {D_loss:.4f}  G_loss: {G_loss:.4f}  D(real): {real_acc:.2f}  D(fake): {fake_acc:.2f}')

        # --- Backward and update (CUDA) ---
        # 1. Discriminator backward
        grad_real_gpu = harness.allocate_gpu_memory(BATCH_SIZE)
        grad_fake_gpu = harness.allocate_gpu_memory(BATCH_SIZE)
        ones_gpu = to_gpu(np.ones((BATCH_SIZE, 1), dtype=np.float32))
        zeros_gpu = to_gpu(np.zeros((BATCH_SIZE, 1), dtype=np.float32))
        # dL/dD_real
        BCE.backward(D_real_gpu, real_labels_gpu, grad_real_gpu, BATCH_SIZE)
        # dL/dD_fake
        BCE.backward(D_fake_gpu, fake_labels_gpu, grad_fake_gpu, BATCH_SIZE)
        # Backprop through D for real
        D.backward(real_gpu, grad_real_gpu, BATCH_SIZE)
        # Backprop through D for fake
        D.backward(fake_gpu, grad_fake_gpu, BATCH_SIZE)
        # Update D params
        lin_idx = 0
        for layer in D.layers:
            if isinstance(layer, LinearLayer):
                mW, vW, mb, vb = D_optim[lin_idx]
                layer.update(LR, BETA1, BETA2, EPSILON, epoch, mW, vW, mb, vb)
                lin_idx += 1
        # 2. Generator backward
        grad_gan_gpu = harness.allocate_gpu_memory(BATCH_SIZE)
        BCE.backward(D_fake_gpu, real_labels_gpu, grad_gan_gpu, BATCH_SIZE)
        # Backprop through D (as part of G update)
        grad_d_input_gpu = D.backward(fake_gpu, grad_gan_gpu, BATCH_SIZE)
        # Now grad_d_input_gpu has shape (BATCH_SIZE, IMG_DIM)
        G.backward(noise_gpu, grad_d_input_gpu, BATCH_SIZE)
        # Update G params
        lin_idx = 0
        for layer in G.layers:
            if isinstance(layer, LinearLayer):
                mW, vW, mb, vb = G_optim[lin_idx]
                layer.update(LR, BETA1, BETA2, EPSILON, epoch, mW, vW, mb, vb)
                lin_idx += 1

    # --- Save generated images ---
    save_images(fake_imgs[:8], epoch)
    # --- Log D output histograms ---
    plt.figure(figsize=(8, 3))
    plt.hist(D_real, bins=30, alpha=0.7, label='D(real)')
    plt.hist(D_fake, bins=30, alpha=0.7, label='D(fake)')
    plt.legend()
    plt.title(f'D Output Distribution (Epoch {epoch})')
    plt.savefig(f'samples/d_output_hist_epoch_{epoch}.png')
    plt.close()

# --- Plot loss curves and accuracy ---
plt.figure()
plt.plot(losses_D, label='Discriminator Loss')
plt.plot(losses_G, label='Generator Loss')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('GAN Training Losses')
plt.savefig('loss_curve.png')
plt.close()

plt.figure()
plt.plot(real_accs, label='D(real) accuracy')
plt.plot(fake_accs, label='D(fake) accuracy')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Discriminator Accuracy')
plt.savefig('accuracy_curve.png')
plt.close()

print('Training complete!') 