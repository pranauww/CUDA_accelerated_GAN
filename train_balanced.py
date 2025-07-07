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

# --- Balanced Training Hyperparameters ---
BATCH_SIZE = 32  # Smaller batch size for more frequent updates
NOISE_DIM = 100
IMG_DIM = 28 * 28
EPOCHS = 100
LR_G = 1e-3  # Much higher learning rate for generator
LR_D = 1e-4  # Much lower learning rate for discriminator
BETA1 = 0.5
BETA2 = 0.999
EPSILON = 1e-8
SAVE_EVERY = 10
DEVICE = 'cuda'

# --- Data Loading (MNIST, [0, 1] range) ---
transform = transforms.Compose([
    transforms.ToTensor(),  # Keep in [0, 1] range
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
def save_images(samples, epoch, out_dir='samples_balanced'):
    os.makedirs(out_dir, exist_ok=True)
    grid = samples.reshape(-1, 1, 28, 28)
    grid = np.clip(grid, 0, 1)
    fig, axes = plt.subplots(2, 4, figsize=(8, 4))
    for i in range(8):
        row, col = i // 4, i % 4
        axes[row, col].imshow(grid[i, 0], cmap='gray')
        axes[row, col].axis('off')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/epoch_{epoch}.png', dpi=150, bbox_inches='tight')
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

print('Starting balanced training...')
losses_G, losses_D = [], []
real_accs, fake_accs = [], []
d_real_means, d_fake_means = [], []

for epoch in range(1, EPOCHS + 1):
    epoch_losses_G, epoch_losses_D = [], []
    epoch_real_accs, epoch_fake_accs = [], []
    epoch_d_real_means, epoch_d_fake_means = [], []
    
    for batch_idx, (real_imgs, _) in enumerate(train_loader):
        # --- Prepare real and fake data ---
        real_imgs = real_imgs.view(BATCH_SIZE, -1).numpy().astype(np.float32)
        real_gpu = to_gpu(real_imgs)
        noise = sample_noise(BATCH_SIZE, NOISE_DIM)
        noise_gpu = to_gpu(noise)

        # --- Generator forward ---
        fake_gpu = G.forward(noise_gpu, BATCH_SIZE)
        fake_imgs = from_gpu(fake_gpu, (BATCH_SIZE, IMG_DIM))
        
        # Scale generator output from [-1, 1] to [0, 1]
        fake_imgs = (fake_imgs + 1.0) / 2.0
        fake_imgs = np.clip(fake_imgs, 0, 1)
        fake_gpu = to_gpu(fake_imgs)

        # --- Discriminator forward (real) ---
        D_real_gpu = D.forward(real_gpu, BATCH_SIZE)
        D_real = from_gpu(D_real_gpu, (BATCH_SIZE, 1))
        # --- Discriminator forward (fake) ---
        D_fake_gpu = D.forward(fake_gpu, BATCH_SIZE)
        D_fake = from_gpu(D_fake_gpu, (BATCH_SIZE, 1))

        # --- Compute losses with different strategies ---
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
        
        epoch_losses_D.append(D_loss)
        epoch_losses_G.append(G_loss)

        # --- Accuracy and statistics ---
        real_acc, fake_acc = compute_accuracy(D_real, D_fake)
        epoch_real_accs.append(real_acc)
        epoch_fake_accs.append(fake_acc)
        epoch_d_real_means.append(D_real.mean())
        epoch_d_fake_means.append(D_fake.mean())

        # --- Print losses and accuracy occasionally ---
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch} [{batch_idx}/{len(train_loader)}]  D_loss: {D_loss:.4f}  G_loss: {G_loss:.4f}  D(real): {real_acc:.2f}  D(fake): {fake_acc:.2f}  D_real_mean: {D_real.mean():.3f}  D_fake_mean: {D_fake.mean():.3f}')

        # --- Adaptive training strategy ---
        # Train discriminator less frequently if it's too confident
        d_real_mean = D_real.mean()
        d_fake_mean = D_fake.mean()
        
        # Only train discriminator if it's not too confident
        train_discriminator = (d_real_mean < 0.9) and (d_fake_mean > 0.1)
        
        if train_discriminator:
            # Discriminator backward
            grad_real_gpu = harness.allocate_gpu_memory(BATCH_SIZE)
            grad_fake_gpu = harness.allocate_gpu_memory(BATCH_SIZE)
            
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
                    layer.update(LR_D, BETA1, BETA2, EPSILON, epoch, mW, vW, mb, vb)
                    lin_idx += 1

        # --- Train Generator (always) ---
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
                layer.update(LR_G, BETA1, BETA2, EPSILON, epoch, mW, vW, mb, vb)
                lin_idx += 1

    # --- Epoch statistics ---
    avg_D_loss = np.mean(epoch_losses_D)
    avg_G_loss = np.mean(epoch_losses_G)
    avg_real_acc = np.mean(epoch_real_accs)
    avg_fake_acc = np.mean(epoch_fake_accs)
    avg_d_real_mean = np.mean(epoch_d_real_means)
    avg_d_fake_mean = np.mean(epoch_d_fake_means)
    
    losses_D.extend(epoch_losses_D)
    losses_G.extend(epoch_losses_G)
    real_accs.extend(epoch_real_accs)
    fake_accs.extend(epoch_fake_accs)
    d_real_means.extend(epoch_d_real_means)
    d_fake_means.extend(epoch_d_fake_means)
    
    print(f'Epoch {epoch} Summary: D_loss: {avg_D_loss:.4f}  G_loss: {avg_G_loss:.4f}  D(real): {avg_real_acc:.2f}  D(fake): {avg_fake_acc:.2f}  D_real_mean: {avg_d_real_mean:.3f}  D_fake_mean: {avg_d_fake_mean:.3f}')

    # --- Save generated images ---
    if epoch % SAVE_EVERY == 0:
        save_images(fake_imgs[:8], epoch)
        
        # --- Log D output histograms ---
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.hist(D_real, bins=30, alpha=0.7, label='D(real)', color='blue')
        plt.hist(D_fake, bins=30, alpha=0.7, label='D(fake)', color='red')
        plt.legend()
        plt.title(f'D Output Distribution (Epoch {epoch})')
        plt.xlabel('D(x)')
        plt.ylabel('Count')
        
        plt.subplot(1, 2, 2)
        plt.plot(d_real_means[-100:], label='D(real) mean', color='blue')
        plt.plot(d_fake_means[-100:], label='D(fake) mean', color='red')
        plt.legend()
        plt.title('D Output Means (Last 100 batches)')
        plt.xlabel('Batch')
        plt.ylabel('Mean D(x)')
        
        plt.tight_layout()
        plt.savefig(f'samples_balanced/d_output_hist_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
        plt.close()

# --- Plot comprehensive training curves ---
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Loss curves
axes[0, 0].plot(losses_D, label='Discriminator Loss', alpha=0.7)
axes[0, 0].plot(losses_G, label='Generator Loss', alpha=0.7)
axes[0, 0].legend()
axes[0, 0].set_xlabel('Iteration')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('GAN Training Losses')
axes[0, 0].grid(True, alpha=0.3)

# Accuracy curves
axes[0, 1].plot(real_accs, label='D(real) accuracy', alpha=0.7)
axes[0, 1].plot(fake_accs, label='D(fake) accuracy', alpha=0.7)
axes[0, 1].legend()
axes[0, 1].set_xlabel('Iteration')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].set_title('Discriminator Accuracy')
axes[0, 1].grid(True, alpha=0.3)

# D output means
axes[1, 0].plot(d_real_means, label='D(real) mean', alpha=0.7)
axes[1, 0].plot(d_fake_means, label='D(fake) mean', alpha=0.7)
axes[1, 0].legend()
axes[1, 0].set_xlabel('Iteration')
axes[1, 0].set_ylabel('Mean D(x)')
axes[1, 0].set_title('Discriminator Output Means')
axes[1, 0].grid(True, alpha=0.3)

# Loss ratio
loss_ratio = [d/g if g > 0 else 0 for d, g in zip(losses_D, losses_G)]
axes[1, 1].plot(loss_ratio, label='D_loss / G_loss', alpha=0.7)
axes[1, 1].axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Ratio = 1')
axes[1, 1].legend()
axes[1, 1].set_xlabel('Iteration')
axes[1, 1].set_ylabel('Loss Ratio')
axes[1, 1].set_title('Discriminator/Generator Loss Ratio')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves_balanced.png', dpi=150, bbox_inches='tight')
plt.close()

print('Balanced training complete!')
print(f'Final D loss: {losses_D[-1]:.4f}')
print(f'Final G loss: {losses_G[-1]:.4f}')
print(f'Final D(real) accuracy: {real_accs[-1]:.2f}')
print(f'Final D(fake) accuracy: {fake_accs[-1]:.2f}') 