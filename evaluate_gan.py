import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import glob
from scipy import stats

def calculate_image_statistics(image_path):
    """Calculate basic statistics for an image"""
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img_array = np.array(img)
    
    stats = {
        'mean': img_array.mean(),
        'std': img_array.std(),
        'min': img_array.min(),
        'max': img_array.max(),
        'contrast': img_array.max() - img_array.min(),
        'entropy': calculate_entropy(img_array)
    }
    return stats

def calculate_entropy(img_array):
    """Calculate image entropy as a measure of information content"""
    hist, _ = np.histogram(img_array, bins=256, range=(0, 256))
    hist = hist[hist > 0]  # Remove zero bins
    prob = hist / hist.sum()
    entropy = -np.sum(prob * np.log2(prob))
    return entropy

def evaluate_generated_images(samples_dir='samples_improved'):
    """Evaluate the quality of generated images"""
    
    if not os.path.exists(samples_dir):
        print(f"❌ Samples directory '{samples_dir}' not found")
        return
    
    sample_files = sorted(glob.glob(f'{samples_dir}/epoch_*.png'))
    if not sample_files:
        print(f"❌ No sample files found in '{samples_dir}'")
        return
    
    print(f"🔍 Evaluating {len(sample_files)} sample files...")
    
    # Analyze each epoch
    epoch_stats = []
    for i, file_path in enumerate(sample_files):
        epoch_num = int(file_path.split('epoch_')[-1].split('.')[0])
        stats = calculate_image_statistics(file_path)
        stats['epoch'] = epoch_num
        epoch_stats.append(stats)
        
        print(f"Epoch {epoch_num}: Mean={stats['mean']:.1f}, Std={stats['std']:.1f}, Entropy={stats['entropy']:.2f}")
    
    # Plot evolution of statistics
    epochs = [s['epoch'] for s in epoch_stats]
    means = [s['mean'] for s in epoch_stats]
    stds = [s['std'] for s in epoch_stats]
    entropies = [s['entropy'] for s in epoch_stats]
    contrasts = [s['contrast'] for s in epoch_stats]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Mean pixel values
    axes[0, 0].plot(epochs, means, 'b-o')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Mean Pixel Value')
    axes[0, 0].set_title('Mean Pixel Values Over Time')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Standard deviation
    axes[0, 1].plot(epochs, stds, 'r-o')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Standard Deviation')
    axes[0, 1].set_title('Pixel Standard Deviation Over Time')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Entropy
    axes[1, 0].plot(epochs, entropies, 'g-o')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Entropy')
    axes[1, 0].set_title('Image Entropy Over Time')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Contrast
    axes[1, 1].plot(epochs, contrasts, 'm-o')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Contrast')
    axes[1, 1].set_title('Image Contrast Over Time')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{samples_dir}_evaluation.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Quality assessment
    print(f"\n📊 Quality Assessment:")
    
    # Check for mode collapse (low entropy)
    avg_entropy = np.mean(entropies)
    if avg_entropy < 4.0:
        print(f"⚠️  Low entropy ({avg_entropy:.2f}) - possible mode collapse")
    else:
        print(f"✅ Good entropy ({avg_entropy:.2f}) - diverse images")
    
    # Check for proper contrast
    avg_contrast = np.mean(contrasts)
    if avg_contrast < 100:
        print(f"⚠️  Low contrast ({avg_contrast:.1f}) - images may be too uniform")
    else:
        print(f"✅ Good contrast ({avg_contrast:.1f}) - clear image features")
    
    # Check for reasonable mean values
    avg_mean = np.mean(means)
    if avg_mean < 100 or avg_mean > 200:
        print(f"⚠️  Unusual mean pixel value ({avg_mean:.1f}) - may indicate issues")
    else:
        print(f"✅ Reasonable mean pixel value ({avg_mean:.1f})")
    
    # Trend analysis
    if len(epochs) > 1:
        entropy_trend = np.polyfit(epochs, entropies, 1)[0]
        if entropy_trend > 0.1:
            print(f"✅ Entropy increasing (trend: {entropy_trend:.3f}) - good sign")
        elif entropy_trend < -0.1:
            print(f"⚠️  Entropy decreasing (trend: {entropy_trend:.3f}) - concerning")
        else:
            print(f"➡️  Entropy stable (trend: {entropy_trend:.3f})")

def compare_training_runs():
    """Compare original vs improved training results"""
    
    print("🔄 Comparing training runs...")
    
    # Check if both directories exist
    original_exists = os.path.exists('samples')
    improved_exists = os.path.exists('samples_improved')
    
    if not original_exists and not improved_exists:
        print("❌ No training results found")
        return
    
    if original_exists:
        original_files = sorted(glob.glob('samples/epoch_*.png'))
        if original_files:
            original_last = calculate_image_statistics(original_files[-1])
            print(f"📊 Original training (last epoch):")
            print(f"   - Mean: {original_last['mean']:.1f}")
            print(f"   - Std: {original_last['std']:.1f}")
            print(f"   - Entropy: {original_last['entropy']:.2f}")
    
    if improved_exists:
        improved_files = sorted(glob.glob('samples_improved/epoch_*.png'))
        if improved_files:
            improved_last = calculate_image_statistics(improved_files[-1])
            print(f"📊 Improved training (last epoch):")
            print(f"   - Mean: {improved_last['mean']:.1f}")
            print(f"   - Std: {improved_last['std']:.1f}")
            print(f"   - Entropy: {improved_last['entropy']:.2f}")
    
    if original_exists and improved_exists and original_files and improved_files:
        print(f"\n📈 Comparison:")
        entropy_improvement = improved_last['entropy'] - original_last['entropy']
        contrast_improvement = improved_last['contrast'] - original_last['contrast']
        
        if entropy_improvement > 0.5:
            print(f"✅ Significant entropy improvement: +{entropy_improvement:.2f}")
        elif entropy_improvement > 0:
            print(f"➡️  Slight entropy improvement: +{entropy_improvement:.2f}")
        else:
            print(f"⚠️  Entropy decreased: {entropy_improvement:.2f}")
        
        if contrast_improvement > 10:
            print(f"✅ Significant contrast improvement: +{contrast_improvement:.1f}")
        elif contrast_improvement > 0:
            print(f"➡️  Slight contrast improvement: +{contrast_improvement:.1f}")
        else:
            print(f"⚠️  Contrast decreased: {contrast_improvement:.1f}")

def generate_sample_grid(samples_dir='samples_improved', output_file='sample_grid.png'):
    """Create a grid of samples from different epochs"""
    
    if not os.path.exists(samples_dir):
        print(f"❌ Samples directory '{samples_dir}' not found")
        return
    
    sample_files = sorted(glob.glob(f'{samples_dir}/epoch_*.png'))
    if not sample_files:
        print(f"❌ No sample files found in '{samples_dir}'")
        return
    
    # Select samples from different epochs
    num_samples = min(8, len(sample_files))
    step = max(1, len(sample_files) // num_samples)
    selected_files = sample_files[::step][:num_samples]
    
    # Create grid
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, file_path in enumerate(selected_files):
        row, col = i // 4, i % 4
        epoch_num = int(file_path.split('epoch_')[-1].split('.')[0])
        
        img = Image.open(file_path)
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].set_title(f'Epoch {epoch_num}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📸 Created sample grid: {output_file}")

if __name__ == "__main__":
    print("=== GAN Evaluation Tool ===\n")
    
    # Evaluate improved training if available
    if os.path.exists('samples_improved'):
        print("🔍 Evaluating improved training results...")
        evaluate_generated_images('samples_improved')
        generate_sample_grid('samples_improved', 'improved_sample_grid.png')
    else:
        print("⚠️  No improved training results found. Run 'python train_improved.py' first.")
    
    # Compare with original training
    compare_training_runs()
    
    print("\n✅ Evaluation complete!") 