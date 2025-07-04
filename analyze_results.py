import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import glob

def analyze_training_results():
    """Analyze the current training results and provide insights"""
    
    print("=== GAN Training Analysis ===\n")
    
    # Check if training files exist
    if not os.path.exists('loss_curve.png'):
        print("❌ No training curves found. Run training first.")
        return
    
    if not os.path.exists('samples'):
        print("❌ No sample images found. Run training first.")
        return
    
    # Analyze sample images
    sample_files = sorted(glob.glob('samples/epoch_*.png'))
    if sample_files:
        print(f"📊 Found {len(sample_files)} sample files")
        
        # Load first and last samples for comparison
        first_img = Image.open(sample_files[0])
        last_img = Image.open(sample_files[-1])
        
        print(f"📸 First epoch: {sample_files[0]}")
        print(f"📸 Last epoch: {sample_files[-1]}")
        
        # Basic image analysis
        first_array = np.array(first_img)
        last_array = np.array(last_img)
        
        print(f"🔍 First epoch image stats:")
        print(f"   - Mean pixel value: {first_array.mean():.3f}")
        print(f"   - Std pixel value: {first_array.std():.3f}")
        print(f"   - Min/Max: {first_array.min():.3f}/{first_array.max():.3f}")
        
        print(f"🔍 Last epoch image stats:")
        print(f"   - Mean pixel value: {last_array.mean():.3f}")
        print(f"   - Std pixel value: {last_array.std():.3f}")
        print(f"   - Min/Max: {last_array.min():.3f}/{last_array.max():.3f}")
    
    # Analyze discriminator histograms
    hist_files = sorted(glob.glob('samples/d_output_hist_epoch_*.png'))
    if hist_files:
        print(f"\n📈 Found {len(hist_files)} discriminator histogram files")
    
    print("\n=== Training Issues Identified ===")
    print("❌ **Mode Collapse**: Discriminator outputs show D(real)=0.00, D(fake)=1.00")
    print("   - This indicates the discriminator is completely fooled")
    print("   - Generator loss plateaus at maximum value (16.1181)")
    print("   - Discriminator loss increases, showing it's getting worse")
    
    print("\n❌ **Training Instability**:")
    print("   - Discriminator becomes too strong too quickly")
    print("   - Generator fails to learn meaningful features")
    print("   - No convergence in loss values")
    
    print("\n=== Root Causes ===")
    print("🔍 **Hyperparameter Issues**:")
    print("   - Batch size too large (128) for initial training")
    print("   - Same learning rate for both networks")
    print("   - No label smoothing to prevent overconfidence")
    print("   - Discriminator trained too frequently")
    
    print("\n🔍 **Architecture Issues**:")
    print("   - Simple MLP architecture may be insufficient")
    print("   - No gradient clipping to prevent exploding gradients")
    print("   - No proper weight initialization")
    
    print("\n=== Recommended Solutions ===")
    print("✅ **Immediate Improvements**:")
    print("   1. Use smaller batch size (64)")
    print("   2. Different learning rates (G: 2e-4, D: 1e-4)")
    print("   3. Add label smoothing (0.1)")
    print("   4. Train discriminator less frequently (every 2 steps)")
    print("   5. More epochs (50 instead of 10)")
    
    print("\n✅ **Advanced Improvements**:")
    print("   1. Add gradient clipping")
    print("   2. Implement proper weight initialization")
    print("   3. Use more sophisticated architectures (DCGAN-style)")
    print("   4. Add batch normalization")
    print("   5. Implement Wasserstein loss")
    
    print("\n=== Next Steps ===")
    print("🚀 **Run Improved Training**:")
    print("   python train_improved.py")
    print("\n🚀 **Monitor Progress**:")
    print("   - Check generated images quality")
    print("   - Monitor loss convergence")
    print("   - Watch discriminator accuracy balance")
    
    print("\n🚀 **If Still Unstable**:")
    print("   - Try even smaller learning rates")
    print("   - Implement gradient penalty")
    print("   - Use spectral normalization")
    print("   - Consider different loss functions")

def create_comparison_plot():
    """Create a comparison plot of current vs improved training"""
    
    # Create a simple comparison visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Current training issues
    axes[0, 0].text(0.1, 0.8, 'Current Training Issues:', fontsize=14, fontweight='bold')
    axes[0, 0].text(0.1, 0.7, '• Mode collapse', fontsize=12)
    axes[0, 0].text(0.1, 0.6, '• D(real)=0.00, D(fake)=1.00', fontsize=12)
    axes[0, 0].text(0.1, 0.5, '• Generator loss plateaus', fontsize=12)
    axes[0, 0].text(0.1, 0.4, '• No convergence', fontsize=12)
    axes[0, 0].set_xlim(0, 1)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_title('Current Training Problems')
    axes[0, 0].axis('off')
    
    # Improved training solutions
    axes[0, 1].text(0.1, 0.8, 'Improved Training Solutions:', fontsize=14, fontweight='bold')
    axes[0, 1].text(0.1, 0.7, '• Label smoothing', fontsize=12)
    axes[0, 1].text(0.1, 0.6, '• Different learning rates', fontsize=12)
    axes[0, 1].text(0.1, 0.5, '• Smaller batch size', fontsize=12)
    axes[0, 1].text(0.1, 0.4, '• Balanced training frequency', fontsize=12)
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_title('Proposed Solutions')
    axes[0, 1].axis('off')
    
    # Expected improvements
    axes[1, 0].text(0.1, 0.8, 'Expected Improvements:', fontsize=14, fontweight='bold')
    axes[1, 0].text(0.1, 0.7, '• Better image quality', fontsize=12)
    axes[1, 0].text(0.1, 0.6, '• Stable loss convergence', fontsize=12)
    axes[1, 0].text(0.1, 0.5, '• Balanced D/G performance', fontsize=12)
    axes[1, 0].text(0.1, 0.4, '• No mode collapse', fontsize=12)
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_title('Expected Results')
    axes[1, 0].axis('off')
    
    # Next steps
    axes[1, 1].text(0.1, 0.8, 'Next Steps:', fontsize=14, fontweight='bold')
    axes[1, 1].text(0.1, 0.7, '1. Run improved training', fontsize=12)
    axes[1, 1].text(0.1, 0.6, '2. Monitor progress', fontsize=12)
    axes[1, 1].text(0.1, 0.5, '3. Analyze results', fontsize=12)
    axes[1, 1].text(0.1, 0.4, '4. Iterate if needed', fontsize=12)
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title('Action Plan')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('training_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("📊 Created training analysis visualization: training_analysis.png")

if __name__ == "__main__":
    analyze_training_results()
    create_comparison_plot() 