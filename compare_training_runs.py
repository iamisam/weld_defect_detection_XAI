import pandas as pd
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
NORMAL_LOG = os.path.join("results", "training_logs.csv")
OPTIMIZED_LOG = os.path.join("results_optimized", "training_logs_optimized.csv")
OUTPUT_IMAGE = "training_comparison.png"

def compare_training():
    # 1. Load Data
    if not os.path.exists(NORMAL_LOG) or not os.path.exists(OPTIMIZED_LOG):
        print("Error: Could not find one of the log files.")
        print(f"Checking: {NORMAL_LOG}")
        print(f"Checking: {OPTIMIZED_LOG}")
        return

    df_norm = pd.read_csv(NORMAL_LOG)
    df_opt = pd.read_csv(OPTIMIZED_LOG)

    # 2. Setup Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Graph A: Accuracy Comparison
    axes[0].plot(df_norm['epoch'], df_norm['val_acc'], 
                 label='Baseline (ResNet18)', color='grey', linestyle='--', linewidth=2)
    axes[0].plot(df_opt['epoch'], df_opt['val_acc'], 
                 label='Optimized (Fine-Tuned)', color='green', linewidth=2.5)
    
    axes[0].set_title('Validation Accuracy Improvement', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Accuracy (0-1)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Graph B: Loss Comparison
    axes[1].plot(df_norm['epoch'], df_norm['train_loss'], 
                 label='Baseline Loss', color='grey', linestyle='--', linewidth=2)
    axes[1].plot(df_opt['epoch'], df_opt['train_loss'], 
                 label='Optimized Loss', color='red', linewidth=2.5)
    
    axes[1].set_title('Training Convergence Speed', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Loss (Lower is Better)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. Save
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"Comparison graphs saved to {OUTPUT_IMAGE}")
    plt.show()

if __name__ == "__main__":
    compare_training()