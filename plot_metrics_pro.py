import pandas as pd
import matplotlib.pyplot as plt
import os

# --- CONFIG ---
LOG_FILE = os.path.join("results_optimized", "training_logs_optimized.csv")
OUTPUT_IMAGE = os.path.join("results_optimized", "training_graphs_pro.png")

def plot_logs():
    if not os.path.exists(LOG_FILE):
        print(f"Error: Could not find {LOG_FILE}.")
        print("Make sure you have run 'train_pro.py' completely.")
        return

    # Load Data
    df = pd.read_csv(LOG_FILE)
    df.set_index('epoch', inplace=True)
    
    print(f"Found {len(df)} epochs of data.")

    # Setup Plots (1 Row, 3 Columns)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Graph 1: Loss (important for overfitting)
    axes[0].plot(df.index, df['train_loss'], label='Train Loss', marker='o', color='blue')
    axes[0].plot(df.index, df['val_loss'], label='Val Loss', marker='o', color='red', linestyle='--')
    axes[0].set_title('Loss vs Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (Lower is Better)')
    axes[0].legend()
    axes[0].grid(True, which='both', linestyle='--', linewidth=0.5)

    # Graph 2: Accuracy
    axes[1].plot(df.index, df['train_acc'], label='Train Acc', marker='o', color='green')
    axes[1].plot(df.index, df['val_acc'], label='Val Acc', marker='o', color='orange', linestyle='--')
    axes[1].set_title('Accuracy vs Epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (Higher is Better)')
    axes[1].legend()
    axes[1].grid(True, which='both', linestyle='--', linewidth=0.5)

    # Graph 3: Learning Rate Decay
    axes[2].plot(df.index, df['learning_rate'], label='Learning Rate', color='purple', linestyle='-')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_yscale('log')
    axes[2].legend()
    axes[2].grid(True, which='both', linestyle='--', linewidth=0.5)

    # 3. Save
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"Graph saved successfully to: {OUTPUT_IMAGE}")
    plt.show()

if __name__ == "__main__":
    plot_logs()