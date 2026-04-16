import pandas as pd
import matplotlib.pyplot as plt
import os

# CONFIGURATION 
LOG_FILE = os.path.join("results", "training_logs.csv")
OUTPUT_IMAGE = os.path.join("results", "training_graphs.png")

def plot_logs():
    if not os.path.exists(LOG_FILE):
        print(f"Error: Could not find {LOG_FILE}. Did you run training first?")
        return

    # 1. Read the CSV data
    df = pd.read_csv(LOG_FILE)
    
    # Set the 'epoch' column as the index so the x-axis is correct
    df.set_index('epoch', inplace=True)

    # 2. Setup the Plot (1 row, 3 columns)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Graph 1: Loss Curve (Train vs Val)
    axes[0].plot(df.index, df['train_loss'], label='Train Loss', marker='o', linestyle='-')
    axes[0].plot(df.index, df['val_loss'], label='Val Loss', marker='o', linestyle='--')
    axes[0].set_title('Loss Over Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Graph 2: Accuracy Curve (Train vs Val)
    axes[1].plot(df.index, df['train_acc'], label='Train Acc', color='green', marker='o')
    axes[1].plot(df.index, df['val_acc'], label='Val Acc', color='orange', marker='o', linestyle='--')
    axes[1].set_title('Accuracy Over Epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    # Graph 3: Detailed Metrics (Precision/Recall/F1)
    axes[2].plot(df.index, df['val_precision'], label='Precision', color='purple', linestyle=':')
    axes[2].plot(df.index, df['val_recall'], label='Recall', color='brown', linestyle=':')
    axes[2].plot(df.index, df['val_f1'], label='F1 Score', color='red', marker='x')
    axes[2].set_title('Validation Metrics')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Score')
    axes[2].legend()
    axes[2].grid(True)

    # 3. Save and Show
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE)
    print(f"Graphs saved to {OUTPUT_IMAGE}")
    plt.show()

if __name__ == "__main__":
    plot_logs()