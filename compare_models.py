import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# CONFIGURATION
NORMAL_CSV = os.path.join("results", "detailed_test_metrics_adv_gradcam.csv")
OPTIMIZED_CSV = os.path.join("results_optimized", "detailed_test_metrics_pro_gradcam.csv")
OUTPUT_IMAGE = "model_comparison_charts.png"

def plot_comparison():
    # 1. Load Data
    if not os.path.exists(NORMAL_CSV) or not os.path.exists(OPTIMIZED_CSV):
        print("Error: Could not find CSV files.")
        print(f"Checking: {NORMAL_CSV}")
        print(f"Checking: {OPTIMIZED_CSV}")
        print("Make sure you ran 'final_evaluation.py' for BOTH models first.")
        return

    df_normal = pd.read_csv(NORMAL_CSV)
    df_opt = pd.read_csv(OPTIMIZED_CSV)

    # 2. Add 'Model' column to distinguish them
    df_normal['Model'] = 'Baseline (ResNet18)'
    df_opt['Model'] = 'Optimized (Fine-Tuned)'

    # 3. Combine DataFrames
    df_combined = pd.concat([df_normal, df_opt])

    # 5. Setup Plotting (3 Metrics: Recall, Precision, F1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    sns.set_theme(style="whitegrid")

    metrics_to_plot = [
        ('Recall (Sensitivity)', 'Recall (Safety) - "Did we find it?"'),
        ('Precision', 'Precision - "Is it really a defect?"'),
        ('F1-Score', 'F1-Score - "Overall Balance"')
    ]

    print("Generating comparison plots...")

    for i, (metric, title) in enumerate(metrics_to_plot):
        sns.barplot(
            data=df_combined, 
            x='Class', 
            y=metric, 
            hue='Model', 
            ax=axes[i],
            palette=['#95a5a6', '#2ecc71'] # Grey for Baseline, Green for Optimized
        )
        
        axes[i].set_title(title, fontsize=14, fontweight='bold')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Score (0-1)')
        axes[i].set_ylim(0, 1.05)
        
        # Rotate x-labels if you have many classes
        axes[i].tick_params(axis='x', rotation=45)
        
        # Add values on top of bars
        for container in axes[i].containers:
            axes[i].bar_label(container, fmt='%.2f', padding=3, fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"Comparison graph saved to {OUTPUT_IMAGE}")
    plt.show()

if __name__ == "__main__":
    plot_comparison()