import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

BASE = "D:/weld_defect_project/segmentation_model"
OUT = f"{BASE}/plots/analysis"
Path(OUT).mkdir(exist_ok=True)

# Name mappings
MODEL_NAMES = {
    'neu': 'NEU-DET',
    'dagm': 'DAGM',
    'sev': 'SeverstalSteel',
    'gokul': 'Custom Model',
    'final': 'Final Model'
}

DATASET_NAMES = {
    'neu': 'NEU-DET',
    'dagm': 'DAGM',
    'sev': 'SeverstalSteel',
    'gokul': 'Custom Set',
    'gdx': 'Final Dataset'
}

# ========== TRAINING PLOTS ========== #

def plot_training_curves(model_key):
    df = pd.read_csv(f"{BASE}/logs/{model_key}.csv")
    model_name = MODEL_NAMES[model_key]
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    
    # Loss
    axes[0].plot(df['epoch'], df['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(df['epoch'], df['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_title(f'{model_name} - Training Progress', fontsize=14, fontweight='bold')
    
    # Dice
    axes[1].plot(df['epoch'], df['dice'], color='green', linewidth=2)
    axes[1].set_ylabel('Dice Score', fontsize=12)
    axes[1].grid(alpha=0.3)
    
    # F1 (calculated)
    f1 = 2 * df['precision'] * df['recall'] / (df['precision'] + df['recall'] + 1e-7)
    axes[2].plot(df['epoch'], f1, color='orange', linewidth=2)
    axes[2].set_ylabel('F1 Score', fontsize=12)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUT}/{model_key}_training_main.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_metrics_detail(model_key):
    df = pd.read_csv(f"{BASE}/logs/{model_key}.csv")
    model_name = MODEL_NAMES[model_key]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Precision + Recall
    axes[0,0].plot(df['epoch'], df['precision'], label='Precision', linewidth=2)
    axes[0,0].plot(df['epoch'], df['recall'], label='Recall', linewidth=2)
    axes[0,0].set_ylabel('Score', fontsize=11)
    axes[0,0].set_xlabel('Epoch', fontsize=11)
    axes[0,0].legend()
    axes[0,0].grid(alpha=0.3)
    axes[0,0].set_title('Precision & Recall', fontsize=12)
    
    # Dice empty vs non-empty
    axes[0,1].plot(df['epoch'], df['dice_empty'], label='Empty', linewidth=2)
    axes[0,1].plot(df['epoch'], df['dice_non_empty'], label='Non-Empty', linewidth=2)
    axes[0,1].set_ylabel('Dice Score', fontsize=11)
    axes[0,1].set_xlabel('Epoch', fontsize=11)
    axes[0,1].legend()
    axes[0,1].grid(alpha=0.3)
    axes[0,1].set_title('Dice: Empty vs Non-Empty', fontsize=12)
    
    # Pred empty ratio
    axes[1,0].plot(df['epoch'], df['pred_empty_ratio'], color='purple', linewidth=2)
    axes[1,0].set_ylabel('Ratio', fontsize=11)
    axes[1,0].set_xlabel('Epoch', fontsize=11)
    axes[1,0].grid(alpha=0.3)
    axes[1,0].set_title('Prediction Empty Ratio', fontsize=12)
    
    # Loss components
    axes[1,1].plot(df['epoch'], df['bce_loss'], label='BCE', linewidth=2)
    axes[1,1].plot(df['epoch'], df['dice_loss'], label='Dice', linewidth=2)
    axes[1,1].plot(df['epoch'], df['focal_loss'], label='Focal', linewidth=2)
    axes[1,1].set_ylabel('Loss', fontsize=11)
    axes[1,1].set_xlabel('Epoch', fontsize=11)
    axes[1,1].legend()
    axes[1,1].grid(alpha=0.3)
    axes[1,1].set_title('Loss Components', fontsize=12)
    
    fig.suptitle(f'{model_name} - Detailed Metrics', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(f"{OUT}/{model_key}_training_detail.png", dpi=300, bbox_inches='tight')
    plt.close()

# ========== TEST PLOTS ========== #

def plot_final_vs_custom_comparison():
    # Read test results on final dataset
    final_df = pd.read_csv(f"{BASE}/logs/final_gdx_own.csv")
    custom_df = pd.read_csv(f"{BASE}/logs/gokul_gdx_cross.csv")
    
    metrics = ['dice', 'iou', 'precision', 'recall', 'f1']
    final_vals = [final_df[m].values[0] for m in metrics]
    custom_vals = [custom_df[m].values[0] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, final_vals, width, label='Final Model', color='#2ecc71')
    ax.bar(x + width/2, custom_vals, width, label='Custom Model', color='#3498db')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Final Model vs Custom Model - Performance on Final Dataset', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics], fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    # Add value labels
    for i, (fv, cv) in enumerate(zip(final_vals, custom_vals)):
        ax.text(i - width/2, fv + 0.02, f'{fv:.3f}', ha='center', fontsize=9)
        ax.text(i + width/2, cv + 0.02, f'{cv:.3f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{OUT}/final_vs_custom_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_cross_domain_heatmap():
    models = ['final', 'gokul']
    datasets = ['neu', 'dagm', 'sev', 'gokul', 'gdx']
    
    dice_matrix = []
    
    for model in models:
        row = []
        for ds in datasets:
            if model == 'final' and ds == 'gdx':
                path = f"{BASE}/logs/final_gdx_own.csv"
            elif model == 'gokul' and ds == 'gokul':
                path = f"{BASE}/logs/gokul_gokul_own.csv"
            elif model == 'gokul' and ds == 'gdx':
                path = f"{BASE}/logs/gokul_gdx_cross.csv"
            else:
                path = f"{BASE}/logs/{model}_{ds}_cross.csv"
            
            df = pd.read_csv(path)
            row.append(df['dice'].values[0])
        dice_matrix.append(row)
    
    dice_matrix = np.array(dice_matrix)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(dice_matrix, annot=True, fmt='.3f', cmap='RdYlGn', 
                xticklabels=[DATASET_NAMES[d] for d in datasets],
                yticklabels=[MODEL_NAMES[m] for m in models],
                vmin=0, vmax=1, cbar_kws={'label': 'Dice Score'},
                ax=ax, linewidths=1, linecolor='black')
    
    ax.set_title('Cross-Domain Generalization - Dice Scores', fontsize=14, fontweight='bold')
    ax.set_xlabel('Test Dataset', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{OUT}/cross_domain_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_empty_vs_nonempty_scatter():
    models = ['neu', 'dagm', 'sev', 'gokul', 'final']
    datasets_map = {
        'neu': 'neu_neu_own',
        'dagm': 'dagm_dagm_own',
        'sev': 'sev_sev_own',
        'gokul': 'gokul_gokul_own',
        'final': 'final_gdx_own'
    }
    
    empty_scores = []
    nonempty_scores = []
    labels = []
    
    for model in models:
        path = f"{BASE}/logs/{datasets_map[model]}.csv"
        df = pd.read_csv(path)
        empty_scores.append(df['dice_empty'].values[0])
        nonempty_scores.append(df['dice_non_empty'].values[0])
        labels.append(MODEL_NAMES[model])
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    colors = ['#e74c3c', '#f39c12', '#9b59b6', '#3498db', '#2ecc71']
    
    for i, (e, ne, label) in enumerate(zip(empty_scores, nonempty_scores, labels)):
        ax.scatter(e, ne, s=200, alpha=0.7, color=colors[i], label=label, edgecolors='black', linewidth=1.5)
    
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
    ax.set_xlabel('Dice Score (Empty Masks)', fontsize=12)
    ax.set_ylabel('Dice Score (Non-Empty Masks)', fontsize=12)
    ax.set_title('Empty vs Non-Empty Performance', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(f"{OUT}/empty_vs_nonempty_scatter.png", dpi=300, bbox_inches='tight')
    plt.close()

# ========== CURRICULUM PROGRESSION ========== #

def plot_curriculum_progression():
    models = ['neu', 'dagm', 'sev', 'gokul', 'final']
    datasets_map = {
        'neu': 'neu_neu_own',
        'dagm': 'dagm_dagm_own',
        'sev': 'sev_sev_own',
        'gokul': 'gokul_gokul_own',
        'final': 'final_gdx_own'
    }
    
    dice_scores = []
    iou_scores = []
    f1_scores = []
    
    for model in models:
        path = f"{BASE}/logs/{datasets_map[model]}.csv"
        df = pd.read_csv(path)
        dice_scores.append(df['dice'].values[0])
        iou_scores.append(df['iou'].values[0])
        f1_scores.append(df['f1'].values[0])
    
    x = np.arange(len(models))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(x, dice_scores, marker='o', linewidth=2, markersize=10, label='Dice', color='#2ecc71')
    ax.plot(x, iou_scores, marker='s', linewidth=2, markersize=10, label='IoU', color='#3498db')
    ax.plot(x, f1_scores, marker='^', linewidth=2, markersize=10, label='F1', color='#e74c3c')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xlabel('Curriculum Stage', fontsize=12)
    ax.set_title('Curriculum Learning Progression', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_NAMES[m] for m in models], fontsize=10, rotation=15, ha='right')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    plt.savefig(f"{OUT}/curriculum_progression.png", dpi=300, bbox_inches='tight')
    plt.close()

# ========== MAIN ========== #

print("Generating training plots...")
for model in ['neu', 'dagm', 'sev', 'gokul', 'final']:
    print(f"  {model}...")
    plot_training_curves(model)

print("\nGenerating detailed metric plots for final model...")
plot_metrics_detail('final')

print("\nGenerating test comparison plots...")
plot_final_vs_custom_comparison()
plot_cross_domain_heatmap()
plot_empty_vs_nonempty_scatter()
plot_curriculum_progression()

print(f"\nDone! Plots saved to {OUT}/")