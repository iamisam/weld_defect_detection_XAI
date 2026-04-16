import pandas as pd
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
BASE_FOLDER = "final_report_tables"
BASELINE_CSV = os.path.join(BASE_FOLDER, "baseline_model_table.csv")
OPTIMIZED_CSV = os.path.join(BASE_FOLDER, "optimized_model_table.csv")

def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in mpl_table.get_celld().items():
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax

def process_and_plot(csv_path, title, output_name):
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        return

    # 1. Read CSV
    df = pd.read_csv(csv_path)
    if 'Accuracy (%)' in df.columns:
        df['Accuracy (%)'] = df['Accuracy (%)'].apply(lambda x: f"{x:.2f}%")

    # 2. Setup Plot
    rows, cols = df.shape
    fig_height = rows * 0.8 + 2 # Base height + per row
    fig_width = cols * 2.5      # Width per column
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)

    render_mpl_table(df, header_color='#2c3e50', row_colors=['#ecf0f1', '#ffffff'], ax=ax)

    # 4. Save
    save_path = os.path.join(BASE_FOLDER, output_name)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Table image saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    import numpy as np # Needed for size calculation
    
    print("Generating table images...")
    
    # Plot Baseline
    process_and_plot(
        BASELINE_CSV, 
        "Baseline Model Performance (ResNet18)", 
        "baseline_table_plot.png"
    )
    
    # Plot Optimized
    process_and_plot(
        OPTIMIZED_CSV, 
        "Optimized Model Performance (Fine-Tuned)", 
        "optimized_table_plot.png"
    )