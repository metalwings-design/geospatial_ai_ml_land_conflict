import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_transition_matrix(csv_path="data/processed/lgbm_transition_matrix_2017_2026.csv", out_dir="docs/figures"):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path, index_col=0)

    plt.figure(figsize=(8, 6))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="Blues", cbar_kws={'label': 'Area Converted (Hectares)'})
    plt.title("Land Cover Transition Heatmap (2017 Baseline -> 2026 Target)\nFermi Solar Park Case Study", fontsize=12)
    plt.xlabel("2026 Target Land Cover Class", fontsize=10)
    plt.ylabel("2017 Baseline Land Cover Class", fontsize=10)
    plt.tight_layout()
    
    heatmap_path = os.path.join(out_dir, "transition_matrix_heatmap.png")
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"Saved transition heatmap to: {heatmap_path}")

def plot_land_cover_change_bar(csv_path="data/processed/lgbm_transition_matrix_2017_2026.csv", out_dir="docs/figures"):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path, index_col=0)

    # Calculate Total Hectares per Class in 2017 vs 2026
    total_2017 = df.sum(axis=1)
    total_2026 = df.sum(axis=0)

    df_comp = pd.DataFrame({'2017 Baseline': total_2017, '2026 Target': total_2026})

    ax = df_comp.plot(kind='bar', figsize=(9, 6), color=['#1f77b4', '#ff7f0e'], width=0.7)
    plt.title("Net Land Cover Shift in Hectares (2017 vs 2026)", fontsize=12)
    plt.ylabel("Area (Hectares)", fontsize=10)
    plt.xlabel("Land Cover Class", fontsize=10)
    plt.xticks(rotation=15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}", (p.get_x() * 1.005, p.get_height() * 1.005), fontsize=8)

    plt.tight_layout()
    bar_path = os.path.join(out_dir, "land_cover_change_barchart.png")
    plt.savefig(bar_path, dpi=300)
    plt.close()
    print(f"Saved comparison bar chart to: {bar_path}")

if __name__ == "__main__":
    plot_transition_matrix()
    plot_land_cover_change_bar()