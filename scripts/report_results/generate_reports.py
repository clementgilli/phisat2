import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def generate_reports(csv_path: str):
    df = pd.read_csv(csv_path)

    df['primary_metric'] = df.apply(
        lambda row: row['iou'] if row['task_type'] == 'segmentation' 
        else (row['rmse'] if row['task_type'] == 'pixel_regression' else row['f1']),
        axis=1
    )

    pivot_df = df.pivot_table(
        index="model", 
        columns="task_name", 
        values="primary_metric",
        aggfunc="mean"
    )

    normalized_df = pivot_df.copy()
    
    regression_tasks = df[df['task_type'] == 'pixel_regression']['task_name'].unique()

    for col in normalized_df.columns:
        col_min = normalized_df[col].min()
        col_max = normalized_df[col].max()
        
        if col_max != col_min:
            norm_values = (normalized_df[col] - col_min) / (col_max - col_min)
            if col in regression_tasks:
                normalized_df[col] = 1.0 - norm_values
            else:
                normalized_df[col] = norm_values
        else:
            normalized_df[col] = 0.5

    print(pivot_df.round(4))

    plt.figure(figsize=(14, 8))
    sns.heatmap(
        data=normalized_df, 
        annot=pivot_df, 
        cmap="viridis", 
        fmt=".3f", 
        linewidths=.5, 
        cbar=False
    )
    plt.title("Downstream Tasks Performance (Column-Normalized)")
    plt.tight_layout()
    plt.savefig("downstream_heatmap_normalized.png", dpi=300)

if __name__ == "__main__":
    generate_reports("/lustre/home/u10010021/phisat2/runs/downstream_metrics.csv")