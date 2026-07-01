import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = "runs/eval_encoder"

def parse_metrics():
    base_path = Path(BASE_DIR)
    data = []

    for json_file in base_path.rglob("test_metrics.json"):
        parts = json_file.parts
        if len(parts) < 5:
            continue
            
        dataset = parts[-5]
        model_name = parts[-4]

        with open(json_file, 'r') as f:
            metrics = json.load(f)
            if isinstance(metrics, list):
                metrics = metrics[0]

            row = {"Dataset": dataset, "Model": model_name}
            row.update(metrics)
            data.append(row)

    if not data:
        print("Error: No test_metrics.json files found.")
        return None
    
    return pd.DataFrame(data)

def generate_summary_and_plots(df):
    datasets = df['Dataset'].unique()
    sns.set_theme(style="whitegrid")
    layers = ["enc_0", "enc_1", "enc_2", "bottleneck"]

    for ds in datasets:
        print(f"\n{'='*50}")
        print(f"GLOBAL SUMMARY: {ds.upper()}")
        print(f"{'='*50}")
        
        df_ds = df[df['Dataset'] == ds].copy()
        df_ds = df_ds.dropna(axis=1, how='all')
        
        csv_filename = f"{BASE_DIR}/metrics_summary_{ds}.csv"
        df_ds.to_csv(csv_filename, index=False)
        print(f"Successfully generated {csv_filename}")
        
        for shot in [1, 5]:
            target_cols = [
                c for c in df_ds.columns 
                if f'{shot}_shot' in c and 'macro' not in c and any(l in c for l in layers) and ds in c
            ]
            
            if len(target_cols) == 4:
                avg_col_name = f'Global_{shot}_Shot_Avg'
                df_ds[avg_col_name] = df_ds[target_cols].mean(axis=1)
                
                cols_to_show = ['Model', avg_col_name] + target_cols
                summary_table = df_ds[cols_to_show].sort_values(by=avg_col_name, ascending=False)
                
                print(f"\n--- {shot}-Shot Evaluation ---")
                print(summary_table[['Model', avg_col_name]].to_string(index=False))
                
                plt.figure(figsize=(14, 7))
                ax = sns.barplot(
                    data=summary_table, 
                    x=avg_col_name, 
                    y='Model', 
                    hue='Model', 
                    palette='magma',
                    legend=False
                )
                
                plt.title(f"Global Distillation Score ({shot}-Shot Avg across 4 layers) - {ds.upper()}", fontsize=16, fontweight='bold')
                plt.xlabel("Average Accuracy", fontsize=12)
                plt.ylabel("", fontsize=12)
                plt.xlim(0, 1.0)
                
                for p in ax.patches:
                    width = p.get_width()
                    if width > 0:
                        ax.annotate(f"{width:.4f}", 
                                    (width + 0.01, p.get_y() + p.get_height() / 2.), 
                                    va='center', fontsize=11, fontweight='bold')

                plt.tight_layout()
                plot_filename = f"{BASE_DIR}/plot_global_{shot}shot_avg_{ds}.png"
                plt.savefig(plot_filename, dpi=300)
                print(f"Saved plot: {plot_filename}")
                plt.close()
            else:
                print(f"Warning: Could not find all 4 layers for {shot}-shot computation. Found: {target_cols}")

if __name__ == "__main__":
    df_metrics = parse_metrics()
    if df_metrics is not None:
        generate_summary_and_plots(df_metrics)