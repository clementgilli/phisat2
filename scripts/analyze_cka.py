import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_cka(base_dir: str | Path):
    base_path = Path(base_dir) / "knowledge_distillation" / "triplets"
    files = list(base_path.rglob("cka_matrix.csv"))
    
    data = {}
    for f in files:
        model_name = f.parents[2].name
        data[model_name] = pd.read_csv(f, index_col=0)

    if not data:
        return

    top_models = [
        "satlas_resnet50_sentinel2_si_ms_satlas",
        "ssl4eos12_resnet50_sentinel2_all_decur",
        "ssl4eos12_resnet50_sentinel2_all_dino",
        #"ssl4eos12_resnet18_sentinel2_all_moco",
        "ssl4eos12_resnet50_sentinel2_all_moco",
        #"ssl4eos12_resnet50_sentinel2_all_softcon",
        #"seco_resnet18_sentinel2_rgb_seco",
        #"seco_resnet50_sentinel2_rgb_seco",
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, model in enumerate(top_models):
        if model in data:
            sns.heatmap(
                data[model], 
                ax=axes[i], 
                cmap="viridis", 
                vmin=0.0, 
                vmax=1.0, 
                cbar=(i % 2 != 0),
                annot=True,
                fmt=".2f",
                annot_kws={"size": 8}
            )
            clean_title = model.replace("_sentinel2", "").replace("_resnet50", "").upper()
            axes[i].set_title(clean_title, fontweight="bold")
            axes[i].set_xlabel("Teacher Layers")
            axes[i].set_ylabel("Student Layers")
            
    plt.tight_layout()
    plt.savefig("cka_heatmaps_grid.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    for model, df in data.items():
        diagonal = np.diag(df.values)
        clean_label = model.replace("ssl4eos12_resnet50_sentinel2_all_", "").replace("satlas_resnet50_sentinel2_si_ms_", "").upper()
        plt.plot(df.index, diagonal, marker="o", linewidth=2, label=clean_label)
        
    plt.title("Student-Teacher Alignment (CKA Diagonal Mapping)", fontsize=14, fontweight="bold")
    plt.ylabel("CKA Similarity Score", fontsize=12)
    plt.xlabel("Student Layer Depth", fontsize=12)
    plt.ylim(0, 1.05)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("cka_diagonal_lines.png", dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    analyze_cka("/lustre/home/u10010021/phisat2/runs")