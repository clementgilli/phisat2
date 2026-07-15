import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
import os

CSV_FILE = "/lustre/home/u10010021/phisat2/runs/eval_domain_gap/triplets/phisatnet/full_dataset/eval_seed_42/da_nshot.csv"

sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0

PLOT_GROUPS = {
    "01_consistency_mIoU": [
        ("Burned Area", "consistency_burned_miou"),
        ("Clouds", "consistency_clouds_miou"),
        ("Floods", "consistency_floods_miou"),
        ("LULC", "consistency_lulc_miou"),
    ],
    "02_consistency_KL_Div": [
        ("Burned", "consistency_burned_kl"),
        ("Clouds", "consistency_clouds_kl"),
        ("Floods", "consistency_floods_kl"),
        ("LULC", "consistency_lulc_kl"),
        ("Router", "consistency_router_kl"),
    ],
    "03_downstream_Accuracy_MSE": [
        ("Router (Accuracy)", "consistency_router_acc"),
        ("Building (MSE)", "consistency_building_mse"),
        ("Roads (MSE)", "consistency_roads_mse"),
    ],
    "04_LULC_Absolute": [
        ("LULC Micro (mIoU)", "lulc_iou"),
        ("LULC Macro (mIoU)", "lulc_macro_iou"),
        ("LULC Micro (F1)", "lulc_f1"),
        ("LULC Macro (F1)", "lulc_macro_f1"),
    ],
    "05_Latent_Cosine": [
        ("Enc 0", "cosine_enc_0"),
        ("Enc 1", "cosine_enc_1"),
        ("Enc 2", "cosine_enc_2"),
        ("Bottleneck", "cosine_bottleneck"),
    ],
    "06_Latent_PAD": [
        ("Enc 0", "pad_enc_0"),
        ("Enc 1", "pad_enc_1"),
        ("Enc 2", "pad_enc_2"),
        ("Bottleneck", "pad_bottleneck"),
    ]
}

def plot_panoramic_group(df: pd.DataFrame, group_name: str, metrics: list):
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 4))
    
    if n_metrics == 1:
        axes = [axes]
        
    handles_list, labels_list = [], []

    for i, (title, metric_suffix) in enumerate(metrics):
        ax = axes[i]
        
        col_after = f"after/{metric_suffix}"
        col_before = f"before/{metric_suffix}"
        col_ub = f"upper_bound/{metric_suffix}"
        
        if col_before in df.columns:
            baseline_val = df[col_before].mean()
            line1 = ax.axhline(baseline_val, color="crimson", linestyle="--", linewidth=2, label="Student (Before DA)")
            
        if col_ub in df.columns:
            ub_val = df[col_ub].mean()
            line2 = ax.axhline(ub_val, color="forestgreen", linestyle=":", linewidth=2.5, label="Teacher (Upper Bound)")
            
        if col_after in df.columns:
            line3 = ax.plot(
                df["shots"], df[col_after], 
                marker="o", linewidth=2.5, markersize=7, 
                color="royalblue", label="Student (After DA)"
            )[0]
        else:
            print(f"[WARN] Colonne manquante : {col_after}")
            
        ax.set_xscale("log")
        ax.set_xticks(df["shots"])
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        ax.xaxis.set_major_formatter(formatter)
        ax.set_xticklabels(df["shots"], rotation=45)
        
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel("Number of Shots", fontsize=11)
        
        if i == 0:
            y_label = "Score / Metric"
            if "mIoU" in group_name: y_label = "mIoU"
            elif "KL" in group_name: y_label = "KL Divergence"
            elif "Cosine" in group_name: y_label = "Cosine Similarity"
            elif "PAD" in group_name: y_label = "Proxy A-Distance"
            ax.set_ylabel(y_label, fontsize=11)
        else:
            ax.set_ylabel("")
            
        if i == 0:
            handles_list, labels_list = ax.get_legend_handles_labels()
            
        ax.grid(True, which='major', color='#dddddd', linestyle='-')
        ax.grid(False, which='minor')

    sns.despine()

    fig.legend(handles_list, labels_list, 
               loc='upper center', 
               ncol=3, 
               bbox_to_anchor=(0.5, 1.12),
               frameon=False, 
               fontsize=11)

    plt.tight_layout()
    
    output_filename = f"plot_{group_name}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"[INFO] Graphique panoramique généré : {output_filename}")
    plt.close()

def main():
    if not os.path.exists(CSV_FILE):
        print(f"[ERREUR] Le fichier {CSV_FILE} n'existe pas dans le répertoire courant.")
        return
        
    print(f"Chargement des données depuis {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE)
    df = df.sort_values(by="shots")
    
    print("Génération des graphiques par familles (Format Panorama)...")
    for group_name, metrics in PLOT_GROUPS.items():
        plot_panoramic_group(df, group_name, metrics)
        
    print("Tous les graphiques ont été générés avec succès !")

if __name__ == "__main__":
    main()