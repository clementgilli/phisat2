import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CSV_FILE = "/lustre/home/u10010021/phisat2/runs/eval_domain_gap/triplets/phisatnet/full_dataset/eval_seed_42/da_nshot.csv"

MODELS = {
    "phisatnet": {"color": "#2563eb", "label": "PhiSatNet"},
    "terramind_v1_large": {"color": "#dc2626", "label": "TerraMind-Large"}
}

sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0

def sort_shots(val):
    try:
        return float(val)
    except ValueError:
        return float('inf')

def plot_metric_axis(ax, df, metric_after, metric_before=None, metric_ub=None, title="", ylabel="", x_labels=None):
   
    if metric_before and metric_before in df.columns:
        baseline_val = df[df['model'] == 'phisatnet'][metric_before].mean()
        ax.axhline(baseline_val, color='#333333', linestyle=':', linewidth=2, label='Student (Before DA)', zorder=1)

    for model_key, model_info in MODELS.items():
        df_model = df[df['model'] == model_key]
        if df_model.empty or metric_after not in df_model.columns:
            continue
            
        ax.plot(
            df_model["shots_str"], df_model[metric_after], 
            marker="o", linewidth=2.5, markersize=6, 
            color=model_info["color"], label=f"{model_info['label']} (After DA)", zorder=3
        )
        
        if metric_ub and metric_ub in df_model.columns:
            ub_val = df_model[metric_ub].mean()
            ax.axhline(ub_val, color=model_info["color"], linestyle='--', linewidth=2, alpha=0.5, 
                       label=f"Upper Bound ({model_info['label']})", zorder=2)

    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)
        
    ax.set_xlabel("Number of Shots", fontsize=11)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.grid(True, which='major', color='#e5e5e5', linestyle='-', linewidth=0.7)

def main():
    if not os.path.exists(CSV_FILE):
        print(f"[ERREUR] Fichier introuvable : {CSV_FILE}")
        return
        
    df = pd.read_csv(CSV_FILE)
    df = df[df['model'].isin(MODELS.keys())].copy()
    
    # Tri des shots
    df['shots_sort'] = df['shots'].apply(sort_shots)
    df = df.sort_values(['model', 'shots_sort'])
    df['shots_str'] = df['shots'].astype(str)
    x_labels = df['shots_str'].unique()

    print("Génération de Figure 1 : Latent Space Dynamics...")
    
    fig1, axes1 = plt.subplots(2, 4, figsize=(18, 8))
    layers = ["enc_0", "enc_1", "enc_2", "bottleneck"]
    layer_names = ["Enc 0", "Enc 1", "Enc 2", "Bottleneck"]
    
    for i, (layer, name) in enumerate(zip(layers, layer_names)):
        plot_metric_axis(axes1[0, i], df, 
                         metric_after=f"after/cosine_{layer}", 
                         metric_before=f"before/cosine_{layer}", 
                         title=name, 
                         ylabel="Cosine Similarity" if i == 0 else "", 
                         x_labels=x_labels)
        plot_metric_axis(axes1[1, i], df, 
                         metric_after=f"after/pad_{layer}", 
                         metric_before=f"before/pad_{layer}", 
                         title="", 
                         ylabel="Proxy A-Distance" if i == 0 else "", 
                         x_labels=x_labels)

    sns.despine(fig=fig1)
    handles, labels = axes1[0, 0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig1.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.05), frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig1.savefig("fig_latent_space.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig1)

    
    fig2, axes2 = plt.subplots(2, 4, figsize=(18, 8))
    
    tasks = ["burned", "clouds", "floods", "lulc"]
    task_names = ["Burned Area", "Clouds", "Floods", "LULC"]
    for i, (task, name) in enumerate(zip(tasks, task_names)):
        plot_metric_axis(axes2[0, i], df, 
                         metric_after=f"after/consistency_{task}_miou", 
                         metric_before=f"before/consistency_{task}_miou", 
                         title=name, 
                         ylabel="Consistency mIoU" if i == 0 else "", 
                         x_labels=x_labels)

    bottom_metrics = [
        ("after/lulc_iou", "before/lulc_iou", "upper_bound/lulc_iou", "LULC Micro (True mIoU)", "True mIoU"),
        ("after/lulc_macro_iou", "before/lulc_macro_iou", "upper_bound/lulc_macro_iou", "LULC Macro (True mIoU)", ""),
        ("after/consistency_building_mse", "before/consistency_building_mse", None, "Building (Consistency MSE)", "Consistency MSE"),
        ("after/consistency_roads_mse", "before/consistency_roads_mse", None, "Roads (Consistency MSE)", "")
    ]
    
    for i, (m_after, m_before, m_ub, name, ylabel) in enumerate(bottom_metrics):
        plot_metric_axis(axes2[1, i], df, 
                         metric_after=m_after, 
                         metric_before=m_before, 
                         metric_ub=m_ub, 
                         title=name, 
                         ylabel=ylabel, 
                         x_labels=x_labels)

    sns.despine(fig=fig2)
    handles, labels = axes2[1, 0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig2.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.05), frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig2.savefig("fig_performance.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig2)

    print("[SUCCÈS] Fichiers générés : fig_latent_space.pdf, fig_performance.pdf")

if __name__ == "__main__":
    main()