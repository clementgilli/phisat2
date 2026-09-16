import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CSV_FILE = "/lustre/home/u10010021/phisat2/runs/eval_domain_gap/triplets/phisatnet/full_dataset/eval_seed_42/domain_adaptation_final.csv"

MODELS = {
    "phisatnet": {"color": "#2563eb", "label": "MiM baseline"},
    "terramind_v1_large": {"color": "#dc2626", "label": "TerraMind KD"}
}

LEGEND_ORDER = [
    "MiM baseline (Before DA)", 
    "TerraMind KD (Before DA)", 
    "MiM baseline (After DA)", 
    "TerraMind KD (After DA)", 
    "MiM baseline (Reference Source Baseline)",
    "TerraMind KD (Reference Source Baseline)"
]

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
   
    for model_key, model_info in MODELS.items():
        df_model = df[df['model'] == model_key]
        if df_model.empty:
            continue
            
        if metric_before and metric_before in df_model.columns:
            baseline_val = df_model[metric_before].mean()
            ax.axhline(baseline_val, color=model_info["color"], linestyle=':', linewidth=2, alpha=0.7,
                       label=f"{model_info['label']} (Before DA)", zorder=1)

        if metric_ub and metric_ub in df_model.columns:
            ub_val = df_model[metric_ub].mean()
            ax.axhline(ub_val, color=model_info["color"], linestyle='--', linewidth=2, alpha=0.5, 
                       label=f"{model_info['label']} (Reference Source Baseline)", zorder=2)

        if metric_after in df_model.columns:
            ax.plot(
                df_model["shots_str"], df_model[metric_after], 
                marker="o", linewidth=2.5, markersize=6, 
                color=model_info["color"], label=f"{model_info['label']} (After DA)", zorder=3
            )

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
    
    df['shots_sort'] = df['shots'].apply(sort_shots)
    df = df.sort_values(['model', 'shots_sort'])
    df['shots_str'] = df['shots'].astype(str)
    x_labels = df['shots_str'].unique()

    # ==========================================
    # FIGURE 1 : LATENT SPACE
    # ==========================================
    print("Génération de Figure 1 : Latent Space Dynamics...")
    fig1, axes1 = plt.subplots(2, 4, figsize=(18, 6))
    layers = ["enc_0", "enc_1", "enc_2", "bottleneck"]
    layer_names = ["Enc 0", "Enc 1", "Enc 2", "Bottleneck"]
    
    for i, (layer, name) in enumerate(zip(layers, layer_names)):
        plot_metric_axis(axes1[0, i], df, metric_after=f"after/cosine_{layer}", metric_before=f"before/cosine_{layer}", title=name, ylabel="Cosine Similarity" if i == 0 else "", x_labels=x_labels)
        plot_metric_axis(axes1[1, i], df, metric_after=f"after/pad_{layer}", metric_before=f"before/pad_{layer}", title="", ylabel="Proxy A-Distance" if i == 0 else "", x_labels=x_labels)

    sns.despine(fig=fig1)
    
    by_label_1 = {}
    for ax in fig1.axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            by_label_1[label] = handle
            
    handles_1 = [by_label_1[l] for l in LEGEND_ORDER if l in by_label_1]
    labels_1 = [l for l in LEGEND_ORDER if l in by_label_1]
            
    fig1.legend(handles_1, labels_1, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.08), frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    fig1.savefig("fig_latent_space.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig1)

    # ==========================================
    # FIGURE 2 : DOWNSTREAM
    # ==========================================
    print("Génération de Figure 2 : Downstream Performance...")
    fig2, axes2 = plt.subplots(2, 3, figsize=(14, 7))
    
    top_metrics = [
        ("after/consistency_burned_miou", "before/consistency_burned_miou", None, "Burned Area Cons. (mIoU)", "Consistency"),
        ("after/consistency_clouds_miou", "before/consistency_clouds_miou", None, "Clouds Cons. (mIoU)", ""),
        ("after/consistency_floods_miou", "before/consistency_floods_miou", None, "Floods Cons. (mIoU)", "")
    ]
    for i, (m_after, m_before, m_ub, name, ylabel) in enumerate(top_metrics):
        plot_metric_axis(axes2[0, i], df, metric_after=m_after, metric_before=m_before, metric_ub=m_ub, title=name, ylabel=ylabel, x_labels=x_labels)

    bottom_metrics = [
        ("after/consistency_lulc_miou", "before/consistency_lulc_miou", None, "LULC Cons. (mIoU)", "Consistency / True GT"),
        ("after/consistency_eurosat_acc", "before/consistency_eurosat_acc", None, "EuroSAT Cons. (Accuracy)", ""),
        ("after/lulc_iou", "before/lulc_iou", "upper_bound/lulc_iou", "LULC Micro GT (True mIoU)", "")
    ]
    for i, (m_after, m_before, m_ub, name, ylabel) in enumerate(bottom_metrics):
        plot_metric_axis(axes2[1, i], df, metric_after=m_after, metric_before=m_before, metric_ub=m_ub, title=name, ylabel=ylabel, x_labels=x_labels)

    sns.despine(fig=fig2)
    
    by_label_2 = {}
    for ax in fig2.axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            by_label_2[label] = handle
            
    handles_2 = [by_label_2[l] for l in LEGEND_ORDER if l in by_label_2]
    labels_2 = [l for l in LEGEND_ORDER if l in by_label_2]
            
    fig2.legend(handles_2, labels_2, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.08), frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    fig2.savefig("fig_performance.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig2)

    print("[SUCCÈS] Fichiers générés : fig_latent_space.pdf, fig_performance.pdf")

if __name__ == "__main__":
    main()