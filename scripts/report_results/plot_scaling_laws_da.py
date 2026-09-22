import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pandas as pd
import matplotlib.pyplot as plt

from cvpr_style import apply_style, CVPR_TEXTWIDTH_IN, COLOR_MIM, COLOR_TERRAMIND

CSV_FILE = "/lustre/home/u10010021/phisat2/runs/eval_domain_gap/triplets/phisatnet/full_dataset/eval_seed_42/domain_adaptation_final.csv"

MODELS = {
    "phisatnet": {"color": COLOR_MIM, "label": "MiM baseline"},
    "terramind_v1_large": {"color": COLOR_TERRAMIND, "label": "TerraMind KD"},
}

LEGEND_ORDER = [
    "MiM baseline (Before DA)",
    "TerraMind KD (Before DA)",
    "MiM baseline (After DA)",
    "TerraMind KD (After DA)",
    "MiM baseline (Reference Source Baseline)",
    "TerraMind KD (Reference Source Baseline)",
]

apply_style()


def sort_shots(val):
    try:
        return float(val)
    except ValueError:
        return float("inf")


def plot_metric_axis(ax, df, metric_after, metric_before=None, metric_ub=None,
                      title="", ylabel="", x_labels=None):
    for model_key, model_info in MODELS.items():
        df_model = df[df["model"] == model_key]
        if df_model.empty:
            continue

        if metric_before and metric_before in df_model.columns:
            baseline_val = df_model[metric_before].mean()
            ax.axhline(baseline_val, color=model_info["color"], linestyle=":",
                        linewidth=1.4, alpha=0.7,
                        label=f"{model_info['label']} (Before DA)", zorder=1)

        if metric_ub and metric_ub in df_model.columns:
            ub_val = df_model[metric_ub].mean()
            ax.axhline(ub_val, color=model_info["color"], linestyle="--",
                        linewidth=1.4, alpha=0.5,
                        label=f"{model_info['label']} (Reference Source Baseline)", zorder=2)

        if metric_after in df_model.columns:
            ax.plot(df_model["shots_str"], df_model[metric_after],
                     marker="o", linewidth=1.8, markersize=4,
                     color=model_info["color"], label=f"{model_info['label']} (After DA)",
                     zorder=3)

    ax.set_title(title, pad=6)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_xlabel("Number of shots")
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right")


def collect_ordered_legend(fig):
    by_label = {}
    for ax in fig.axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            by_label[label] = handle
    handles = [by_label[l] for l in LEGEND_ORDER if l in by_label]
    labels = [l for l in LEGEND_ORDER if l in by_label]
    return handles, labels


def main():
    if not os.path.exists(CSV_FILE):
        print(f"[ERREUR] Fichier introuvable : {CSV_FILE}")
        return

    df = pd.read_csv(CSV_FILE)
    df = df[df["model"].isin(MODELS.keys())].copy()

    df["shots_sort"] = df["shots"].apply(sort_shots)
    df = df.sort_values(["model", "shots_sort"])
    df["shots_str"] = df["shots"].astype(str)
    x_labels = df["shots_str"].unique()

    # ==========================================
    # FIGURE 1: LATENT SPACE  -- 2 rows x 4 cols
    # ==========================================
    # figsize: full CVPR page width, height tuned so each of the 8 panels
    # stays near-square once shrunk into the page (rather than the
    # original 18x6in on-screen size, ~2.7x wider than the printed page).
    fig1, axes1 = plt.subplots(2, 4, figsize=(CVPR_TEXTWIDTH_IN, 3.3))
    layers = ["enc_0", "enc_1", "enc_2", "bottleneck"]
    layer_names = ["Enc 0", "Enc 1", "Enc 2", "Bottleneck"]

    for i, (layer, name) in enumerate(zip(layers, layer_names)):
        plot_metric_axis(axes1[0, i], df, metric_after=f"after/cosine_{layer}",
                          metric_before=f"before/cosine_{layer}", title=name,
                          ylabel="Cosine similarity" if i == 0 else "", x_labels=x_labels)
        plot_metric_axis(axes1[1, i], df, metric_after=f"after/pad_{layer}",
                          metric_before=f"before/pad_{layer}", title="",
                          ylabel="Proxy A-Distance" if i == 0 else "", x_labels=x_labels)

    handles_1, labels_1 = collect_ordered_legend(fig1)
    # Only 4 entries here (no "Reference Source Baseline" for this figure),
    # so ncol=2 gives a clean 2x2 grid with no leftover/orphan entry --
    # avoids the awkward 3-and-1 wrap that ncol=3 produced.
    fig1.legend(handles_1, labels_1, loc="upper center", ncol=2,
                bbox_to_anchor=(0.5, 1.1), frameon=False)
    fig1.tight_layout(rect=[0, 0, 1, 0.97])
    fig1.savefig("fig_latent_space.pdf", bbox_inches="tight")
    plt.close(fig1)

    # ==========================================
    # FIGURE 2: DOWNSTREAM CONSISTENCY -- 2 rows x 3 cols
    # ==========================================
    fig2, axes2 = plt.subplots(2, 3, figsize=(CVPR_TEXTWIDTH_IN, 4.0))

    top_metrics = [
        ("after/consistency_burned_miou", "before/consistency_burned_miou", None,
         "Burned area cons. (mIoU)", "Consistency"),
        ("after/consistency_clouds_miou", "before/consistency_clouds_miou", None,
         "Clouds cons. (mIoU)", ""),
        ("after/consistency_floods_miou", "before/consistency_floods_miou", None,
         "Floods cons. (mIoU)", ""),
    ]
    for i, (m_after, m_before, m_ub, name, ylabel) in enumerate(top_metrics):
        plot_metric_axis(axes2[0, i], df, metric_after=m_after, metric_before=m_before,
                          metric_ub=m_ub, title=name, ylabel=ylabel, x_labels=x_labels)

    # NOTE: LULC "true" values come from ESA WorldCover, a derived product,
    # not manually annotated ground truth -- labeled "Ref." (reference), not
    # "GT", to avoid overclaiming. Mirror this wording in Table 4's caption
    # in the main text, which currently says "LULC Ground Truth".
    bottom_metrics = [
        ("after/consistency_lulc_miou", "before/consistency_lulc_miou", None,
         "LULC cons. (mIoU)", "Consistency / ref."),
        ("after/consistency_eurosat_acc", "before/consistency_eurosat_acc", None,
         "EuroSAT cons. (accuracy)", ""),
        ("after/lulc_iou", "before/lulc_iou", "upper_bound/lulc_iou",
         "LULC vs. WorldCover ref. (mIoU)", ""),
    ]
    for i, (m_after, m_before, m_ub, name, ylabel) in enumerate(bottom_metrics):
        plot_metric_axis(axes2[1, i], df, metric_after=m_after, metric_before=m_before,
                          metric_ub=m_ub, title=name, ylabel=ylabel, x_labels=x_labels)

    handles_2, labels_2 = collect_ordered_legend(fig2)
    fig2.legend(handles_2, labels_2, loc="upper center", ncol=3,
                bbox_to_anchor=(0.5, 1.08), frameon=False)
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    fig2.savefig("fig_performance.pdf", bbox_inches="tight")
    plt.close(fig2)

    print("[SUCCES] Fichiers generes : fig_latent_space.pdf, fig_performance.pdf")


if __name__ == "__main__":
    main()