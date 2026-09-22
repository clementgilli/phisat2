import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pandas as pd
import matplotlib.pyplot as plt

from cvpr_style import apply_style, CVPR_TEXTWIDTH_IN, COLOR_MIM, COLOR_TERRAMIND

CSV_FILE = "/lustre/home/u10010021/phisat2/runs/downstream_metrics2.csv"

MODELS_TO_PLOT = ["phisatnet", "terramind_v1_large"]
MODEL_LABELS = {"phisatnet": "MiM baseline", "terramind_v1_large": "TerraMind KD"}
MODEL_COLORS = {"phisatnet": COLOR_MIM, "terramind_v1_large": COLOR_TERRAMIND}

TARGET_METRIC_BY_TYPE = {"segmentation": "iou", "pixel_regression": "rmse"}
METRIC_LABELS = {"iou": "mIoU", "rmse": "RMSE"}

apply_style()


def sort_shots(val):
    val_str = str(val).strip().lower()
    if val_str == "full":
        return float("inf")
    try:
        return float(val_str)
    except ValueError:
        return -1


def main():
    if not os.path.exists(CSV_FILE):
        print(f"[ERREUR] Le fichier {CSV_FILE} n'existe pas. Modifie le chemin dans le script.")
        return

    df = pd.read_csv(CSV_FILE)
    df = df[df["model"].isin(MODELS_TO_PLOT)].copy()
    if df.empty:
        print("[ERREUR] Aucun des modeles specifies n'a ete trouve dans le CSV.")
        return

    df = df[df["task_type"] != "classification"].copy()
    df["shots_sort"] = df["shots"].apply(sort_shots)
    df = df.sort_values(by=["task_name", "shots_sort"])
    df["shots_str"] = df["shots"].astype(str)

    tasks = df["task_name"].unique()
    n_tasks = len(tasks)
    if n_tasks == 0:
        print("[ERREUR] Aucune tache valide trouvee apres filtrage.")
        return

    # figsize: ~1.7in per panel (matches the per-panel width used in the
    # latent-space/performance figures for visual consistency), capped at
    # the CVPR full page width regardless of how many tasks are plotted.
    fig_width = min(CVPR_TEXTWIDTH_IN, 1.7 * n_tasks)
    fig, axes = plt.subplots(1, n_tasks, figsize=(fig_width, 2.3), sharey=False)
    if n_tasks == 1:
        axes = [axes]

    handles_list, labels_list = [], []

    for i, task in enumerate(tasks):
        ax = axes[i]
        df_task = df[df["task_name"] == task].copy()
        task_type = df_task["task_type"].iloc[0]
        target_metric = TARGET_METRIC_BY_TYPE.get(task_type)

        if not target_metric or target_metric not in df_task.columns or df_task[target_metric].isna().all():
            print(f"[WARN] Metrique '{target_metric}' introuvable ou vide pour la tache : {task}")
            continue

        unique_shots = df_task["shots_str"].unique()
        shot_to_x = {shot: idx for idx, shot in enumerate(unique_shots)}

        # The last (largest) shot count per task is the full dataset --
        # display "full" there instead of the raw sample count (e.g.
        # 76912), matching the x-axis convention used in the other
        # figures ("100, 1000, 10000, full").
        tick_display = list(unique_shots)
        if tick_display:
            tick_display[-1] = "full"

        for model_name in MODELS_TO_PLOT:
            df_model = df_task[df_task["model"] == model_name]
            if df_model.empty:
                continue
            x_coords = df_model["shots_str"].map(shot_to_x)
            ax.plot(x_coords, df_model[target_metric],
                    marker="o", linewidth=1.8, markersize=4,
                    color=MODEL_COLORS[model_name], label=MODEL_LABELS[model_name])

        letter = chr(ord("a") + i)
        formatted_type = task_type.replace("_", " ").title()
        if task != "LULC":
            ax.set_title(f"({letter}) {task.capitalize()}", pad=6)
        else:
            ax.set_title(f"({letter}) {task}", pad=6)
        ax.set_xticks(range(len(unique_shots)))
        ax.set_xticklabels(tick_display, rotation=0)
        ax.set_xlabel("n-shot")

        metric_display_name = METRIC_LABELS.get(target_metric, target_metric.upper())
        ax.set_ylabel(metric_display_name)

        if i == 0:
            handles_list, labels_list = ax.get_legend_handles_labels()

    fig.tight_layout()
    fig.legend(handles_list, labels_list, loc="lower center", ncol=len(MODELS_TO_PLOT),
               bbox_to_anchor=(0.5, 0.95), frameon=False)

    output_filename = "downstream_merged_figure.pdf"
    fig.savefig(output_filename, bbox_inches="tight")
    print(f"[INFO] Figure generee avec succes : {output_filename}")
    plt.close(fig)


if __name__ == "__main__":
    main()