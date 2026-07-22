import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CSV_FILE = "/lustre/home/u10010021/phisat2/runs/downstream_metrics.csv"

MODELS_TO_PLOT = [
    "phisatnet",
    "terramind_v1_large"
]

TARGET_METRIC_BY_TYPE = {
    "segmentation": "iou",           
    "pixel_regression": "rmse"       
}

METRIC_LABELS = {
    "iou": "mIoU",
    "rmse": "RMSE"
}
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0

palette_colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c", "#0891b2"]
CUSTOM_PALETTE = {
    MODELS_TO_PLOT[i]: palette_colors[i] for i in range(len(MODELS_TO_PLOT))
}

def sort_shots(val):
    val_str = str(val).strip().lower()
    if val_str == "full":
        return float('inf')
    try:
        return float(val_str)
    except ValueError:
        return -1

def main():
    if not os.path.exists(CSV_FILE):
        print(f"[ERREUR] Le fichier {CSV_FILE} n'existe pas. Modifie le chemin dans le script.")
        return
        
    print(f"Chargement des données depuis {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE)
    
    df = df[df['model'].isin(MODELS_TO_PLOT)].copy()
    
    if df.empty:
        print("[ERREUR] Aucun des modèles spécifiés n'a été trouvé dans le CSV.")
        return

    df = df[df['task_type'] != 'classification'].copy()

    df['shots_sort'] = df['shots'].apply(sort_shots)
    df = df.sort_values(by=['task_name', 'shots_sort'])
    
    df['shots_str'] = df['shots'].astype(str)

    tasks = df['task_name'].unique()
    n_tasks = len(tasks)
    
    if n_tasks == 0:
        print("[ERREUR] Aucune tâche valide trouvée après filtrage.")
        return

    print(f"Génération d'une figure unique pour {n_tasks} tâches...")

    fig, axes = plt.subplots(1, n_tasks, figsize=(4.5 * n_tasks, 4.5))
    if n_tasks == 1:
        axes = [axes]

    handles_list, labels_list = [], []

    for i, task in enumerate(tasks):
        ax = axes[i]
        df_task = df[df['task_name'] == task].copy()
        
        task_type = df_task['task_type'].iloc[0]
        target_metric = TARGET_METRIC_BY_TYPE.get(task_type)
        
        if not target_metric or target_metric not in df_task.columns or df_task[target_metric].isna().all():
            print(f"[WARN] Métrique '{target_metric}' introuvable ou vide pour la tâche : {task}")
            continue

        unique_shots = df_task["shots_str"].unique()
        
        shot_to_x = {shot: idx for idx, shot in enumerate(unique_shots)}
        
        for model_name in MODELS_TO_PLOT:
            df_model = df_task[df_task["model"] == model_name]
            if df_model.empty: 
                continue
                
            color = CUSTOM_PALETTE.get(model_name, "#333333")
            
            if model_name == "phisatnet":
                model_label = "PhiSatNet"
            elif model_name == "terramind_v1_large":
                model_label = "TerraMind-Large"
                
            x_coords = df_model["shots_str"].map(shot_to_x)
            
            ax.plot(
                x_coords, df_model[target_metric], 
                marker="o", linewidth=2.5, markersize=7, 
                color=color, label=model_label
            )
            
        letter = chr(ord('a') + i)
        formatted_type = task_type.replace('_', ' ').title()
            
        ax.set_title(f"{letter}) {task.capitalize()} {formatted_type}", fontsize=12, pad=12)
        
        ax.set_xticks(range(len(unique_shots)))
        ax.set_xticklabels(unique_shots, rotation=0)
        ax.set_xlabel("n-shot", fontsize=11)
        
        metric_display_name = METRIC_LABELS.get(target_metric, target_metric.upper())
        ax.set_ylabel(metric_display_name, fontsize=11)
        
        if i == 0:
            handles_list, labels_list = ax.get_legend_handles_labels()
            
        ax.grid(True, which='major', color='#e5e5e5', linestyle='-', linewidth=0.7)

    sns.despine()

    fig.legend(handles_list, labels_list, 
               loc='lower center', 
               ncol=len(MODELS_TO_PLOT), 
               bbox_to_anchor=(0.5, -0.05),
               frameon=True, 
               fontsize=11)

    plt.tight_layout()
    output_filename = "downstream_merged_figure.pdf"
    plt.savefig(output_filename, format="pdf", bbox_inches="tight")
    print(f"[INFO] Figure panoramique générée avec succès : {output_filename}")
    plt.close()

if __name__ == "__main__":
    main()