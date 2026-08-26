import json
from pathlib import Path
import pandas as pd
    
FULL_DATASET_SIZES = {
    "lulc": 50080,
    "floods": 243904,
    "burned": 76912,
    "clouds": 35336,
    "eurosat": -1, 
}

ALLOWED_TASKS = {"eurosat", "lulc", "clouds", "floods", "burned"}
ALLOWED_SHOTS = {100,1000,10000} 
ALLOWED_MODEL = {"random", "phisatnet", "terramind_v1_large", "dofa_large_patch16_224", "ssl4eos12_vit_small_patch16_224_sentinel2_all_dino", "ssl4eos12_vit_small_patch16_224_sentinel2_all_moco", "prithvi_eo_v2_300"}

def scrape_metrics(base_dir: str | Path, output_csv: str = "downstream_metrics.csv") -> pd.DataFrame:
    base_path = Path(base_dir)
    target_dirs = ["segmentation", "classification"]
    results = []

    for task_type in target_dirs:
        type_dir = base_path / task_type
        if not type_dir.exists():
            continue

        pattern = "*/*/*/eval_seed_42/test_metrics.json"

        for filepath in type_dir.glob(pattern):
            try:
                model_name = filepath.parents[2].name
                if model_name not in ALLOWED_MODEL:
                    continue

                task_name = filepath.parents[3].name.lower()
                shot_raw = filepath.parents[1].name.lower()

                # 1. Filtre strict : est-ce que la tâche est autorisée ?
                if task_name not in ALLOWED_TASKS:
                    continue

                # 2. Vérification STRICTE du nom du dossier
                if shot_raw == "full_dataset":
                    shot = FULL_DATASET_SIZES.get(task_name, "full")
                else:
                    # On génère les noms exacts attendus (ex: "floods_split_100")
                    allowed_dirs = {f"{task_name}_split_{n}": n for n in ALLOWED_SHOTS}
                    
                    if shot_raw in allowed_dirs:
                        shot = allowed_dirs[shot_raw]
                    else:
                        # Si ce n'est ni "full_dataset" ni un de nos splits stricts, on ignore
                        continue

                with open(filepath, "r") as f:
                    metrics = json.load(f)

                record = {
                    "task_type": task_type,
                    "task_name": task_name,
                    "shots": shot,
                    "model": model_name,
                }

                for k, v in metrics.items():
                    record[k.replace("test_", "")] = v

                results.append(record)

            except Exception as e:
                print(f"[ERREUR] Échec du traitement pour {filepath}: {e}")

    if not results:
        print("[ATTENTION] Aucune métrique trouvée avec ces filtres stricts.")
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Tri pour avoir un beau CSV
    df['shots_sort'] = pd.to_numeric(df['shots'], errors='coerce').fillna(float('inf'))
    df = df.sort_values(by=["task_type", "task_name", "model", "shots_sort"])
    df = df.drop(columns=['shots_sort']).reset_index(drop=True)

    output_path = base_path / output_csv
    df.to_csv(output_path, index=False)
    print(f"[SUCCÈS] {len(df)} entrées extraites et sauvegardées dans {output_path}")

    return df

if __name__ == "__main__":
    runs_dir = "/lustre/home/u10010021/phisat2/runs"
    df_results = scrape_metrics(runs_dir)