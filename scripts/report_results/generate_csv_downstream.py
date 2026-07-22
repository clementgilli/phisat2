import json
import re
from pathlib import Path
import pandas as pd

FULL_DATASET_SIZES = {
    "lulc": 183596,
    "building": 114213,
    "roads": 76892,
    "floods": 61622,
    "burned": 7783,
    "clouds": 6860,
}

def scrape_metrics(base_dir: str | Path, output_csv: str = "downstream_metrics.csv") -> pd.DataFrame:
    base_path = Path(base_dir)
    target_dirs = ["segmentation", "pixel_regression", "classification"]
    results = []

    for task_type in target_dirs:
        type_dir = base_path / task_type
        if not type_dir.exists():
            continue

        pattern = "*/*/*/eval_seed_42/test_metrics.json"
        
        for filepath in type_dir.glob(pattern):
            try:
                model_name = filepath.parents[2].name
                task_name = filepath.parents[3].name
                shot_raw = filepath.parents[1].name.lower()
                
                if "full" in shot_raw:
                    shot = FULL_DATASET_SIZES.get(task_name.lower(), "full")
                else:
                    match = re.search(r'\d+', shot_raw)
                    if match:
                        shot = int(match.group())
                    else:
                        shot = shot_raw

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
        print("[ATTENTION] Aucune métrique trouvée.")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    
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