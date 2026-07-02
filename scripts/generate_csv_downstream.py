import json
from pathlib import Path
import pandas as pd

def scrape_metrics(base_dir: str | Path, output_csv: str = "downstream_metrics.csv") -> pd.DataFrame:
    base_path = Path(base_dir)
    target_dirs = ["segmentation", "pixel_regression", "classification"]
    results = []

    for task_type in target_dirs:
        type_dir = base_path / task_type
        if not type_dir.exists():
            continue

        pattern = "*/*/full_dataset/eval_seed_42/test_metrics.json"
        
        for filepath in type_dir.glob(pattern):
            try:
                model_name = filepath.parents[2].name
                task_name = filepath.parents[3].name
                
                with open(filepath, "r") as f:
                    metrics = json.load(f)

                record = {
                    "task_type": task_type,
                    "task_name": task_name,
                    "model": model_name,
                }
                
                for k, v in metrics.items():
                    record[k.replace("test_", "")] = v

                results.append(record)
            
            except Exception as e:
                print(f"Failed to process {filepath}: {e}")

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df.sort_values(by=["task_type", "task_name", "model"]).reset_index(drop=True)
    df.to_csv(base_dir + "/" + output_csv, index=False)
    
    return df

if __name__ == "__main__":
    runs_dir = "/lustre/home/u10010021/phisat2/runs"
    df_results = scrape_metrics(runs_dir)