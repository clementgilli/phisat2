import json
import re
import pandas as pd
from pathlib import Path

def build_metrics_csv(data_dir: str | Path, output_csv: str = "final_results.csv"):
    base_path = Path(data_dir)
    results = []

    if not base_path.exists():
        return

    pattern = re.compile(r"^test_metrics_(.*)_([^_]+)\.json$")

    for filepath in base_path.glob("*.json"):
        match = pattern.search(filepath.name)
        if match:
            model_name = match.group(1)
            shots = match.group(2)

            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
                
                record = {
                    "model": model_name,
                    "shots": shots,
                }
                record.update(metrics)
                results.append(record)
                
        else:
            print(f"File {filepath.name} does not match the expected pattern.")

    if not results:
        return

    df = pd.DataFrame(results)

    cols = ['model', 'shots']
    other_cols = sorted([c for c in df.columns if c not in cols])
    df = df[cols + other_cols]

    df['shots_numeric'] = pd.to_numeric(df['shots'], errors='coerce').fillna(float('inf'))
    df = df.sort_values(by=["model", "shots_numeric"])
    df = df.drop(columns=['shots_numeric']).reset_index(drop=True)

    df.to_csv(output_csv, index=False)
    
    print(f"File CSV generated : {output_csv}")

if __name__ == "__main__":
    TARGET_DIR = "/lustre/home/u10010021/phisat2/runs/eval_domain_gap/triplets/phisatnet/full_dataset/eval_seed_42/" 
    OUTPUT_FILE = "domain_adaptation_final.csv"
    
    build_metrics_csv(TARGET_DIR, OUTPUT_FILE)