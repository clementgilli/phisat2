#!/usr/bin/env python3
"""Stratify classification samples to create a representative N-shot subset.

This script takes the full CSV of labels and extracts a subset of N samples
while preserving the exact class proportions of the original dataset.
"""

import argparse
import pandas as pd
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a stratified N-shot subset for classification.")
    parser.add_argument("--labels-csv", required=True, help="Input CSV containing 'sample_id' and 'label'.")
    parser.add_argument(
        "--n-shot",
        type=int,
        required=True,
        help="Target size for the subset (e.g., 50 for a 50-shot).",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility.")
    parser.add_argument("--sample-id-col", default="sample_id", help="Name of the sample ID column.")
    parser.add_argument("--label-col", default="label", help="Name of the target label column.")
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Output CSV with selected samples (default auto-generated).",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Output JSON summary with class distributions.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    input_path = Path(args.labels_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Cannot find input file: {input_path}")
        
    print(f"[INFO] Loading dataset from {input_path}...")
    df = pd.read_csv(input_path)
    
    total_samples = len(df)
    
    if args.n_shot >= total_samples:
        raise ValueError(f"Requested n-shot ({args.n_shot}) is >= total dataset size ({total_samples}).")

    print(f"[INFO] Total samples: {total_samples} | Target subset: {args.n_shot}")
    
    subset_df, _ = train_test_split(
        df, 
        train_size=args.n_shot, 
        random_state=args.seed, 
        stratify=df[args.label_col]
    )
    
    base_name = input_path.stem
    out_csv = Path(args.output_csv) if args.output_csv else input_path.with_name(f"{base_name}_{args.n_shot}_global.csv")
    out_json = Path(args.summary-json) if args.summary_json else input_path.with_name(f"{base_name}_{args.n_shot}_global.json")
    
    subset_df.to_csv(out_csv, index=False)
    
    orig_counts = df[args.label_col].value_counts(normalize=True).to_dict()
    subset_counts = subset_df[args.label_col].value_counts(normalize=True).to_dict()
    
    orig_abs = df[args.label_col].value_counts().to_dict()
    subset_abs = subset_df[args.label_col].value_counts().to_dict()
    
    summary = {
        "metadata": {
            "source_file": str(input_path),
            "n_shot": args.n_shot,
            "seed": args.seed,
            "total_original_samples": total_samples
        },
        "distributions": {
            "original_proportions": {str(k): round(v, 4) for k, v in orig_counts.items()},
            "subset_proportions": {str(k): round(v, 4) for k, v in subset_counts.items()},
            "subset_absolute_counts": {str(k): int(v) for k, v in subset_abs.items()},
            "original_absolute_counts": {str(k): int(v) for k, v in orig_abs.items()}
        }
    }
    
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
        
    print("\n=========================================")
    print(f"[OK] Stratification successful!")
    print(f"-> Selected file saved to: {out_csv}")
    print(f"-> Summary saved to: {out_json}")
    print("\nQuick Distribution Check (Subset vs Original):")
    for cls in subset_abs.keys():
        print(f"   Class {cls}: {subset_counts[cls]*100:.1f}% (Subset) vs {orig_counts[cls]*100:.1f}% (Original) -> {subset_abs[cls]} samples")
    print("=========================================")

if __name__ == "__main__":
    main()