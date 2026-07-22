#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np

ZARR_DATASET_NAMES = {
    "clouds": ("clouds_dataset", "clouds", "cloud_segmentation"),
}

def _resolve_base_path(root_dir: Path, dataset: str) -> Path:
    if root_dir.suffix == ".zarr":
        return root_dir
    dataset_names = ZARR_DATASET_NAMES.get(dataset, (dataset,))
    for dataset_name in dataset_names:
        candidate = root_dir / f"{dataset_name}.zarr"
        if candidate.exists():
            return candidate
    return root_dir / f"{dataset}.zarr"

def _list_patch_dirs(source_folder: Path) -> list[Path]:
    paths: list[Path] = []
    try:
        with os.scandir(source_folder) as entries:
            for entry in entries:
                try:
                    if entry.is_dir():
                        paths.append(Path(entry.path))
                except OSError:
                    continue
    except OSError:
        pass
    return sorted(paths)

def write_csv(output_path: Path, items: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id"])
        for item in sorted(items):
            writer.writerow([item])

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split dataset into trainval/test and generate n-shot subsets.")
    parser.add_argument("--dataset", default="clouds", help="Dataset name.")
    parser.add_argument("--root-dir", default=".", help="Root directory containing the zarr datasets.")
    parser.add_argument("--out-dir", default=".", help="Directory to save the generated CSVs.")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Ratio of data to use for trainval (default: 0.9).")
    parser.add_argument("--shots", type=int, nargs="+", default=[50, 100, 500, 1000, 5000], help="List of n-shot sizes to extract.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible shuffling.")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    dataset = args.dataset.lower()
    dataset_path = _resolve_base_path(root_dir, dataset)
    
    if not dataset_path.exists():
        print(f"[ERROR] Zarr dataset not found at {dataset_path}")
        return

    scan_folder = dataset_path / "trainval"
    if not scan_folder.exists():
        scan_folder = dataset_path
        
    print(f"[INFO] Scanning Zarr patches in {scan_folder}...")
    patch_paths = _list_patch_dirs(scan_folder)
    items = sorted([p.name for p in patch_paths])
    total_samples = len(items)
    
    if total_samples == 0:
        print(f"[ERROR] No patch directories found in {scan_folder}.")
        return

    rng = np.random.default_rng(args.seed)
    shuffled_items = items.copy()
    rng.shuffle(shuffled_items)

    split_idx = int(total_samples * args.train_ratio)
    trainval_items = shuffled_items[:split_idx]
    test_items = shuffled_items[split_idx:]

    print(f"[INFO] Total samples: {total_samples}")
    print(f"[INFO] Trainval split: {len(trainval_items)} samples")
    print(f"[INFO] Test split: {len(test_items)} samples")

    write_csv(out_dir / f"{dataset}_trainval.csv", trainval_items)
    write_csv(out_dir / f"{dataset}_test.csv", test_items)

    for n in sorted(args.shots):
        if n > len(trainval_items):
            print(f"[WARN] Requested {n}-shot, but trainval only has {len(trainval_items)} samples. Skipping.")
            continue
        
        n_shot_items = trainval_items[:n]
        write_csv(out_dir / f"{dataset}_{n}shot.csv", n_shot_items)
        print(f"[INFO] Generated {n}-shot subset.")

    print(f"[INFO] All splits successfully saved to {out_dir}")

if __name__ == "__main__":
    main()