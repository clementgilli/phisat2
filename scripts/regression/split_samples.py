#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np

ZARR_DATASET_NAMES = {
    "roads": ("phileo-bench_roads", "roads"),
    "building": ("phileo-bench_building", "building"),
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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract n random samples for n-shot learning.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--source-folder", default="trainval")
    parser.add_argument("--n-shot", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-csv", default=None)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir).expanduser().resolve()
    dataset = args.dataset.lower()
    dataset_path = _resolve_base_path(root_dir, dataset)
    
    source_folder = dataset_path / args.source_folder
    if not source_folder.exists():
        if _list_patch_dirs(dataset_path):
            source_folder = dataset_path
        else:
            print(f"[ERROR] Directory not found: {source_folder} and no patches found in {dataset_path}")
            return

    print(f"[INFO] Scanning Zarr patches in {source_folder}...")
    
    patch_paths = _list_patch_dirs(source_folder)
    items = sorted([p.name for p in patch_paths])
    total_samples = len(items)
    
    if total_samples == 0:
        print("[ERROR] No patch directories found.")
        return

    if args.n_shot > total_samples:
        print(f"[ERROR] Requested {args.n_shot} samples but only found {total_samples}.")
        return

    rng = np.random.default_rng(args.seed)
    selected_items = rng.choice(items, size=args.n_shot, replace=False).tolist()
    
    selected_items.sort()

    if args.output_csv:
        out_csv = Path(args.output_csv)
    else:
        out_csv = Path.cwd() / f"{dataset}_{args.n_shot}shot_seed{args.seed}.csv"

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id"])
        for item in selected_items:
            writer.writerow([item])

    print(f"[INFO] Successfully selected {args.n_shot} samples out of {total_samples}.")
    print(f"[INFO] Saved to {out_csv}")

if __name__ == "__main__":
    main()