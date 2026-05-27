#!/usr/bin/env python3
"""Extract labels for classification datasets.

The output is a simple CSV mapping sample_id to its class label.
It scans all directories first to perfectly distribute workers.
"""

import argparse
import csv
import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import numpy as np
import zarr

ZARR_DATASET_NAMES = {
    "fire": ("fire",),
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
    with os.scandir(source_folder) as entries:
        for entry in entries:
            try:
                if entry.is_dir():
                    paths.append(Path(entry.path))
            except OSError:
                continue
    return sorted(paths)

def _open_array(array_path: Path):
    for attempt in range(3):
        try:
            return zarr.open_array(str(array_path), mode="r", zarr_format=3)
        except (FileNotFoundError, ValueError, OSError):
            if attempt == 2:
                break
            time.sleep(0.2 * (attempt + 1))
    return zarr.open_array(str(array_path), mode="r")

def _read_sample_label(sample_path: str, label_key: str) -> tuple[str, int, bool, str]:
    sample_name = Path(sample_path).name
    try:
        array = _open_array(Path(sample_path) / label_key)
        data = np.asarray(array)
        label_val = int(data.ravel()[0])
        return sample_name, label_val, True, ""
    except Exception as e:
        return sample_name, -1, False, str(e)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--label-key", default="label")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir).expanduser().resolve()
    dataset = args.dataset.lower()
    dataset_path = _resolve_base_path(root_dir, dataset)
    
    source_folder = dataset_path / ("trainval" if args.split in {"train", "val"} else args.split)
    if not source_folder.exists():
        raise FileNotFoundError(f"Expected split folder at {source_folder}")

    output_csv = Path(args.output_csv) if args.output_csv else (
        Path.cwd() / f"{dataset}_{args.split}_labels.csv"
    )

    print(f"[INFO] Listing all patches in {source_folder}...")
    patch_paths = _list_patch_dirs(source_folder)
    if args.max_samples:
        patch_paths = patch_paths[:args.max_samples]
        
    total_patches = len(patch_paths)
    if total_patches == 0:
        raise FileNotFoundError("No patches found in the directory.")
        
    print(f"[INFO] Found {total_patches} patches. Starting extraction with {args.workers} workers...")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    workers = max(1, args.workers)
    processed_count = 0
    skipped_count = 0
    first_error_printed = False

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "label"])

        if workers == 1:
            for i, sample_path in enumerate(patch_paths, start=1):
                sample_name, label_val, ok, err = _read_sample_label(str(sample_path), args.label_key)
                if ok:
                    writer.writerow([sample_name, label_val])
                    processed_count += 1
                else:
                    skipped_count += 1
                    if not first_error_printed:
                        print(f"\n[DEBUG] Example of failed read on {sample_name}: {err}")
                        first_error_printed = True
                
                if i % args.progress_every == 0:
                    print(f"[INFO] {i}/{total_patches} samples...")
        else:
            ctx = multiprocessing.get_context("fork")
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
                path_strs = [str(p) for p in patch_paths]
                keys = [args.label_key] * total_patches
                
                for i, (sample_name, label_val, ok, err) in enumerate(executor.map(_read_sample_label, path_strs, keys, chunksize=100), start=1):
                    if ok:
                        writer.writerow([sample_name, label_val])
                        processed_count += 1
                    else:
                        skipped_count += 1
                        if not first_error_printed:
                            print(f"\n[DEBUG] Example of failed read on {sample_name}: {err}")
                            first_error_printed = True
                        
                    if i % args.progress_every == 0:
                        print(f"[INFO] {i}/{total_patches} samples...")

    print(f"[OK] Saved {processed_count} sample histograms to {output_csv}")
    if skipped_count:
        print(f"[WARN] Skipped {skipped_count} unreadable samples")

if __name__ == "__main__":
    main()