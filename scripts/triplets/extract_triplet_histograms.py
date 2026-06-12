from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import rasterio

DEFAULT_CLASS_PREFIX = "class_"

def _accumulate_histogram(values: np.ndarray, class_counts: Mapping[int, int]) -> None:
    if values.size == 0:
        return
    
    flat = np.ravel(values).astype(np.int64, copy=False)
    flat = flat[flat >= 0]
    
    if flat.size == 0:
        return
    
    unique, counts = np.unique(flat, return_counts=True)
    for cls, count in zip(unique, counts):
        class_counts[int(cls)] += int(count)

def _index_valid_tiles(pairs_dir: Path) -> list[dict[str, str | Path]]:
    samples = []
    if not pairs_dir.exists():
        print(f"[ERROR] Directory {pairs_dir} not found.")
        return samples

    all_pair_paths = sorted([p for p in pairs_dir.iterdir() if p.is_dir() and p.name.startswith("pair_")])
    print(f"[INFO] Found {len(all_pair_paths)} 'pair_' directories.")

    for pair_path in all_pair_paths:
        base_tiles = pair_path / "lightglue_coregistration"
        sim_dir = base_tiles / "tiles" / "simulated_phisat2"
        real_dir = base_tiles / "tiles" / "phisat2"
        s2_dir = base_tiles / "tiles" / "sentinel2"
        wc_dir = base_tiles / "masks" / "worldcover"

        if not sim_dir.exists():
            continue

        for sim_tile in sim_dir.glob("*.tif"):
            tile_name = sim_tile.name
            real_tile = real_dir / tile_name
            s2_tile = s2_dir / tile_name
            wc_mask = wc_dir / tile_name

            if real_tile.exists() and s2_tile.exists():
                samples.append({
                    "tile_id": f"{pair_path.name}_{tile_name}",
                    "wc_path": wc_mask if wc_mask.exists() else None
                })

    return samples

def _read_sample_histogram(sample: dict) -> tuple[str, int, dict[int, int], bool]:
    tile_id = sample["tile_id"]
    wc_path = sample["wc_path"]
    class_counts: dict[int, int] = defaultdict(int)

    if wc_path is None:
        return tile_id, 0, class_counts, True

    try:
        with rasterio.open(wc_path) as src:
            data = src.read(1)
        _accumulate_histogram(data, class_counts)
        return tile_id, int(data.size), class_counts, True
    except Exception:
        return tile_id, 0, {}, False

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create per-sample class histograms for Triplet datasets.")
    parser.add_argument("--root-dir", required=True, help="Root folder containing the 'pairs' directory.")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap for the number of samples to process.")
    parser.add_argument("--class-prefix", default=DEFAULT_CLASS_PREFIX, help="Prefix for class columns in CSV.")
    parser.add_argument("--output-csv", default="triplet_worldcover_histograms.csv", help="Output path for CSV.")
    parser.add_argument("--progress-every", type=int, default=1000, help="Log progress every N samples.")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers.")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir).expanduser().resolve()
    pairs_dir = root_dir / "pairs"
    
    print(f"[INFO] Indexing tiles in: {pairs_dir}")
    samples = _index_valid_tiles(pairs_dir)
    
    if args.max_samples:
        samples = samples[:args.max_samples]
        
    if not samples:
        print("[ERROR] No valid tiles found.")
        return
        
    print(f"[INFO] {len(samples)} valid tiles found. Starting histogram extraction...")

    rows: list[dict[str, object]] = []
    all_class_ids: set[int] = set()
    skipped_patches = 0

    workers = max(1, args.workers)
    ctx = multiprocessing.get_context("fork")
    
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
        for i, (tile_id, total_pixels, class_counts, ok) in enumerate(
            executor.map(_read_sample_histogram, samples), start=1
        ):
            if not ok:
                skipped_patches += 1
                continue
                
            all_class_ids.update(class_counts)
            rows.append({
                "tile_id": tile_id,
                "total_pixels": total_pixels,
                "class_counts": class_counts,
            })
            
            if args.progress_every > 0 and i % args.progress_every == 0:
                print(f"[INFO] Processed {i}/{len(samples)} tiles")

    if not rows:
        print("[ERROR] No histograms generated.")
        return

    class_ids = sorted(all_class_ids)
    expanded_rows = []
    
    for row in rows:
        class_counts = row.pop("class_counts")
        expanded_row = {
            "tile_id": row["tile_id"],
            "total_pixels": row["total_pixels"],
        }
        for cls in class_ids:
            expanded_row[f"{args.class_prefix}{cls}"] = class_counts.get(cls, 0)
        expanded_rows.append(expanded_row)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = ["tile_id", "total_pixels"] + [f"{args.class_prefix}{cls}" for cls in class_ids]
    
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(expanded_rows)

    print(f"[INFO] Finished. Saved {len(expanded_rows)} histograms to {output_csv}")
    if skipped_patches:
        print(f"[WARN] Skipped {skipped_patches} unreadable masks.")

if __name__ == "__main__":
    main()