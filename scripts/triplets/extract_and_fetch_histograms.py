#!/usr/bin/env python3
"""Build per-sample class histograms. If a WorldCover mask is missing, download it dynamically."""

from __future__ import annotations

import argparse
import csv
import math
import urllib.request
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from affine import Affine

DEFAULT_CLASS_PREFIX = "class_"

# ==========================================
# WORLDCOVER FETCHING LOGIC
# ==========================================

def _log_download(log_file: Path, msg: str):
    with log_file.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg)

def tile_name(lat: float, lon: float) -> str:
    lat0 = math.floor(float(lat) / 3) * 3
    lon0 = math.floor(float(lon) / 3) * 3
    ns = "N" if lat0 >= 0 else "S"
    ew = "E" if lon0 >= 0 else "W"
    return f"ESA_WorldCover_10m_2021_v200_{ns}{abs(lat0):02d}{ew}{abs(lon0):03d}_Map.tif"

def download_tile_if_needed(tile_path: Path, log_file: Path):
    if tile_path.exists() and tile_path.stat().st_size > 0:
        return
    url = f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/{tile_path.name}"
    _log_download(log_file, f"[FETCH] Downloading 3x3 degree tile: {url} -> {tile_path.name}")
    urllib.request.urlretrieve(url, tile_path)

def extract_bounds_from_s2(s2_path: Path) -> dict:
    """Reads the georeferencing metadata directly from the Sentinel-2 reference image."""
    with rasterio.open(s2_path) as src:
        bounds = src.bounds
        # Calculate center for tile matching
        center_lon = (bounds.left + bounds.right) / 2
        center_lat = (bounds.top + bounds.bottom) / 2
        
        return {
            "center_lat": center_lat, "center_lon": center_lon,
            "transform": src.transform,
            "crs": src.crs,
            "width": src.width,
            "height": src.height
        }

def create_missing_worldcover_mask(s2_path: Path, target_wc_path: Path, cache_dir: Path, log_file: Path) -> bool:
    try:
        # 1. Get exact spatial footprint from Sentinel-2 image
        meta = extract_bounds_from_s2(s2_path)
        
        # 2. Find and download the corresponding massive WorldCover tile
        tname = tile_name(meta["center_lat"], meta["center_lon"])
        tpath = cache_dir / tname
        download_tile_if_needed(tpath, log_file)
        
        # 3. Reproject and clip to match the Sentinel-2 patch
        dst_data = np.zeros((meta["height"], meta["width"]), dtype="uint8")
        
        with rasterio.open(tpath) as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=dst_data,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=0,
                dst_transform=meta["transform"],
                dst_crs=meta["crs"],
                dst_nodata=0,
                resampling=Resampling.nearest,
            )
            
        # 4. Save the new mask physically to the triplet directory
        target_wc_path.parent.mkdir(parents=True, exist_ok=True)
        
        # We save it as a GeoTIFF to maintain consistency with the rest of the dataset
        profile = {
            "driver": "GTiff",
            "height": meta["height"],
            "width": meta["width"],
            "count": 1,
            "dtype": "uint8",
            "crs": meta["crs"],
            "transform": meta["transform"],
            "compress": "lzw"
        }
        
        with rasterio.open(target_wc_path, "w", **profile) as dst:
            dst.write(dst_data, 1)
            
        _log_download(log_file, f"[SUCCESS] Generated missing mask: {target_wc_path.name}")
        return True
        
    except Exception as e:
        _log_download(log_file, f"[ERROR] Failed to generate mask for {s2_path.name}: {e}")
        return False

# ==========================================
# HISTOGRAM LOGIC
# ==========================================

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
                    "s2_path": s2_tile,
                    "wc_path": wc_mask
                })
    return samples

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create histograms and fetch missing WorldCover masks.")
    parser.add_argument("--root-dir", required=True, help="Root folder containing the 'pairs' directory.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--class-prefix", default=DEFAULT_CLASS_PREFIX)
    parser.add_argument("--output-csv", default="triplet_worldcover_histograms.csv")
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--workers", type=int, default=4, help="Keep this low if downloading heavily.")
    parser.add_argument("--wc-cache-dir", default="cache/worldcover", help="Directory to store 3x3 degree ESA tiles.")
    parser.add_argument("--log-file", default="downloaded_masks.log", help="Log file for missing masks tracking.")
    parser.add_argument("--stop-after-first-download", action="store_true", default=True, help="Safety break for testing.")
    parser.add_argument("--run-full", action="store_false", dest="stop_after_first_download", help="Run without safety break.")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir).expanduser().resolve()
    pairs_dir = root_dir / "pairs"
    cache_dir = Path(args.wc_cache_dir).expanduser().resolve()
    log_file = Path(args.log_file).expanduser().resolve()
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Indexing tiles in: {pairs_dir}")
    samples = _index_valid_tiles(pairs_dir)
    
    if args.max_samples:
        samples = samples[:args.max_samples]
        
    if not samples:
        print("[ERROR] No valid tiles found.")
        return
        
    print(f"[INFO] {len(samples)} valid tiles found. Processing...")

    rows: list[dict[str, object]] = []
    all_class_ids: set[int] = set()
    skipped_patches = 0

    for i, sample in enumerate(samples, start=1):
        tile_id = sample["tile_id"]
        wc_path = sample["wc_path"]
        s2_path = sample["s2_path"]
        
        # 🟢 MISSING MASK INTERVENTION
        if not wc_path.exists():
            success = create_missing_worldcover_mask(s2_path, wc_path, cache_dir, log_file)
            if not success:
                skipped_patches += 1
                continue
                
            if args.stop_after_first_download:
                print("\n[STOP] Safety break triggered after first download.")
                print(f"Check the generated file at: {wc_path}")
                print("If it looks good, re-run with --run-full to process the rest of the dataset.")
                sys.exit(0)

        # Normal Histogram Reading
        class_counts: dict[int, int] = defaultdict(int)
        try:
            with rasterio.open(wc_path) as src:
                data = src.read(1)
            _accumulate_histogram(data, class_counts)
            
            all_class_ids.update(class_counts)
            rows.append({
                "tile_id": tile_id,
                "total_pixels": int(data.size),
                "class_counts": class_counts,
            })
        except Exception as e:
            print(f"[WARN] Failed to read {wc_path.name}: {e}")
            skipped_patches += 1
            
        if args.progress_every > 0 and i % args.progress_every == 0:
            print(f"[INFO] Processed {i}/{len(samples)} tiles")

    if not rows:
        print("[ERROR] No histograms generated.")
        return

    # Formating Output
    class_ids = sorted(all_class_ids)
    expanded_rows = []
    for row in rows:
        class_counts = row.pop("class_counts")
        expanded_row = {"tile_id": row["tile_id"], "total_pixels": row["total_pixels"]}
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

    print(f"\n[INFO] Finished. Saved {len(expanded_rows)} histograms to {output_csv}")
    if skipped_patches:
        print(f"[WARN] Skipped {skipped_patches} unreadable masks.")

if __name__ == "__main__":
    main()