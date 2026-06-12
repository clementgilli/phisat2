from __future__ import annotations

import argparse
import csv
from pathlib import Path
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import rasterio

# Order of bands exactly as they appear in the TIF files
BANDS_ORDER = {
    "phisat2_sim": [
        "BLUE", "GREEN", "RED", "PAN", 
        "NIR_BROAD", "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3"
    ],
    "phisat2_real": [
        "PAN", "BLUE", "GREEN", "RED", 
        "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3", "NIR_BROAD"
    ],
    "s2": [
        "COASTAL_AEROSOL", "BLUE", "GREEN", "RED", 
        "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3", "NIR_BROAD",
        "NIR_NARROW", "WATER_VAPOR", "CIRRUS", "SWIR_1", "SWIR_2"
    ]
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute dataset statistics (Mean/Std) from train split.")
    parser.add_argument("--root-dir", required=True, help="Root folder containing the 'pairs' directory.")
    parser.add_argument("--train-csv", required=True, help="CSV containing the tile_ids for the train split.")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers.")
    return parser.parse_args()

def _read_train_tiles(train_csv: Path) -> list[str]:
    tile_ids = []
    with train_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            tile_ids.append(row["tile_id"])
    return tile_ids

def _process_single_tile(args: tuple[Path, str, list[str]]) -> dict[str, tuple[np.ndarray, np.ndarray, int]]:
    """Reads the 3 modalities for a single tile and returns sum, sum_sq, and pixel count per band."""
    root_dir, tile_id, valid_pairs = args
    pairs_dir = root_dir / "pairs"
    
    # Robustly find the correct pair_id and tile_name
    pair_id = None
    tile_name = None
    for p_id in valid_pairs:
        if tile_id.startswith(p_id + "_"):
            pair_id = p_id
            tile_name = tile_id[len(p_id) + 1:] # +1 to skip the underscore
            break
            
    if pair_id is None:
        print(f"[ERROR] Could not resolve pair_id for tile: {tile_id}")
        return {}
    
    base_tiles = pairs_dir / pair_id / "lightglue_coregistration" / "tiles"
    
    paths = {
        "phisat2_sim": base_tiles / "simulated_phisat2" / tile_name,
        "phisat2_real": base_tiles / "phisat2" / tile_name,
        "s2": base_tiles / "sentinel2" / tile_name
    }
    
    stats = {}
    for modality, path in paths.items():
        if not path.exists():
            print(f"[WARN] Missing file: {path}")
            continue
            
        try:
            with rasterio.open(path) as src:
                data = src.read().astype(np.float64) # [C, H, W]
                
            # Keep only the first 8 bands for real phisat2 (as per your dataset logic)
            if modality == "phisat2_real":
                data = data[:8, :, :]
                
            # Flatten spatial dimensions: [C, H*W]
            data_flat = data.reshape(data.shape[0], -1)
            
            # Compute partial sums for Welford's algorithm
            band_sum = np.sum(data_flat, axis=1)
            band_sum_sq = np.sum(data_flat ** 2, axis=1)
            num_pixels = data_flat.shape[1]
            
            stats[modality] = (band_sum, band_sum_sq, num_pixels)
            
        except Exception as e:
            print(f"[ERROR] Corrupted file {path}: {e}")
            
    return stats

def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir).expanduser().resolve()
    train_csv = Path(args.train_csv).expanduser().resolve()
    pairs_dir = root_dir / "pairs"
    
    print(f"[INFO] Reading training tiles from: {train_csv}")
    tile_ids = _read_train_tiles(train_csv)
    print(f"[INFO] Found {len(tile_ids)} tiles in the training set.")
    
    # Pre-compute valid pair names to pass to workers
    print("[INFO] Indexing pair directories...")
    valid_pairs = [p.name for p in pairs_dir.iterdir() if p.is_dir() and p.name.startswith("pair_")]
    
    # Global accumulators
    global_stats = {
        "phisat2_sim": {"sum": None, "sum_sq": None, "count": 0},
        "phisat2_real": {"sum": None, "sum_sq": None, "count": 0},
        "s2": {"sum": None, "sum_sq": None, "count": 0}
    }
    
    workers = max(1, args.workers)
    ctx = multiprocessing.get_context("fork")
    
    tasks = [(root_dir, tid, valid_pairs) for tid in tile_ids]
    
    print("[INFO] Computing statistics. This may take a while...")
    
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
        for i, partial_stats in enumerate(executor.map(_process_single_tile, tasks), 1):
            
            for modality, (b_sum, b_sum_sq, p_count) in partial_stats.items():
                if global_stats[modality]["sum"] is None:
                    global_stats[modality]["sum"] = np.zeros_like(b_sum)
                    global_stats[modality]["sum_sq"] = np.zeros_like(b_sum_sq)
                
                global_stats[modality]["sum"] += b_sum
                global_stats[modality]["sum_sq"] += b_sum_sq
                global_stats[modality]["count"] += p_count
                
            if i % 1000 == 0:
                print(f"[INFO] Processed {i}/{len(tile_ids)} tiles")

    # Final Computation: Mean = sum / N | Variance = (sum_sq / N) - (Mean^2)
    print("\n[INFO] Computation finished. Formatting output...\n")
    
    output_dict = "STATS = {\n"
    
    for modality in ["phisat2_sim", "phisat2_real", "s2"]:
        g_sum = global_stats[modality]["sum"]
        g_sum_sq = global_stats[modality]["sum_sq"]
        N = global_stats[modality]["count"]
        
        if N == 0:
            print(f"[WARN] No data processed for {modality}.")
            continue
            
        mean = g_sum / N
        variance = (g_sum_sq / N) - (mean ** 2)
        
        # Clip variance to 0 to avoid floating point inaccuracies causing negative values
        variance = np.maximum(variance, 0)
        std = np.sqrt(variance)
        
        band_names = BANDS_ORDER[modality]
        
        output_dict += f'    "{modality}": {{\n        '
        
        band_strings = []
        for b_idx, b_name in enumerate(band_names):
            m_val = round(float(mean[b_idx]), 4)
            s_val = round(float(std[b_idx]), 4)
            band_strings.append(f'"{b_name}": ({m_val}, {s_val})')
            
        # Group by 4 bands per line for readability
        formatted_lines = []
        for i in range(0, len(band_strings), 4):
            formatted_lines.append(", ".join(band_strings[i:i+4]))
            
        output_dict += ",\n        ".join(formatted_lines)
        output_dict += "\n    },\n"
        
    output_dict = output_dict.rstrip(",\n") + "\n}"
    
    print(output_dict)
    
    # Save to file
    out_file = train_csv.parent / f"{train_csv.stem}_normalization_stats.py"
    with out_file.open("w", encoding="utf-8") as f:
        f.write(output_dict + "\n")
        
    print(f"\n[INFO] Dictionary saved to {out_file}")

if __name__ == "__main__":
    main()