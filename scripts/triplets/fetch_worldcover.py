#!/usr/bin/env python3
"""
Fetch and fix WorldCover masks using CSV product corner coordinates.
Includes a debug mode for visual validation.
"""

from __future__ import annotations

import argparse
import csv
import math
import urllib.request
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping

import numpy as np
import rasterio
from rasterio.control import GroundControlPoint
from rasterio.transform import from_gcps
from rasterio.warp import reproject, Resampling
from affine import Affine

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

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

# ==========================================
# GEOLOCATION INTERPOLATION
# ==========================================

def load_coords_csv(csv_path: str) -> dict:
    """Loads the CSV mapping product_id -> corners."""
    coords = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            coords[row['product_id']] = {
                'ul_lat': float(row['ul_lat']), 'ul_lon': float(row['ul_lon']),
                'ur_lat': float(row['ur_lat']), 'ur_lon': float(row['ur_lon']),
                'll_lat': float(row['ll_lat']), 'll_lon': float(row['ll_lon']),
                'lr_lat': float(row['lr_lat']), 'lr_lon': float(row['lr_lon']),
                'width': int(row['width']),
                'height': int(row['height'])
            }
    return coords

def bilinear_interp(u: float, v: float, ul: tuple, ur: tuple, ll: tuple, lr: tuple) -> tuple:
    """Bilinear interpolation for coordinates (lat, lon)."""
    lat = (1-u)*(1-v)*ul[0] + u*(1-v)*ur[0] + (1-u)*v*ll[0] + u*v*lr[0]
    lon = (1-u)*(1-v)*ul[1] + u*(1-v)*ur[1] + (1-u)*v*ll[1] + u*v*lr[1]
    return lat, lon

def compute_tile_transform(
    product_corners: dict, 
    row_off: int, col_off: int, 
    tile_w: int, tile_h: int, 
    prod_w: int, prod_h: int,
    flip_mode: str = "h"  # <--- TESTER ICI : "v", "h", "vh" ou None
) -> tuple[Affine, float, float]:
    
    ul_lon, ul_lat = product_corners['ul_lon'], product_corners['ul_lat']
    ur_lon, ur_lat = product_corners['ur_lon'], product_corners['ur_lat']
    lr_lon, lr_lat = product_corners['lr_lon'], product_corners['lr_lat']
    ll_lon, ll_lat = product_corners['ll_lon'], product_corners['ll_lat']

    if flip_mode == "v":
        gcps = [
            GroundControlPoint(row=0, col=0, x=ll_lon, y=ll_lat),
            GroundControlPoint(row=0, col=prod_w, x=lr_lon, y=lr_lat),
            GroundControlPoint(row=prod_h, col=prod_w, x=ur_lon, y=ur_lat),
            GroundControlPoint(row=prod_h, col=0, x=ul_lon, y=ul_lat)
        ]
    elif flip_mode == "h":
        gcps = [
            GroundControlPoint(row=0, col=0, x=ur_lon, y=ur_lat),
            GroundControlPoint(row=0, col=prod_w, x=ul_lon, y=ul_lat),
            GroundControlPoint(row=prod_h, col=prod_w, x=ll_lon, y=ll_lat),
            GroundControlPoint(row=prod_h, col=0, x=lr_lon, y=lr_lat)
        ]
    elif flip_mode == "vh":
        gcps = [
            GroundControlPoint(row=0, col=0, x=lr_lon, y=lr_lat),
            GroundControlPoint(row=0, col=prod_w, x=ll_lon, y=ll_lat),
            GroundControlPoint(row=prod_h, col=prod_w, x=ul_lon, y=ul_lat),
            GroundControlPoint(row=prod_h, col=0, x=ur_lon, y=ur_lat)
        ]
    else:
        gcps = [
            GroundControlPoint(row=0, col=0, x=ul_lon, y=ul_lat),
            GroundControlPoint(row=0, col=prod_w, x=ur_lon, y=ur_lat),
            GroundControlPoint(row=prod_h, col=prod_w, x=lr_lon, y=lr_lat),
            GroundControlPoint(row=prod_h, col=0, x=ll_lon, y=ll_lat)
        ]

    global_transform = from_gcps(gcps)

    tile_transform = global_transform * Affine.translation(col_off, row_off)

    center_lon, center_lat = tile_transform * (tile_w / 2.0, tile_h / 2.0)

    return tile_transform, center_lat, center_lon

def create_missing_worldcover_mask(
    s2_path: Path, target_wc_path: Path, 
    cache_dir: Path, log_file: Path, 
    transform: Affine, center_lat: float, center_lon: float,
    width: int = 256, height: int = 256
) -> bool:
    try:
        tname = tile_name(center_lat, center_lon)
        tpath = cache_dir / tname
        download_tile_if_needed(tpath, log_file)
        
        dst_data = np.zeros((height, width), dtype="uint8")
        dst_crs = "EPSG:4326"
        
        with rasterio.open(tpath) as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=dst_data,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=0,
                dst_transform=transform,
                dst_crs=dst_crs,
                dst_nodata=0,
                resampling=Resampling.nearest,
            )
            
        target_wc_path.parent.mkdir(parents=True, exist_ok=True)
        
        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": "uint8",
            "crs": dst_crs,
            "transform": transform,
            "compress": "lzw"
        }
        
        with rasterio.open(target_wc_path, "w", **profile) as dst:
            dst.write(dst_data, 1)
            
        _log_download(log_file, f"[SUCCESS] Generated exact mask: {target_wc_path.name}")
        return True
        
    except Exception as e:
        _log_download(log_file, f"[ERROR] Failed to generate mask for {s2_path.name}: {e}")
        return False

# ==========================================
# MAIN LOGIC & DEBUG
# ==========================================

def plot_debug(s2_path: Path, wc_path: Path):
    if not HAS_MATPLOTLIB:
        print("[WARN] Matplotlib not installed, skipping debug plot.")
        return
        
    with rasterio.open(s2_path) as src:
        img = src.read([1, 2, 3]).transpose(1, 2, 0)
        img = np.clip(img / np.percentile(img, 98), 0, 1)

    with rasterio.open(wc_path) as src:
        mask = src.read(1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(img)
    ax1.set_title(f"Image: {s2_path.name}")
    ax1.axis('off')
    
    c = ax2.imshow(mask, cmap='tab20', vmin=10, vmax=100, interpolation='nearest')
    ax2.set_title("Generated WorldCover Mask")
    ax2.axis('off')
    
    plt.colorbar(c, ax=ax2, fraction=0.046, pad=0.04)
    plt.tight_layout()
    
    output_filename = f"debug_plot_{s2_path.stem}.png"
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"[DEBUG] Plot sauvegardé sous : {output_filename}")
    
    plt.close(fig)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--coords-csv", required=True, help="Path to the product coordinates CSV.")    
    parser.add_argument("--debug", action="store_true", help="Run on max 5 tiles and plot the results.")
    parser.add_argument("--wc-cache-dir", default="cache/worldcover")
    parser.add_argument("--log-file", default="downloaded_masks.log")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir).expanduser().resolve()
    pairs_dir = root_dir / "pairs"
    cache_dir = Path(args.wc_cache_dir).expanduser().resolve()
    log_file = Path(args.log_file).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print("[INFO] Loading coordinates CSV...")
    coords_dict = load_coords_csv(args.coords_csv)
    
    all_pair_paths = sorted([p for p in pairs_dir.iterdir() if p.is_dir() and p.name.startswith("pair_")])
    
    debug_count = 0
    max_debug = 10

    for pair_path in all_pair_paths:
        if args.debug and debug_count >= max_debug:
            break
            
        raw_id = pair_path.name.split("_")[1]
        
        product_id = str(int(raw_id))
        
        if product_id not in coords_dict:
            print(f"[WARN] Product ID {product_id} not found in CSV. Skipping.")
            continue
            
        prod_corners = coords_dict[product_id]
        real_dir = pair_path / "lightglue_coregistration" / "tiles" / "phisat2"
        wc_dir = pair_path / "lightglue_coregistration" / "masks" / "worldcover"
        
        if not real_dir.exists():
            continue
            
        for tile_path in real_dir.glob("*.tif"):
            parts = tile_path.stem.split("_")
            row_off = int(parts[0].replace("r", ""))
            col_off = int(parts[1].replace("c", ""))
            
            with rasterio.open(tile_path) as src:
                tile_w = src.width
                tile_h = src.height

            transform, center_lat, center_lon = compute_tile_transform(
                prod_corners, row_off, col_off, 
                tile_w, tile_h, prod_corners['width'], prod_corners['height']
            )
            
            wc_path = wc_dir / tile_path.name
            
            if args.debug or not wc_path.exists():
                success = create_missing_worldcover_mask(
                    s2_path=tile_path, 
                    target_wc_path=wc_path, 
                    cache_dir=cache_dir, 
                    log_file=log_file,
                    transform=transform,
                    center_lat=center_lat,
                    center_lon=center_lon,
                    width=tile_w, height=tile_h
                )
                
                if success and args.debug:
                    print(f"\n[DEBUG] Showing {tile_path.name}")
                    plot_debug(tile_path, wc_path)
                    debug_count += 1
                    if debug_count >= max_debug:
                        break

    print("\n[INFO] Job finished.")

if __name__ == "__main__":
    main()