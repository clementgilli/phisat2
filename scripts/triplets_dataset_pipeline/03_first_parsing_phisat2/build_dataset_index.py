import os
import json
import rasterio
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# --- CONFIGURATION ---
L1_DIR = Path("/shared/projects/phisat2/data/raw/phisat2/L1")
OUTPUT_CSV = Path("/shared/projects/phisat2/data/index/phisat2_metadata.csv")

def parse_datetime(dt_str):
    """Convert '20260413192234' to a readable datetime object."""
    try:
        return datetime.strptime(dt_str, "%Y%m%d%H%M%S")
    except ValueError:
        return None

def process_product(folder_path):
    
    folder_name = folder_path.name
    
    parts = folder_name.split('_')
    
    row = {
        "product_id": parts[2] if len(parts) > 2 else "UNKNOWN",
        "start_date": parse_datetime(parts[3]) if len(parts) > 3 else None,
        "end_date": parse_datetime(parts[4]) if len(parts) > 4 else None,
        "is_corrupted": False, # TODO later
        "cloud_cover": None,   # TODO later
        "min_lat": None, "max_lat": None,
        "min_lon": None, "max_lon": None,
        "folder_path": str(folder_path) 
    }
    
    json_path = folder_path / "geolocation" / "GL_scene_0.json"
    if json_path.exists():
        try:
            with open(json_path, 'r') as f:
                geo_data = json.load(f)
                
            points = geo_data.get("Geolocated_Points", [])
            if points:
                lats = [pt["Lat"] for pt in points]
                lons = [pt["Lon"] for pt in points]
                
                row["min_lat"] = min(lats)
                row["max_lat"] = max(lats)
                row["min_lon"] = min(lons)
                row["max_lon"] = max(lons)
        except Exception:
            pass 
        
    tiff_path = folder_path / "bands" / "scene_0_BC_multiband.tiff"
    if not tiff_path.exists():
        row["is_corrupted"] = True

    return row

if __name__ == '__main__':
    print("Initialisation...")
    
    product_folders = [d for d in L1_DIR.iterdir() if d.is_dir() and "PHISAT" in d.name]
    num_cores = max(1, (os.cpu_count() or 2) - 1)
    
    print(f"Found {len(product_folders)} product folders to process.")
    print(f"Starting processing on {num_cores} cores...")
    
    results = []
    
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = {executor.submit(process_product, folder): folder for folder in product_folders}
        
        for future in tqdm(as_completed(futures), total=len(product_folders), desc="Scraping & QC"):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Error processing folder {futures[future]}: {e}")

    print("\nCreating CSV...")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(results)
    
    df.set_index("product_id", inplace=True)
    df.sort_index(inplace=True)
    
    df.to_csv(OUTPUT_CSV)
    
    total = len(df)
    corrupted = df['is_corrupted'].sum()
    print(f"\nDone! CSV saved to: {OUTPUT_CSV}")
    print(f"Total products: {total} | Corrupted: {corrupted} ({(corrupted/total)*100:.2f}%)")