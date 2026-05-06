import os
import math
import pandas as pd
import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds
from rasterio.warp import reproject, transform_bounds
from rasterio.enums import Resampling
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing as mp

MASTER_LOG_PATH = "/shared/projects/phisat2/data/index/s2b_croped.csv"
MASKS_CSV_PATH = "/shared/projects/phisat2/data/index/s2b_cloud_masks.csv"

OUTPUT_DIR = "/shared/projects/phisat2/data/interim/s2b_croped"
REPORT_PATH = "/shared/projects/phisat2/data/index/cropped_masks_report.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def crop_mask_to_match_image(row_dict):
    product_id = row_dict['product_id']
    crop_path = row_dict['crop_path']
    huge_mask_path = row_dict['mask_path']
    
    out_mask_path = os.path.join(OUTPUT_DIR, f"{product_id}_s2b_cloud_mask_cropped.tif")
    
    if os.path.exists(out_mask_path):
        return {"product_id": product_id, "mask_crop_status": "ALREADY_EXISTS", "mask_croped_path": out_mask_path}

    try:
        with rasterio.open(crop_path) as src_template:
            dst_transform = src_template.transform
            dst_crs = src_template.crs
            dst_shape = (src_template.height, src_template.width)
            dst_bounds = src_template.bounds
            
            out_meta = src_template.profile.copy()
            out_meta.update({
                'count': 1,
                'dtype': 'uint8',
                'nodata': 0,
                'compress': 'lzw'
            })

        with rasterio.Env(GDAL_NUM_THREADS='1'):
            with rasterio.open(huge_mask_path) as src_mask:
                src_crs = src_mask.crs
                if dst_crs != src_crs:
                    safe_bounds = transform_bounds(dst_crs, src_crs, *dst_bounds)
                else:
                    safe_bounds = dst_bounds

                window = from_bounds(*safe_bounds, transform=src_mask.transform)
                
                padded_window = Window(
                    col_off=math.floor(window.col_off) - 2,
                    row_off=math.floor(window.row_off) - 2,
                    width=math.ceil(window.width) + 4,
                    height=math.ceil(window.height) + 4
                )
                
                full_window = Window(0, 0, src_mask.width, src_mask.height)
                safe_window = padded_window.intersection(full_window)
                
                src_data = src_mask.read(1, window=safe_window)
                src_transform = rasterio.windows.transform(safe_window, src_mask.transform)

        dst_data = np.empty((1, dst_shape[0], dst_shape[1]), dtype=np.uint8)
        
        reproject(
            source=src_data,
            destination=dst_data,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest
        )

        with rasterio.open(out_mask_path, 'w', **out_meta) as dst:
            dst.write(dst_data[0], 1)
            dst.descriptions = tuple(["SCL_CLOUD_MASK"])

        return {"product_id": product_id, "mask_crop_status": "SUCCESS", "mask_croped_path": out_mask_path}

    except Exception as e:
        print(f"[ERROR] Product {product_id} failed: {str(e)}", flush=True)
        return {"product_id": product_id, "mask_crop_status": f"ERROR: {str(e)}", "mask_croped_path": None}

def main():
    print("Loading data...", flush=True)
    
    if not os.path.exists(MASTER_LOG_PATH):
        raise FileNotFoundError(f"Cannot find {MASTER_LOG_PATH}")
        
    df_log = pd.read_csv(MASTER_LOG_PATH)
    df_success = df_log[df_log['status'] == 'SUCCESS'].copy()
    print(f"Found {len(df_success)} successfully cropped S2B images.")

    if not os.path.exists(MASKS_CSV_PATH):
        raise FileNotFoundError(f"Cannot find {MASKS_CSV_PATH}")
        
    df_masks = pd.read_csv(MASKS_CSV_PATH)
    
    df_todo = pd.merge(df_success[['product_id', 'crop_path']], 
                       df_masks[['product_id', 'mask_path']], 
                       on='product_id', 
                       how='inner')
    
    if os.path.exists(REPORT_PATH):
        df_done = pd.read_csv(REPORT_PATH)
        done_ids = set(df_done['product_id'].astype(str))
        df_todo = df_todo[~df_todo['product_id'].astype(str).isin(done_ids)]
        print(f"Resuming... {len(df_todo)} masks left to process.")
    else:
        pd.DataFrame(columns=["product_id", "mask_crop_status", "mask_croped_path"]).to_csv(REPORT_PATH, index=False)
        print(f"Ready to process {len(df_todo)} matching cloud masks.")

    if len(df_todo) == 0:
        print("Nothing to do!")
        return

    tasks = df_todo.to_dict('records')

    NUM_WORKERS = 1
    print(f"Starting ProcessPool with {NUM_WORKERS} workers...", flush=True)
    
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(crop_mask_to_match_image, task) for task in tasks]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Cropping Masks"):
            res = future.result()
            pd.DataFrame([res]).to_csv(REPORT_PATH, mode='a', header=False, index=False)

    print(f"\nDone! Report saved to {REPORT_PATH}")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()