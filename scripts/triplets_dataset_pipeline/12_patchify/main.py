import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import h5py
import pandas as pd
import numpy as np
import rasterio
import cv2
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
CSV_PATH = "/shared/projects/phisat2/data/processed/dataset_master_images_metadata.csv"
OUTPUT_H5 = "/shared/projects/phisat2/data/processed/phisat2_s2b_dataset_v1.h5"
KOPPEN_RASTER_PATH = "/shared/projects/phisat2/data/external/koppen_geiger_0p00833333.tif"

PATCH_SIZE = 256
CLOUD_THRESHOLD = 0.05
NODATA_THRESHOLD = 0.01
WATER_THRESHOLD = 0.95

TEST_MODE = False
TEST_INDICES = [200, 201, 202] 

def get_patch_coords(px_x, px_y, img_w, img_h, corners):
    nx = px_x / img_w
    ny = px_y / img_h
    lon_top = corners['ul_lon'] + nx * (corners['ur_lon'] - corners['ul_lon'])
    lat_top = corners['ul_lat'] + nx * (corners['ur_lat'] - corners['ul_lat'])
    lon_bot = corners['ll_lon'] + nx * (corners['lr_lon'] - corners['ll_lon'])
    lat_bot = corners['ll_lat'] + nx * (corners['lr_lat'] - corners['ll_lat'])
    lon = lon_top + ny * (lon_bot - lon_top)
    lat = lat_top + ny * (lat_bot - lat_top)
    return round(lat, 6), round(lon, 6)

def patchify_and_store(csv_row, h5_file, koppen_raster):
    product_id = csv_row['product_id']
    
    try:
        # --- 1. REAL PHISAT-2 ---
        with rasterio.open(csv_row['phisat2_path']) as src:
            img_real = np.array([cv2.flip(band, 1) for band in src.read()])
            _, h, w = img_real.shape
        
        with rasterio.open(csv_row['phisat2_mask_path']) as src:
            mask_real = cv2.resize(src.read(1), (w, h), interpolation=cv2.INTER_NEAREST)
            mask_real = cv2.flip(mask_real, 1)
            
        # --- 2. SIMULATED PHISAT-2 ---
        # .tif order: [B1, B2, B3, PAN, B7, B4, B5, B6]
        # Target order:   [PAN, B1, B2, B3, B4, B5, B6, B7]
        # Merged index:   [3, 0, 1, 2, 5, 6, 7, 4]
        img_sim = rasterio.open(csv_row['aligned_phisat2sim_path']).read()
        img_sim = img_sim[[3, 0, 1, 2, 5, 6, 7, 4], :, :]
        
        # --- 3. SENTINEL-2B ---
        # .tif order: [B1, B2, B3, B7, B4, B5, B6]
        # Target order:   [B1, B2, B3, B4, B5, B6, B7]
        # Merged index:   [0, 1, 2, 4, 5, 6, 3]
        img_s2b = rasterio.open(csv_row['aligned_s2b_path']).read()
        img_s2b = img_s2b[[0, 1, 2, 4, 5, 6, 3], :, :]
        
        mask_s2b = rasterio.open(csv_row['aligned_mask_path']).read(1)
        
    except Exception as e:
        print(f"Error loading {product_id}: {e}")
        return 0

    batch_real, batch_mask_real = [], []
    batch_sim = []
    batch_s2b, batch_mask_s2b = [], []
    batch_meta = {
        "product_id": [], "date_phi": [], "date_s2b": [],
        "center_lat": [], "center_lon": [],
        "ul_lat": [], "ul_lon": [], "ur_lat": [], "ur_lon": [],
        "lr_lat": [], "lr_lon": [], "ll_lat": [], "ll_lon": []
    }
    
    for y in range(0, h - PATCH_SIZE + 1, PATCH_SIZE):
        for x in range(0, w - PATCH_SIZE + 1, PATCH_SIZE):
            
            pm_real = mask_real[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            pm_s2b = mask_s2b[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            
            if np.mean(pm_real == 1) > CLOUD_THRESHOLD: continue
            if np.mean((pm_s2b == 8) | (pm_s2b == 9) | (pm_s2b == 1)) > CLOUD_THRESHOLD: continue
            if np.mean(pm_s2b == 6) > WATER_THRESHOLD: continue
            
            p_s2b = img_s2b[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            if np.mean(np.isnan(p_s2b) | (p_s2b == 0)) > NODATA_THRESHOLD: continue
            
            p_real = np.round(img_real[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]).astype(np.int16)
            p_sim = np.round(img_sim[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]).astype(np.int16)
            p_s2b = np.round(p_s2b).astype(np.int16)
            
            lat_c, lon_c = get_patch_coords(x + PATCH_SIZE//2, y + PATCH_SIZE//2, w, h, csv_row)
            lat_ul, lon_ul = get_patch_coords(x, y, w, h, csv_row)
            lat_ur, lon_ur = get_patch_coords(x + PATCH_SIZE, y, w, h, csv_row)
            lat_lr, lon_lr = get_patch_coords(x + PATCH_SIZE, y + PATCH_SIZE, w, h, csv_row)
            lat_ll, lon_ll = get_patch_coords(x, y + PATCH_SIZE, w, h, csv_row)
            
            batch_real.append(p_real)
            batch_mask_real.append(pm_real)
            batch_sim.append(p_sim)
            batch_s2b.append(p_s2b)
            batch_mask_s2b.append(pm_s2b)
            
            batch_meta["product_id"].append(product_id)
            batch_meta["date_phi"].append(str(csv_row['phisat2_date']).encode('utf-8'))
            batch_meta["date_s2b"].append(str(csv_row['s2b_date']).encode('utf-8'))
            batch_meta["center_lat"].append(lat_c)
            batch_meta["center_lon"].append(lon_c)
            batch_meta["ul_lat"].append(lat_ul)
            batch_meta["ul_lon"].append(lon_ul)
            batch_meta["ur_lat"].append(lat_ur)
            batch_meta["ur_lon"].append(lon_ur)
            batch_meta["lr_lat"].append(lat_lr)
            batch_meta["lr_lon"].append(lon_lr)
            batch_meta["ll_lat"].append(lat_ll)
            batch_meta["ll_lon"].append(lon_ll)

    num_patches = len(batch_real)
    if num_patches == 0:
        return 0

    # Koppen-Geiger Bulk Sampling
    coords = list(zip(batch_meta["center_lon"], batch_meta["center_lat"]))
    sampled_values = list(koppen_raster.sample(coords))
    batch_meta["koppen_zone"] = [int(val[0]) for val in sampled_values]

    idx = h5_file["real/images"].shape[0]
    
    for ds_path in ["real/images", "real/masks", "sim/images", "s2b/images", "s2b/masks"]:
        h5_file[ds_path].resize(idx + num_patches, axis=0)
        
    for m_key in batch_meta.keys():
        h5_file[f"metadata/{m_key}"].resize(idx + num_patches, axis=0)

    h5_file["real/images"][idx : idx + num_patches] = np.array(batch_real)
    h5_file["real/masks"][idx : idx + num_patches]  = np.array(batch_mask_real)
    h5_file["sim/images"][idx : idx + num_patches]  = np.array(batch_sim)
    h5_file["s2b/images"][idx : idx + num_patches]  = np.array(batch_s2b)
    h5_file["s2b/masks"][idx : idx + num_patches]   = np.array(batch_mask_s2b)
    
    for m_key, m_list in batch_meta.items():
        h5_file[f"metadata/{m_key}"][idx : idx + num_patches] = np.array(m_list)

    return num_patches

# --- MAIN ---
df_master = pd.read_csv(CSV_PATH)

if TEST_MODE:
    df_master = df_master.iloc[TEST_INDICES]
    OUTPUT_H5 = OUTPUT_H5.replace(".h5", "_TEST.h5")
    print(f"TEST MODE: Processing {len(df_master)} images.")
    print(f"Output: {OUTPUT_H5}")
        
file_exists = os.path.exists(OUTPUT_H5)
mode = 'a' if file_exists else 'w'

with rasterio.open(KOPPEN_RASTER_PATH) as koppen_raster:
    with h5py.File(OUTPUT_H5, mode) as f:
        if mode == 'w':
            f.create_dataset("real/images", (0, 8, PATCH_SIZE, PATCH_SIZE), maxshape=(None, 8, PATCH_SIZE, PATCH_SIZE), dtype='int16', chunks=(1, 8, PATCH_SIZE, PATCH_SIZE), compression="lzf")
            f.create_dataset("real/masks",  (0, PATCH_SIZE, PATCH_SIZE),    maxshape=(None, PATCH_SIZE, PATCH_SIZE),    dtype='uint8', chunks=(1, PATCH_SIZE, PATCH_SIZE), compression="lzf")
            f.create_dataset("sim/images",  (0, 8, PATCH_SIZE, PATCH_SIZE), maxshape=(None, 8, PATCH_SIZE, PATCH_SIZE), dtype='int16', chunks=(1, 8, PATCH_SIZE, PATCH_SIZE), compression="lzf")
            f.create_dataset("s2b/images",  (0, 7, PATCH_SIZE, PATCH_SIZE), maxshape=(None, 7, PATCH_SIZE, PATCH_SIZE), dtype='int16', chunks=(1, 7, PATCH_SIZE, PATCH_SIZE), compression="lzf")
            f.create_dataset("s2b/masks",   (0, PATCH_SIZE, PATCH_SIZE),    maxshape=(None, PATCH_SIZE, PATCH_SIZE),    dtype='uint8', chunks=(1, PATCH_SIZE, PATCH_SIZE), compression="lzf")
            
            f.create_dataset("metadata/product_id", (0,), maxshape=(None,), dtype='int32')
            f.create_dataset("metadata/date_phi",   (0,), maxshape=(None,), dtype=h5py.special_dtype(vlen=bytes))
            f.create_dataset("metadata/date_s2b",   (0,), maxshape=(None,), dtype=h5py.special_dtype(vlen=bytes))
            f.create_dataset("metadata/koppen_zone", (0,), maxshape=(None,), dtype='int8')
            
            for coord in ["center_lat", "center_lon", "ul_lat", "ul_lon", "ur_lat", "ur_lon", "lr_lat", "lr_lon", "ll_lat", "ll_lon"]:
                f.create_dataset(f"metadata/{coord}", (0,), maxshape=(None,), dtype='float32')
            processed_ids = []
        else:
            processed_ids = np.unique(f["metadata/product_id"][:])
            print(f"Resuming: {len(processed_ids)} products found in H5.")

        df_todo = df_master[~df_master['product_id'].isin(processed_ids)]
        
        total_patches = 0
        if len(df_todo) == 0:
            print("Done: All processed.")
        else:
            for _, row in tqdm(df_todo.iterrows(), total=len(df_todo), desc="Patchification"):
                num = patchify_and_store(row, f, koppen_raster)
                f.flush()
                total_patches += num

print(f"\nFinished: {total_patches} patches added.")
if os.path.exists(OUTPUT_H5):
    print(f"Size: {os.path.getsize(OUTPUT_H5) / (1024*1024):.2f} MB")