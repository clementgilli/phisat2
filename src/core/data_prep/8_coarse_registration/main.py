import os
import time
import random
import torch
import numpy as np
import pandas as pd
import cv2
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
import concurrent.futures
import multiprocessing as mp
import math

# ==========================================
# CONFIGURATION & HYPERPARAMETERS
# ==========================================
CIBLE_RES = 10.0
P2_RES = 4.75
S2B_RES = 10.0
SCALE_P2_EXACT = CIBLE_RES / P2_RES
SCALE_S2_EXACT = CIBLE_RES / S2B_RES

BAND_P2 = 4
BAND_S2B = 3

NORM_CLASSES_P2 = [0]
NORM_CLASSES_S2B = [1, 2, 4, 5, 6, 7, 11]

FILTER_CLASSES_P2 = [0, 2, 3]
FILTER_CLASSES_S2B = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]

MIN_KPTS = 4       
RANSAC_THRESH = 15.0   
MIN_INLIERS = 3        
MARGIN_PCT = 0.10

OUTPUT_DIR = "/shared/projects/phisat2/data/interim/s2b_croped"
LOG_DIR = "/shared/projects/phisat2/data/index/logs"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ==========================================
# UTILITY FUNCTIONS
# ==========================================
def load_and_normalize_simple(img_path, mask_path, band_idx, scale, norm_classes, device, flip_horizontal=False):
    with rasterio.Env(GDAL_NUM_THREADS='6'):
        with rasterio.open(img_path) as src:
            out_h, out_w = int(src.height / scale), int(src.width / scale)
            img = src.read(band_idx, out_shape=(out_h, out_w), resampling=Resampling.nearest).astype(np.float32)
            
        with rasterio.open(mask_path) as src:
            mask = src.read(1, out_shape=(out_h, out_w), resampling=Resampling.nearest)
            
    if flip_horizontal:
        img = cv2.flip(img, 1)
        mask = cv2.flip(mask, 1)
    
    is_valid_ground = (img > 0) & np.isin(mask, norm_classes)
    ground_pixels = img[is_valid_ground]
    
    if len(ground_pixels) > 1000:
        vmin = np.percentile(ground_pixels, 2)
        vmax = np.percentile(ground_pixels, 98)
    else:
        vmin, vmax = 0.0, 10000.0
        
    img_norm = np.clip((img - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)
    
    tensor = torch.from_numpy(img_norm).unsqueeze(0).float().to(device)
    mask_tensor = torch.from_numpy(mask).to(device)
    
    return img, img_norm, tensor, mask_tensor

def filter_features_by_mask(feats, mask_tensor, valid_classes, device):
    kpts = feats['keypoints'][0].long() 
    x = torch.clamp(kpts[:, 0], 0, mask_tensor.shape[1] - 1)
    y = torch.clamp(kpts[:, 1], 0, mask_tensor.shape[0] - 1)
    
    mask_values = mask_tensor[y, x]
    is_valid = torch.isin(mask_values, torch.tensor(valid_classes).to(device))
    
    filtered_feats = {k: (v if k == 'image_size' else v[:, is_valid]) for k, v in feats.items()}
    return filtered_feats

# ==========================================
# WORKER FUNCTION (Runs on a single GPU)
# ==========================================
def process_chunk(chunk_df, gpu_id, worker_id):
    """
    Function executed by each independent process.
    It forces execution on a specific GPU and uses 8 CPU threads for GDAL.
    """
    # Force C/C++ libraries to respect the 8 CPUs allocated per worker
    os.environ["GDAL_NUM_THREADS"] = "6"
    os.environ["OMP_NUM_THREADS"] = "6"
    
    # Isolate the GPU
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"[Worker {worker_id}] Started on {device} processing {len(chunk_df)} images.", flush=True)

    # Set seeds per worker to maintain determinism
    seed = 49 + worker_id
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Initialize models locally on the assigned GPU
    from lightglue import LightGlue, SIFT
    from lightglue.utils import rbd
    
    extractor = SIFT(max_num_keypoints=4096).eval().to(device) 
    matcher = LightGlue(features='sift').eval().to(device)

    results_log = []
    log_csv_path = os.path.join(LOG_DIR, f"processing_log_worker_{worker_id}.csv")

    for i, (_, product) in enumerate(chunk_df.iterrows()):
        product_id = product["product_id"]
        start_time = time.time()
        
        try:
            p2_img_path   = product["folder_path"] + "/bands/scene_0_BC_multiband.tiff"
            p2_mask_path  = f"/shared/projects/phisat2/data/interim/phisat2_cloud_masks/{product_id}_phisat2_OCM_v1_7_1.tif" 
            s2b_img_path  = product["merged_tif_path"]
            s2b_mask_path = product['mask_path']
            
            # 1. Load and Normalize
            img_p2_orig, img_p2_norm, tensor_p2, mask_p2_tensor = load_and_normalize_simple(
                p2_img_path, p2_mask_path, BAND_P2, SCALE_P2_EXACT, NORM_CLASSES_P2, device, flip_horizontal=True
            )

            img_s2_orig, img_s2_norm, tensor_s2, mask_s2_tensor = load_and_normalize_simple(
                s2b_img_path, s2b_mask_path, BAND_S2B, SCALE_S2_EXACT, NORM_CLASSES_S2B, device, flip_horizontal=False
            )
            
            # 2. Extract Features
            feats_p2 = extractor.extract(tensor_p2)
            feats_s2 = extractor.extract(tensor_s2)

            # 3. Filter Features
            feats_p2_clean = filter_features_by_mask(feats_p2, mask_p2_tensor, FILTER_CLASSES_P2, device)
            feats_s2_clean = filter_features_by_mask(feats_s2, mask_s2_tensor, FILTER_CLASSES_S2B, device)
            
            # 4. Matching
            matches01 = matcher({"image0": feats_p2_clean, "image1": feats_s2_clean})

            feats0, feats1, matches01 = [rbd(x) for x in [feats_p2_clean, feats_s2_clean, matches01]]
            kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
            m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

            match_count = len(m_kpts0)
            if match_count < MIN_KPTS:
                raise ValueError(f"Only {match_count} matches.")

            # 5. RANSAC
            m_kpts0_np = m_kpts0.cpu().numpy()
            m_kpts1_np = m_kpts1.cpu().numpy()

            M, inliers = cv2.estimateAffinePartial2D(m_kpts0_np, m_kpts1_np, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_THRESH, maxIters=5000)

            if M is None:
                raise ValueError("RANSAC failed.")

            inliers = inliers.flatten().astype(bool)
            inliers_count = inliers.sum()
            inlier_ratio = inliers_count / match_count if match_count > 0 else 0

            if inliers_count < MIN_INLIERS:
                raise ValueError(f"Only {inliers_count} inliers.")

            if match_count < 40 and inlier_ratio < 0.15:
                raise ValueError(f"Low match count & ratio.")
            elif match_count >= 40 and inliers_count < 10:
                raise ValueError(f"RANSAC hallucination.")

            scale_x = np.sqrt(M[0, 0]**2 + M[1, 0]**2)
            scale_y = np.sqrt(M[0, 1]**2 + M[1, 1]**2)
            aspect_ratio = scale_x / scale_y

            if not (0.75 < scale_x < 1.25) or not (0.75 < scale_y < 1.25):
                raise ValueError("Unrealistic scale.")
            if not (0.8 < aspect_ratio < 1.2):
                raise ValueError("Unrealistic aspect ratio.")

            # 6. Geometric Bounding Box Projection
            H = np.vstack([M, [0, 0, 1]])
            h_p2, w_p2 = img_p2_norm.shape
            corners_p2 = np.array([[0, 0], [w_p2, 0], [w_p2, h_p2], [0, h_p2]], dtype=np.float32).reshape(-1, 1, 2)
            warped_corners = cv2.perspectiveTransform(corners_p2, H).squeeze()
            
            x_min, y_min = np.min(warped_corners, axis=0).astype(int)
            x_max, y_max = np.max(warped_corners, axis=0).astype(int)

            w_crop = x_max - x_min
            h_crop = y_max - y_min

            scaled_x_min_safe = max(0, int(x_min - w_crop * MARGIN_PCT))
            scaled_y_min_safe = max(0, int(y_min - h_crop * MARGIN_PCT))
            scaled_x_max_safe = min(img_s2_norm.shape[1], int(x_max + w_crop * MARGIN_PCT))
            scaled_y_max_safe = min(img_s2_norm.shape[0], int(y_max + h_crop * MARGIN_PCT))

            # 7. Map back to Native Original Resolution
            native_x_min = int(scaled_x_min_safe * SCALE_S2_EXACT)
            native_y_min = int(scaled_y_min_safe * SCALE_S2_EXACT)
            native_x_max = int(scaled_x_max_safe * SCALE_S2_EXACT)
            native_y_max = int(scaled_y_max_safe * SCALE_S2_EXACT)
            
            native_width = native_x_max - native_x_min
            native_height = native_y_max - native_y_min

            # 8. Extract & Save the multi-band crop
            out_crop_path = os.path.join(OUTPUT_DIR, f"{product_id}_s2b_cropped.tif")
            
            with rasterio.Env(GDAL_NUM_THREADS='6'):
                with rasterio.open(s2b_img_path) as src:
                    native_x_min = max(0, native_x_min)
                    native_y_min = max(0, native_y_min)
                    native_width = min(native_width, src.width - native_x_min)
                    native_height = min(native_height, src.height - native_y_min)
                    
                    window = Window(native_x_min, native_y_min, native_width, native_height)
                    
                    kwargs = src.meta.copy()
                    kwargs.update({
                        'height': window.height,
                        'width': window.width,
                        'transform': rasterio.windows.transform(window, src.transform)
                    })

                    with rasterio.open(out_crop_path, 'w', **kwargs) as dst:
                        dst.write(src.read(window=window))
            
            elapsed_time = time.time() - start_time
            if i % 10 == 0:
                print(f"[Worker {worker_id}] Processed {i}/{len(chunk_df)} - ID {product_id} SUCCESS ({elapsed_time:.1f}s)", flush=True)
            
            results_log.append({
                "product_id": product_id,
                "status": "SUCCESS",
                "matches": match_count,
                "inliers": inliers_count,
                "crop_path": out_crop_path,
                "error_msg": ""
            })

        except Exception as e:
            if i % 10 == 0:
                print(f"[Worker {worker_id}] Processed {i}/{len(chunk_df)} - ID {product_id} FAILED: {str(e)}", flush=True)
                
            results_log.append({
                "product_id": product_id,
                "status": "FAILED",
                "matches": 0,
                "inliers": 0,
                "crop_path": "",
                "error_msg": str(e)
            })

        # Save worker-specific CSV incrementally
        if i % 20 == 0:
            pd.DataFrame(results_log).to_csv(log_csv_path, index=False)

    # Final save
    pd.DataFrame(results_log).to_csv(log_csv_path, index=False)
    print(f"[Worker {worker_id}] Finished successfully.", flush=True)

# ==========================================
# MAIN
# ==========================================
def main():
    print("Loading dataframes...", flush=True)
    df_meta_phi = pd.read_csv("/shared/projects/phisat2/data/index/phisat2_metadata.csv")
    df_meta_s2b = pd.read_csv("/shared/projects/phisat2/data/index/phisat2_to_s2b.csv")
    df_cloud_s2b = pd.read_csv("/shared/projects/phisat2/data/index/s2b_cloud_masks.csv")
    df_merged = pd.read_csv("/shared/projects/phisat2/data/index/s2b_merged.csv")
    df_cloud_phi = pd.read_csv("/shared/projects/phisat2/data/index/phisat2_cloud_stats_detailed.csv")

    df_full = pd.merge(df_meta_phi, df_meta_s2b, on="product_id", how="inner")
    df_full = pd.merge(df_full, df_merged, on="product_id", how="inner")
    df_full = pd.merge(df_full, df_cloud_s2b, on="product_id", how="inner")
    df_full = pd.merge(df_full, df_cloud_phi, on="product_id", how="inner")
    df_valid = df_full[df_full['merged_tif_path'].notna()].reset_index(drop=True)

    print(f"Found {len(df_valid)} valid products to process.", flush=True)

    NUM_GPUS = 4
    print(f"Splitting workload across {NUM_GPUS} GPUs...", flush=True)
    
    # Split dataframe into 4 chunks
    chunk_size = math.ceil(len(df_valid) / NUM_GPUS)
    chunks = [df_valid.iloc[i * chunk_size : (i + 1) * chunk_size] for i in range(NUM_GPUS)]

    # Use ProcessPoolExecutor to spawn 4 independent processes
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_GPUS) as executor:
        futures = []
        for i in range(NUM_GPUS):
            # i serves as both the GPU ID (0, 1, 2, 3) and the Worker ID
            futures.append(executor.submit(process_chunk, chunks[i], i, i))
            
        # Wait for all processes to complete
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"A worker process crashed: {str(e)}", flush=True)

    print("\nAll workers finished. Merging log files...", flush=True)
    
    # Merge the 4 partial CSV logs into one final master log
    master_log = []
    for i in range(NUM_GPUS):
        worker_log_path = os.path.join(LOG_DIR, f"processing_log_worker_{i}.csv")
        if os.path.exists(worker_log_path):
            master_log.append(pd.read_csv(worker_log_path))
            
    if master_log:
        pd.concat(master_log).to_csv("/shared/projects/phisat2/data/index/master_processing_log.csv", index=False)
        print(f"Master log saved.", flush=True)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()