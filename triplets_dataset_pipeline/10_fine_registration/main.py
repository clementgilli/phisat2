import os
import time
import math
import random
import torch
import numpy as np
import pandas as pd
import cv2
import rasterio
from rasterio.enums import Resampling

##############################################################################
#                             /!\ WARNING /!\                                #
#                                                                            #
# SIMULATED PHISAT-2 .tif order saving : [B1, B2, B3, PAN, B7, B4, B5, B6].  #
# SENTINEL-2B .tif order saving : [B1, B2, B3, B7, B4, B5, B6]               #
# => Be careful when loading and reordering bands for processing !           #
##############################################################################

# ==========================================
# CONFIGURATION & HYPERPARAMETERS
# ==========================================
OUTPUT_DIR = "/shared/projects/phisat2/data/interim/s2b_simulated_aligned"
LOG_DIR = "/shared/projects/phisat2/data/index/logs"
FINAL_LOG_PATH = "/shared/projects/phisat2/data/index/s2b_simulated_aligned_log.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Band Indices
BAND_PAN_REAL = 0 + 1  
BAND_PAN_SIM  = 3 + 1

# LightGlue / RANSAC Config
MAX_KPTS = 2048
RANSAC_THRESH = 3.0
MIN_INLIERS = 10

# Classes for normalization & filtering
NORM_CLASSES_P2 = [0]
NORM_CLASSES_S2B = [1, 2, 4, 5, 6, 7, 11]

FILTER_CLASSES_P2 = [0, 2, 3]
FILTER_CLASSES_S2B = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]

# ==========================================
# UTILITY FUNCTIONS
# ==========================================
def load_and_normalize_simple(img_path, mask_path, band_idx, norm_classes, device, flip_horizontal=False):
    with rasterio.Env(GDAL_NUM_THREADS='1'):
        with rasterio.open(img_path) as src:
            out_h, out_w = src.height, src.width
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
    
    tensor = torch.from_numpy(img_norm).unsqueeze(0).to(torch.float32).to(device)
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
# MAIN
# ==========================================
def main():
    os.environ["GDAL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Started single-process alignment on {device}...", flush=True)

    seed = 49
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    df_meta_phi = pd.read_csv("/shared/projects/phisat2/data/index/phisat2_metadata.csv")
    df_croped_s2b = pd.read_csv("/shared/projects/phisat2/data/index/s2b_croped.csv")
    df_croped_masks = pd.read_csv("/shared/projects/phisat2/data/index/cropped_masks_report.csv")
    SIM_LOG_PATH = "/shared/projects/phisat2/data/index/simulation_status_report.csv"
    df_simulated = pd.read_csv(SIM_LOG_PATH)
    
    df_valid = pd.merge(df_croped_s2b, df_croped_masks, on="product_id", how="inner")
    df_valid = pd.merge(df_valid, df_meta_phi[['product_id', 'folder_path']], on="product_id", how="inner")
    df_valid = pd.merge(df_valid, df_simulated[['product_id', 'simulation_status', 'simulation_path']], on="product_id", how="inner")
    
    df_valid = df_valid[
        (df_valid['status'] == 'SUCCESS') & 
        (df_valid['mask_crop_status'].isin(['SUCCESS', 'ALREADY_EXISTS'])) &
        (df_valid['simulation_status'] == 'SUCCESS')
    ].reset_index(drop=True)

    if os.path.exists(FINAL_LOG_PATH):
        df_existing_log = pd.read_csv(FINAL_LOG_PATH)
        processed_ids = df_existing_log['product_id'].unique()
        df_valid = df_valid[~df_valid['product_id'].isin(processed_ids)].reset_index(drop=True)
        results_log = df_existing_log.to_dict('records')
    else:
        results_log = []

    if len(df_valid) == 0:
        print("Nothing to process. Exiting.")
        return

    print(f"Total products to process: {len(df_valid)}", flush=True)

    from lightglue import LightGlue, SIFT
    from lightglue.utils import rbd
    
    extractor = SIFT(max_num_keypoints=MAX_KPTS).eval().to(device) 
    matcher = LightGlue(features='sift').eval().to(device)

    for i, (_, product) in enumerate(df_valid.iterrows()):
        product_id = product["product_id"]
        
        out_img_path = os.path.join(OUTPUT_DIR, f"{product_id}_simulated_aligned.tif")
        out_mask_path = os.path.join(OUTPUT_DIR, f"{product_id}_simulated_mask_aligned.tif")
            
        try:
            REAL_PATH = product["folder_path"] + "/bands/scene_0_BC_multiband.tiff"
            REAL_MASK_PATH = f"/shared/projects/phisat2/data/interim/phisat2_cloud_masks/{product_id}_phisat2_OCM_v1_7_1.tif"
            SIM_PATH = f"/shared/projects/phisat2/data/interim/s2b_simulated/simulated_L1C_{product_id}_s2b_cropped.tif"
            MASK_SIM_PATH = product["mask_croped_path"]
            
            img_real_raw, _, tensor_real, mask_real_tensor = load_and_normalize_simple(
                REAL_PATH, REAL_MASK_PATH, BAND_PAN_REAL, NORM_CLASSES_P2, device, flip_horizontal=True
            )
            img_sim_raw, _, tensor_sim, mask_sim_tensor = load_and_normalize_simple(
                SIM_PATH, MASK_SIM_PATH, BAND_PAN_SIM, NORM_CLASSES_S2B, device, flip_horizontal=False
            )

            feats_real = extractor.extract(tensor_real)
            feats_sim = extractor.extract(tensor_sim)

            feats_real_clean = filter_features_by_mask(feats_real, mask_real_tensor, FILTER_CLASSES_P2, device)
            feats_sim_clean = filter_features_by_mask(feats_sim, mask_sim_tensor, FILTER_CLASSES_S2B, device)

            matches01 = matcher({"image0": feats_real_clean, "image1": feats_sim_clean})
            feats0, feats1, matches01 = [rbd(x) for x in [feats_real_clean, feats_sim_clean, matches01]]
            
            kpts_real = feats0["keypoints"][matches01["matches"][..., 0]].cpu().numpy()
            kpts_sim = feats1["keypoints"][matches01["matches"][..., 1]].cpu().numpy()

            if len(kpts_real) < MIN_INLIERS:
                raise ValueError(f"Not enough matches ({len(kpts_real)}) to compute Homography.")

            H, inliers = cv2.findHomography(kpts_real, kpts_sim, cv2.RANSAC, RANSAC_THRESH)
            
            if H is None:
                raise ValueError("RANSAC failed to find Homography matrix.")
                
            inliers_count = inliers.sum()
            if inliers_count < MIN_INLIERS:
                raise ValueError(f"Not enough inliers ({inliers_count}).")

            H_inv = np.linalg.inv(H)
            
            h_ref, w_ref = img_real_raw.shape
            
            corners_real = np.array([
                [0, 0],           
                [w_ref, 0],       
                [w_ref, h_ref],   
                [0, h_ref]        
            ], dtype=np.float32).reshape(-1, 1, 2)
            
            corners_sim = cv2.perspectiveTransform(corners_real, H).reshape(-1, 2)
            
            from rasterio.warp import transform as warp_transform
            
            with rasterio.open(SIM_PATH) as src_sim_gps:
                sim_transform = src_sim_gps.transform
                sim_crs = src_sim_gps.crs
                
            lons_utm, lats_utm = [], []
            for pt in corners_sim:
                lon, lat = sim_transform * (pt[0], pt[1])
                lons_utm.append(lon)
                lats_utm.append(lat)
                
            lons_wgs84, lats_wgs84 = warp_transform(sim_crs, 'EPSG:4326', lons_utm, lats_utm)
            
            ul_lon, ur_lon, lr_lon, ll_lon = [round(x, 6) for x in lons_wgs84]
            ul_lat, ur_lat, lr_lat, ll_lat = [round(x, 6) for x in lats_wgs84]
            center_lon = round(sum(lons_wgs84)/4, 6)
            center_lat = round(sum(lats_wgs84)/4, 6)
            
            with rasterio.Env(GDAL_NUM_THREADS='1'):
                with rasterio.open(SIM_PATH) as src_sim:
                    sim_meta = src_sim.profile.copy()
                    sim_data = src_sim.read()
                    
                with rasterio.open(REAL_PATH) as src_real:
                    real_transform = src_real.transform
                    real_crs = src_real.crs

                sim_aligned_data = np.zeros((sim_meta['count'], h_ref, w_ref), dtype=sim_data.dtype)
                for b in range(sim_meta['count']):
                    sim_aligned_data[b] = cv2.warpPerspective(
                        sim_data[b], H_inv, (w_ref, h_ref), 
                        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0
                    )
                
                sim_meta.update({
                    'height': h_ref,
                    'width': w_ref,
                    'transform': real_transform,
                    'crs': real_crs
                })
                
                with rasterio.open(out_img_path, 'w', **sim_meta) as dst:
                    dst.write(sim_aligned_data)

                with rasterio.open(MASK_SIM_PATH) as src_mask:
                    mask_meta = src_mask.profile.copy()
                    
                mask_sim_upscaled = mask_sim_tensor.detach().cpu().numpy()
                    
                mask_aligned_data = cv2.warpPerspective(
                    mask_sim_upscaled, H_inv, (w_ref, h_ref), 
                    flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
                )
                
                mask_meta.update({
                    'height': h_ref,
                    'width': w_ref,
                    'transform': real_transform,
                    'crs': real_crs
                })
                
                with rasterio.open(out_mask_path, 'w', **mask_meta) as dst:
                    dst.write(mask_aligned_data, 1)

                S2B_PATH = product["crop_path"]
                out_s2b_path = os.path.join(OUTPUT_DIR, f"{product_id}_s2b_aligned.tif")
                h_sim, w_sim = img_sim_raw.shape
                with rasterio.open(S2B_PATH) as src_s2b:
                    s2b_meta = src_s2b.profile.copy()
                    
                    s2b_upscaled = src_s2b.read(
                        out_shape=(src_s2b.count, h_sim, w_sim),
                        resampling=Resampling.cubic
                    )
                
                s2b_aligned_data = np.zeros((s2b_meta['count'], h_ref, w_ref), dtype=s2b_upscaled.dtype)
                for b in range(s2b_meta['count']):
                    s2b_aligned_data[b] = cv2.warpPerspective(
                        s2b_upscaled[b], H_inv, (w_ref, h_ref), 
                        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0
                    )
                
                s2b_meta.update({
                    'height': h_ref,
                    'width': w_ref,
                    'transform': real_transform,
                    'crs': real_crs
                })
                
                with rasterio.open(out_s2b_path, 'w', **s2b_meta) as dst:
                    dst.write(s2b_aligned_data)

            results_log.append({
                "product_id": product_id, "status": "SUCCESS", 
                "inliers": inliers_count, 
                "center_lat": center_lat, "center_lon": center_lon,
                "ul_lat": ul_lat, "ul_lon": ul_lon,
                "ur_lat": ur_lat, "ur_lon": ur_lon,
                "lr_lat": lr_lat, "lr_lon": lr_lon,
                "ll_lat": ll_lat, "ll_lon": ll_lon,
                "aligned_phisat2sim_path": out_img_path, 
                "aligned_mask_path": out_mask_path, 
                "aligned_s2b_path": out_s2b_path, 
                "error_msg": ""
            })

        except Exception as e:
            results_log.append({
                "product_id": product_id, "status": "FAILED", 
                "inliers": 0, 
                "center_lat": None, "center_lon": None,
                "ul_lat": None, "ul_lon": None,
                "ur_lat": None, "ur_lon": None,
                "lr_lat": None, "lr_lon": None,
                "ll_lat": None, "ll_lon": None,
                "aligned_phisat2sim_path": "", "aligned_mask_path": "", "aligned_s2b_path": "", "error_msg": str(e)
            })

        print(f"Processed {product_id}: {results_log[-1]['status']}", flush=True)
        pd.DataFrame(results_log).to_csv(FINAL_LOG_PATH, index=False)

    print("Finished single-process alignment.", flush=True)

if __name__ == "__main__":
    main()