import os
import argparse
import h5py
import numpy as np
from tqdm import tqdm

# --- CONFIGURATION ---
H5_PATH = "/shared/projects/phisat2/data/processed/triplets_v1/phisat2_s2b_dataset_v1.h5"

BAD_PRODUCT_IDS = [
    1296, 1342, 1385, 1397, 1420, 1460, 1497, 1647, 1854, 2223, 2246, 2259, 
    2373, 2631, 2640, 2743, 2834, 2853, 3374, 3619, 4071, 4693, 4813, 4942, 
    2352, 2882, 3322, 3914, 4702, 1333, 1466, 1615, 2460, 2729, 2763
]

def process_modality(ds_name):
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    
    # Plafond selon le domaine
    clip_max = np.sqrt(1500.0) if ds_name == "real/images" else 100.0

    print(f"{ds_name}")

    with h5py.File(H5_PATH, 'r') as f:
        all_ids = f["metadata/product_id"][:]
        mask = ~np.isin(all_ids, BAD_PRODUCT_IDS)
        valid_indices = np.where(mask)[0]
        num_valid = len(valid_indices)
        
        ds = f[ds_name]
        _, num_channels, h, w = ds.shape
        
        pixel_sum = np.zeros(num_channels, dtype=np.float64)
        pixel_sum_sq = np.zeros(num_channels, dtype=np.float64)
        raw_min = np.full(num_channels, np.inf)
        raw_max = np.full(num_channels, -np.inf)
        
        total_pixels_count = num_valid * h * w
        chunk_size = 1000 

        for i in tqdm(range(0, num_valid, chunk_size), desc=ds_name):
            end = min(i + chunk_size, num_valid)
            data_batch = ds[valid_indices[i:end]].astype(np.float64)
            
            raw_min = np.minimum(raw_min, np.min(data_batch, axis=(0, 2, 3)))
            raw_max = np.maximum(raw_max, np.max(data_batch, axis=(0, 2, 3)))
            
            data_batch = np.maximum(data_batch, 0)
            data_batch = np.sqrt(data_batch)
            data_batch = np.clip(data_batch, 0, clip_max)
            
            pixel_sum += np.sum(data_batch, axis=(0, 2, 3))
            pixel_sum_sq += np.sum(data_batch**2, axis=(0, 2, 3))

        mean = pixel_sum / total_pixels_count
        variance = (pixel_sum_sq / total_pixels_count) - (mean**2)
        std = np.sqrt(np.maximum(variance, 0))

        print("\n" + "="*50)
        print(f"Results {ds_name}")
        print("="*50)
        print(f"Min (raw) : {list(np.round(raw_min, 2))}")
        print(f"Max (raw) : {list(np.round(raw_max, 2))}")
        print(f"Mean (sqrt)    : {list(np.round(mean, 4))}")
        print(f"Std (sqrt)     : {list(np.round(std, 4))}")
        print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["real", "sim", "s2b"])
    args = parser.ArgumentParser().parse_args() if False else parser.parse_args()
    
    process_modality(f"{args.mode}/images")