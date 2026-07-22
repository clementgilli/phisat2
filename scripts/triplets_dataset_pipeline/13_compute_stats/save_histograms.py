import os
import h5py
import numpy as np
import torch
import json
from tqdm import tqdm

H5_PATH = "/shared/projects/phisat2/data/processed/triplets_v1/phisat2_s2b_dataset_v1.h5"

BAD_PRODUCT_IDS = [
    1296, 1342, 1385, 1397, 1420, 1460, 1497, 1647, 1854, 2223, 2246, 2259, 
    2373, 2631, 2640, 2743, 2834, 2853, 3374, 3619, 4071, 4693, 4813, 4942, 
    2352, 2882, 3322, 3914, 4702, 1333, 1466, 1615, 2460, 2729, 2763
]

def run_mega_diagnostic(ds_name, offset, is_sim, save_dir="/shared/projects/phisat2/data/processed/triplets_v1/diagnostics"):
    os.makedirs(save_dir, exist_ok=True)
    dataset_id = ds_name.replace("/", "_")
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with h5py.File(H5_PATH, 'r') as f:
        all_ids = f["metadata/product_id"][:]
        mask = ~np.isin(all_ids, BAD_PRODUCT_IDS)
        valid_indices = np.where(mask)[0].tolist()
        num_valid = len(valid_indices)
        
        ds = f[ds_name]
        _, num_channels, h, w = ds.shape
        chunk_size = 256 
        total_pixels_count = num_valid * h * w
        
        num_bins = 30000
        max_val = 120.0
        bin_edges = np.linspace(0, max_val, num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        histograms_pre = torch.zeros((num_channels, num_bins), dtype=torch.int64, device=device)

        for i in tqdm(range(0, num_valid, chunk_size), desc="Pass 1"):
            end = min(i + chunk_size, num_valid)
            data_batch = torch.from_numpy(ds[valid_indices[i:end]]).to(device, dtype=torch.float32)
            
            if offset != 0.0:
                data_batch = data_batch - offset
            data_batch = torch.clamp(data_batch, min=0)
            data_batch = torch.sqrt(data_batch)
            
            for c in range(num_channels):
                histograms_pre[c] += torch.histc(data_batch[:, c, :, :], bins=num_bins, min=0, max=max_val).to(torch.int64)
                
        histograms_pre_np = histograms_pre.cpu().numpy()
        
        clip_max_array = np.zeros(num_channels)
        all_percentiles = {c: {} for c in range(num_channels)}
        percentiles_to_check = [0.01, 0.25, 0.50, 0.75, 0.90, 0.95, 0.98, 0.99, 0.999]

        for c in range(num_channels):
            cdf = np.cumsum(histograms_pre_np[c])
            cdf_norm = cdf / cdf[-1]
            for p in percentiles_to_check:
                idx = np.searchsorted(cdf_norm, p)
                all_percentiles[c][f"p_{p*100}"] = float(bin_edges[idx])
                
            if is_sim:
                clip_max_array[c] = 100.0
            else:
                idx_98 = np.searchsorted(cdf_norm, 0.98)
                clip_max_array[c] = bin_edges[idx_98]

        pixel_sum = torch.zeros(num_channels, dtype=torch.float64, device=device)
        pixel_sum_sq = torch.zeros(num_channels, dtype=torch.float64, device=device)
        histograms_post = torch.zeros((num_channels, num_bins), dtype=torch.int64, device=device)

        clip_max_tensor = torch.tensor(clip_max_array, dtype=torch.float32, device=device).view(1, num_channels, 1, 1)

        for i in tqdm(range(0, num_valid, chunk_size), desc="Pass 2"):
            end = min(i + chunk_size, num_valid)
            data_batch = torch.from_numpy(ds[valid_indices[i:end]]).to(device, dtype=torch.float32)
            
            if offset != 0.0:
                data_batch = data_batch - offset
            data_batch = torch.clamp(data_batch, min=0)
            data_batch = torch.sqrt(data_batch)
            
            data_batch = torch.minimum(data_batch, clip_max_tensor)
            
            pixel_sum += torch.sum(data_batch, dim=(0, 2, 3))
            pixel_sum_sq += torch.sum(data_batch**2, dim=(0, 2, 3))
            
            for c in range(num_channels):
                histograms_post[c] += torch.histc(data_batch[:, c, :, :], bins=num_bins, min=0, max=max_val).to(torch.int64)

        histograms_post_np = histograms_post.cpu().numpy()
        mean = (pixel_sum.cpu().numpy() / total_pixels_count)
        variance = (pixel_sum_sq.cpu().numpy() / total_pixels_count) - (mean**2)
        std = np.sqrt(np.maximum(variance, 0))

        results = {
            "dataset": ds_name,
            "is_sim": is_sim,
            "offset_applied": offset,
            "stats": {}
        }

        for c in range(num_channels):
            norm_hist = histograms_post_np[c] / np.sum(histograms_post_np[c])
            m3 = np.sum(norm_hist * (bin_centers - mean[c])**3)
            m4 = np.sum(norm_hist * (bin_centers - mean[c])**4)
            skewness = m3 / (std[c]**3 + 1e-8)
            kurtosis = m4 / (std[c]**4 + 1e-8) - 3 
            
            results["stats"][f"band_{c}"] = {
                "mean": float(mean[c]),
                "std": float(std[c]),
                "skewness": float(skewness),
                "kurtosis": float(kurtosis),
                "clip_value": float(clip_max_array[c]),
                "percentiles_pre_clip": all_percentiles[c]
            }

        with open(os.path.join(save_dir, f"diag2_{dataset_id}.json"), 'w') as f:
            json.dump(results, f, indent=4)
            
        np.savez_compressed(
            os.path.join(save_dir, f"raw_histograms_{dataset_id}.npz"), 
            hist_pre=histograms_pre_np,
            hist_post=histograms_post_np,
            bins=bin_edges
        )

if __name__ == "__main__":
    run_mega_diagnostic(ds_name="real/images", offset=0.0, is_sim=False)
    run_mega_diagnostic(ds_name="sim/images", offset=0.0, is_sim=True)