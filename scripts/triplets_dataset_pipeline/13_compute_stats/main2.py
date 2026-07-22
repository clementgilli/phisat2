import os
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

# --- CONFIGURATION ---
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
    clip_strategy = "HARD CLIP @ 100.0" if is_sim else "PERCENTILE @ 98%"
    print(f"\n🚀 --- MEGA DIAGNOSTIC: {ds_name} ---")
    print(f"Device: {device} | Offset: {offset} | Strategy: {clip_strategy}")

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
        max_val = 120.0  # Couvre le max physique de S2 (100) et Phisat-2 (~64)
        bin_edges = np.linspace(0, max_val, num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # =================================================================
        # PASS 1: PRE-CLIP HISTOGRAMS & PERCENTILES
        # =================================================================
        print(f"[Pass 1/2] Extracting raw distributions ({num_valid} images)...")
        histograms_pre = np.zeros((num_channels, num_bins), dtype=np.int64)

        for i in tqdm(range(0, num_valid, chunk_size), desc="Pass 1", leave=False):
            end = min(i + chunk_size, num_valid)
            data_batch = torch.from_numpy(ds[valid_indices[i:end]]).to(device, dtype=torch.float32)
            
            if offset != 0.0:
                data_batch = data_batch - offset
            data_batch = torch.clamp(data_batch, min=0)
            data_batch = torch.sqrt(data_batch)
            
            for c in range(num_channels):
                hist = torch.histc(data_batch[:, c, :, :], bins=num_bins, min=0, max=max_val)
                histograms_pre[c] += hist.cpu().numpy().astype(np.int64)
                
        # --- Logic de Clipping ---
        clip_max_array = np.zeros(num_channels)
        all_percentiles = {c: {} for c in range(num_channels)}
        percentiles_to_check = [0.01, 0.25, 0.50, 0.75, 0.90, 0.95, 0.98, 0.99, 0.999]

        for c in range(num_channels):
            cdf = np.cumsum(histograms_pre[c])
            cdf_norm = cdf / cdf[-1]
            for p in percentiles_to_check:
                idx = np.searchsorted(cdf_norm, p)
                all_percentiles[c][f"p_{p*100}"] = float(bin_edges[idx])
                
            if is_sim:
                clip_max_array[c] = 100.0  # Hard clip physique
            else:
                idx_98 = np.searchsorted(cdf_norm, 0.98)
                clip_max_array[c] = bin_edges[idx_98]

        print(f"-> Applied Clip Values: {np.round(clip_max_array, 2).tolist()}")

        # =================================================================
        # PASS 2: EXACT STATS & POST-CLIP HISTOGRAMS
        # =================================================================
        print("[Pass 2/2] Computing Exact Mean, Std & Skew/Kurtosis...")
        pixel_sum = torch.zeros(num_channels, dtype=torch.float64, device=device)
        pixel_sum_sq = torch.zeros(num_channels, dtype=torch.float64, device=device)
        histograms_post = np.zeros((num_channels, num_bins), dtype=np.int64)

        clip_max_tensor = torch.tensor(clip_max_array, dtype=torch.float32, device=device).view(1, num_channels, 1, 1)

        for i in tqdm(range(0, num_valid, chunk_size), desc="Pass 2", leave=False):
            end = min(i + chunk_size, num_valid)
            data_batch = torch.from_numpy(ds[valid_indices[i:end]]).to(device, dtype=torch.float32)
            
            if offset != 0.0:
                data_batch = data_batch - offset
            data_batch = torch.clamp(data_batch, min=0)
            data_batch = torch.sqrt(data_batch)
            
            # Clipping (Hard pour SIM, 98% pour REAL)
            data_batch = torch.minimum(data_batch, clip_max_tensor)
            
            # Accumulation Stats
            pixel_sum += torch.sum(data_batch, dim=(0, 2, 3))
            pixel_sum_sq += torch.sum(data_batch**2, dim=(0, 2, 3))
            
            # Accumulation Histo Post-Clip
            for c in range(num_channels):
                hist = torch.histc(data_batch[:, c, :, :], bins=num_bins, min=0, max=max_val)
                histograms_post[c] += hist.cpu().numpy().astype(np.int64)

        # Calcul Maths CPU
        mean = (pixel_sum.cpu().numpy() / total_pixels_count)
        variance = (pixel_sum_sq.cpu().numpy() / total_pixels_count) - (mean**2)
        std = np.sqrt(np.maximum(variance, 0))

        # =================================================================
        # EXPORT & PLOT
        # =================================================================
        results = {
            "dataset": ds_name,
            "is_sim": is_sim,
            "offset_applied": offset,
            "clip_strategy": clip_strategy,
            "stats": {}
        }

        for c in range(num_channels):
            norm_hist = histograms_post[c] / np.sum(histograms_post[c])
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

        # JSON
        with open(os.path.join(save_dir, f"diag_{dataset_id}.json"), 'w') as f:
            json.dump(results, f, indent=4)
            
        # PLOT LOG
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(22, 10))
        fig.suptitle(f"{ds_name.upper()} - Log Distribution \nStrategy: {clip_strategy}", fontsize=16)
        
        for c in range(num_channels):
            ax = axes[c // 4, c % 4]
            ax.plot(bin_centers, histograms_post[c], color='red' if is_sim else 'blue')
            ax.set_yscale('log')
            ax.set_title(f"B{c} | Skew: {skewness:.2f} | Kurt: {kurtosis:.2f}")
            ax.grid(True, alpha=0.3)
            ax.axvline(mean[c], color='black', linestyle='--', label='Mean')
            ax.axvline(clip_max_array[c], color='orange', linestyle=':', label='Clip')
            ax.legend()
            
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"plot_log_{dataset_id}.png"))
        plt.close()

        print("\n" + "="*50)
        print(f"✅ RESULTS FOR YAML: {ds_name.upper()}")
        print("="*50)
        if not is_sim:
            print(f"max_clip: {list(np.round(clip_max_array, 4))}")
        else:
            print("max_clip: 100.0")
        print(f"mean:     {list(np.round(mean, 4))}")
        print(f"std:      {list(np.round(std, 4))}")
        print("="*50 + "\n")

if __name__ == "__main__":
    # 1. Dataset REAL : Pas d'offset, clip 98%
    run_mega_diagnostic(ds_name="real/images", offset=0.0, is_sim=False)
    
    # 2. Dataset SIM : Offset 1000.0, hard clip 100.0
    run_mega_diagnostic(ds_name="sim/images", offset=1000.0, is_sim=True)