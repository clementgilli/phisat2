import os
import h5py
import numpy as np
import torch
from tqdm import tqdm

# --- CONFIGURATION ---
H5_PATH = "/shared/projects/phisat2/data/processed/triplets_v1/phisat2_s2b_dataset_v1.h5"

BAD_PRODUCT_IDS = [
    1296, 1342, 1385, 1397, 1420, 1460, 1497, 1647, 1854, 2223, 2246, 2259, 
    2373, 2631, 2640, 2743, 2834, 2853, 3374, 3619, 4071, 4693, 4813, 4942, 
    2352, 2882, 3322, 3914, 4702, 1333, 1466, 1615, 2460, 2729, 2763
]

def process_real_images_gpu():
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    ds_name = "real/images"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Exact Stats Extraction on {device} for {ds_name} ---")

    with h5py.File(H5_PATH, 'r') as f:
        all_ids = f["metadata/product_id"][:]
        mask = ~np.isin(all_ids, BAD_PRODUCT_IDS)
        valid_indices = np.where(mask)[0].tolist() # Liste native pour h5py
        num_valid = len(valid_indices)
        
        ds = f[ds_name]
        _, num_channels, h, w = ds.shape
        
        # On réduit le chunk_size pour plus de fluidité visuelle
        chunk_size = 256 
        
        # =================================================================
        # STEP 1: EXACT 99.5th PERCENTILE VIA HISTOGRAMS (GPU Accelerated)
        # =================================================================
        print(f"\nStep 1: Computing exact percentiles per band on {num_valid} images...")
        
        num_bins = 30000
        max_val = 70.0  # Parfait pour du 12-bits (sqrt(4095) = 63.99)
        bin_edges = np.linspace(0, max_val, num_bins + 1)
        histograms = np.zeros((num_channels, num_bins), dtype=np.int64)

        for i in tqdm(range(0, num_valid, chunk_size), desc="Pass 1: GPU Histograms"):
            end = min(i + chunk_size, num_valid)
            
            # Lecture H5 (CPU) -> Envoi direct sur GPU
            # h5py préfère une liste Python triée pour lire des indices non-continus
            idx_list = valid_indices[i:end]
            data_numpy = ds[idx_list]
            
            data_batch = torch.from_numpy(data_numpy).to(device, dtype=torch.float32)
            data_batch = torch.clamp(data_batch, min=0)
            data_batch = torch.sqrt(data_batch)
            
            # Calcul de l'histogramme ultra rapide sur GPU
            for c in range(num_channels):
                # torch.histc crée l'histogramme en C++/CUDA
                hist = torch.histc(data_batch[:, c, :, :], bins=num_bins, min=0, max=max_val)
                histograms[c] += hist.cpu().numpy().astype(np.int64)
                
        # Extraction des percentiles (sur CPU, instantané)
        clip_max_array = np.zeros(num_channels)
        percentiles_to_check = [0.90, 0.95, 0.98, 0.99, 0.995, 0.999]
        
        print("\n--- Diagnostic de la queue de distribution ---")
        for p in percentiles_to_check:
            temp_clip_array = np.zeros(num_channels)
            for c in range(num_channels):
                cdf = np.cumsum(histograms[c])
                cdf_norm = cdf / cdf[-1]
                idx = np.searchsorted(cdf_norm, p) 
                temp_clip_array[c] = bin_edges[idx]
            
            print(f"-> {p*100}% max_clip : {np.round(temp_clip_array, 2).tolist()}")
            
            # On assigne silencieusement le 98% à la variable d'origine du script
            if p == 0.98:
                clip_max_array = temp_clip_array.copy()
                
        print("----------------------------------------------")
        
        
        
            
        print(f"-> EXACT max_clip per band: {np.round(clip_max_array, 2).tolist()}")
        
        
        
        # =================================================================
        # STEP 2: EXACT MEAN & STD (GPU Accelerated)
        # =================================================================
        print("\nStep 2: Full dataset pass to compute exact Mean & Std...")
        pixel_sum = torch.zeros(num_channels, dtype=torch.float64, device=device)
        pixel_sum_sq = torch.zeros(num_channels, dtype=torch.float64, device=device)
        
        # Variables CPU pour les stats min/max brutes (moins critiques)
        raw_min = np.full(num_channels, np.inf)
        raw_max = np.full(num_channels, -np.inf)
        
        total_pixels_count = num_valid * h * w

        # Préparation du tenseur pour le broadcast du clip sur le GPU (1, 8, 1, 1)
        clip_max_tensor = torch.tensor(clip_max_array, dtype=torch.float32, device=device).view(1, num_channels, 1, 1)

        for i in tqdm(range(0, num_valid, chunk_size), desc="Pass 2: GPU Mean & Std"):
            end = min(i + chunk_size, num_valid)
            idx_list = valid_indices[i:end]
            data_numpy = ds[idx_list]
            
            # Maj des min/max bruts sur CPU
            raw_min = np.minimum(raw_min, np.min(data_numpy, axis=(0, 2, 3)))
            raw_max = np.maximum(raw_max, np.max(data_numpy, axis=(0, 2, 3)))
            
            # Passage GPU pour les grosses maths
            data_batch = torch.from_numpy(data_numpy).to(device, dtype=torch.float32)
            data_batch = torch.clamp(data_batch, min=0)
            data_batch = torch.sqrt(data_batch)
            
            # Clipping dynamique
            data_batch = torch.minimum(data_batch, clip_max_tensor)
            
            # Cumul sur GPU (très rapide)
            pixel_sum += torch.sum(data_batch, dim=(0, 2, 3))
            pixel_sum_sq += torch.sum(data_batch**2, dim=(0, 2, 3))

        # Rapatriement sur CPU pour le calcul final
        pixel_sum = pixel_sum.cpu().numpy()
        pixel_sum_sq = pixel_sum_sq.cpu().numpy()

        mean = pixel_sum / total_pixels_count
        variance = (pixel_sum_sq / total_pixels_count) - (mean**2)
        std = np.sqrt(np.maximum(variance, 0))

        # =================================================================
        # FINAL OUTPUT
        # =================================================================
        print("\n" + "="*60)
        print(" FINAL CONFIGURATION FOR YAML (real/images) ")
        print("="*60)
        print(f"max_clip: {list(np.round(clip_max_array, 4))}")
        print(f"mean:     {list(np.round(mean, 4))}")
        print(f"std:      {list(np.round(std, 4))}")
        print("="*60)

if __name__ == "__main__":
    process_real_images_gpu()