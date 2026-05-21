import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision.transforms import RandomCrop
from tqdm import tqdm

def compute_and_cache_lut(train_indices, h5_path, config, cache_path):
    """
    Computes the Optimal Transport LUT (CDF matching) strictly on the train set.
    """
    num_channels = 8
    num_bins = 30000
    max_val = 120.0
    
    hist_real = np.zeros((num_channels, num_bins), dtype=np.int64)
    hist_sim = np.zeros((num_channels, num_bins), dtype=np.int64)
    
    chunk_size = 256
    num_valid = len(train_indices)
    
    sorted_indices = sorted(train_indices)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    real_offset = config["normalization"]["real"].get("offset", 0.0)
    sim_offset = config["normalization"]["sim"].get("offset", 0.0)
    
    with h5py.File(h5_path, 'r') as f:
        for i in tqdm(range(0, num_valid, chunk_size), desc="Computing OT Histograms (Train Set)"):
            end = min(i + chunk_size, num_valid)
            idx_chunk = sorted_indices[i:end]
            
            # Process REAL
            batch_real = torch.from_numpy(f["real/images"][idx_chunk]).to(device, dtype=torch.float32)
            if real_offset != 0.0:
                batch_real = batch_real - real_offset
            batch_real = torch.clamp(batch_real, min=0)
            batch_real = torch.sqrt(batch_real)
            
            # Process SIM
            batch_sim = torch.from_numpy(f["sim/images"][idx_chunk]).to(device, dtype=torch.float32)
            if sim_offset != 0.0:
                batch_sim = batch_sim - sim_offset
            batch_sim = torch.clamp(batch_sim, min=0)
            batch_sim = torch.sqrt(batch_sim)
            
            # Accumulate histograms
            for c in range(num_channels):
                hist_real[c] += torch.histc(batch_real[:, c, :, :], bins=num_bins, min=0, max=max_val).cpu().numpy().astype(np.int64)
                hist_sim[c] += torch.histc(batch_sim[:, c, :, :], bins=num_bins, min=0, max=max_val).cpu().numpy().astype(np.int64)
                
    print("Histograms extracted. Building CDFs and interpolating LUT...")
    lut = np.zeros((num_channels, num_bins), dtype=np.float32)
    bin_edges = np.linspace(0, max_val, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    for c in range(num_channels):
        cdf_real = np.cumsum(hist_real[c]) / np.sum(hist_real[c])
        cdf_sim = np.cumsum(hist_sim[c]) / np.sum(hist_sim[c])
        lut[c] = np.interp(cdf_real, cdf_sim, bin_centers)
        
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, lut)
    
    return lut

class PhiSatTransform:
    def __init__(self, mean, std, max_clip, offset=0.0, lut=None):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(8, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(8, 1, 1)
        self.offset = offset
        self.lut = lut

        if isinstance(max_clip, (list, tuple)):
            self.max_clip = torch.tensor(max_clip, dtype=torch.float32).view(8, 1, 1)
        else:
            self.max_clip = torch.tensor(max_clip, dtype=torch.float32)
            
        if self.lut is not None:
            bin_edges = np.linspace(0, 120.0, 30000 + 1)
            self.bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    def __call__(self, img):
        img_tensor = torch.from_numpy(img).to(torch.float32)
        
        if self.offset != 0.0:
            img_tensor = img_tensor - self.offset
            
        img_tensor = torch.clamp(img_tensor, min=0)
        img_tensor = torch.sqrt(img_tensor)
        
        # Apply Optimal Transport LUT if provided (Baseline 1)
        if self.lut is not None:
            img_np = img_tensor.numpy()
            img_adapted = np.zeros_like(img_np)
            for c in range(img_tensor.shape[0]):
                img_adapted[c] = np.interp(img_np[c], self.bin_centers, self.lut[c])
            img_tensor = torch.from_numpy(img_adapted).to(torch.float32)
        
        img_tensor = torch.minimum(img_tensor, self.max_clip)
        img_tensor = (img_tensor - self.mean) / (self.std + 1e-8)
        
        return img_tensor

class PairedPhiSatDataset(Dataset):
    def __init__(self, h5_path, indices, split='train', transform_sim=None, transform_real=None, band_order=None):
        self.h5_path = h5_path
        self.valid_indices = indices  
        self.split = split            
        self.transform_sim = transform_sim
        self.transform_real = transform_real
        self.band_order = band_order
        self.length = len(self.valid_indices)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        h5_idx = self.valid_indices[idx]
        
        with h5py.File(self.h5_path, 'r') as f:
            img_sim = f["sim/images"][h5_idx].astype(np.float32)
            img_real = f["real/images"][h5_idx].astype(np.float32)
            
            climate = f["metadata/koppen_zone"][h5_idx]
            lat = f["metadata/center_lat"][h5_idx]
            lon = f["metadata/center_lon"][h5_idx]
            
            patch_id = h5_idx  # For debugging purposes
            
        if self.transform_sim:
            img_sim = self.transform_sim(img_sim)
        if self.transform_real:
            img_real = self.transform_real(img_real)
            
        if self.split == 'train':
            i, j, h, w = RandomCrop.get_params(img_sim, output_size=(224, 224))
            img_sim = TF.crop(img_sim, i, j, h, w)
            img_real = TF.crop(img_real, i, j, h, w)
        else:
            img_sim = TF.center_crop(img_sim, output_size=(224, 224))
            img_real = TF.center_crop(img_real, output_size=(224, 224))
            
        # Original: [0, 1, 2, 3, 4, 5, 6, 7]
        # Target:   [1, 2, 3, 0, 7, 4, 5, 6]
        img_sim = img_sim[self.band_order, :, :]
        img_real = img_real[self.band_order, :, :]    
        
        return {
            'sim': img_sim,
            'real': img_real,
            'climate': torch.tensor(climate, dtype=torch.long),
            'lat': torch.tensor(lat, dtype=torch.float32),
            'lon': torch.tensor(lon, dtype=torch.float32),
            'patch_id': torch.tensor(patch_id, dtype=torch.long)
        }
        
    def get_climate_class_distribution(self):
        with h5py.File(self.h5_path, 'r') as f:
            all_labels = f["metadata/koppen_zone"][:]
            
        valid_labels = all_labels[self.valid_indices]
        unique_classes, counts = np.unique(valid_labels, return_counts=True)
        return dict(zip(unique_classes, counts))

def get_real_transform(baseline_mode, config, train_indices, h5_path, seed=42):
    """
    Factory function to build the correct REAL transform based on the selected baseline.
    """
    if baseline_mode == 0:
        # Baseline 0: Naive Transfer (Z-score SIM)
        return PhiSatTransform(
            mean=config["normalization"]["sim"]["mean"],
            std=config["normalization"]["sim"]["std"],
            max_clip=config["normalization"]["sim"]["max_clip"],
            offset=config["normalization"]["real"]["offset"]
        )
        
    elif baseline_mode == 0.5:
        # Baseline 0.5: Domain-Specific Norm (Z-score REAL)
        return PhiSatTransform(
            mean=config["normalization"]["real"]["mean"],
            std=config["normalization"]["real"]["std"],
            max_clip=config["normalization"]["real"]["max_clip"],
            offset=config["normalization"]["real"]["offset"]
        )
        
    elif baseline_mode == 1:
        # Baseline 1: Optimal Transport (LUT -> Clip SIM -> Z-score SIM)
        lut_cache_path = config["paths"].get("lut_cache", f"cache/ot_lut_seed_{seed}.npy")
        
        if os.path.exists(lut_cache_path):
            print(f"Loading cached OT LUT from: {lut_cache_path}")
            lut = np.load(lut_cache_path)
        else:
            print(f"LUT cache not found. Calculating on {len(train_indices)} train samples...")
            lut = compute_and_cache_lut(train_indices, h5_path, config, lut_cache_path)
            
        return PhiSatTransform(
            mean=config["normalization"]["sim"]["mean"],
            std=config["normalization"]["sim"]["std"],
            max_clip=config["normalization"]["sim"]["max_clip"], 
            offset=config["normalization"]["real"]["offset"],
            lut=lut
        )
    else:
        raise ValueError(f"Unknown baseline_mode: {baseline_mode}")

def get_dataloaders(config, baseline_mode=1, batch_size=32, num_workers=4, seed=42):
    
    h5_path = config["paths"]["h5_dataset"]
    bad_ids = config["dataset"].get("bad_product_ids", [])
    band_order = config["band_order"]
    
    with h5py.File(h5_path, 'r') as f:
        total_length = f["sim/images"].shape[0]
        if bad_ids is not None and len(bad_ids) > 0:
            print(f"Filtering {len(bad_ids)} bad product_ids...")
            all_ids = f["metadata/product_id"][:] 
            valid_indices = np.where(~np.isin(all_ids, bad_ids))[0]
            print(f"Keeping {len(valid_indices)} out of {total_length} ({len(valid_indices)/total_length*100:.2f}%).")
        else:
            valid_indices = np.arange(total_length)

    generator = torch.Generator().manual_seed(seed)
    shuffled_positions = torch.randperm(len(valid_indices), generator=generator).numpy()
    valid_indices = valid_indices[shuffled_positions]
    
    total_size = len(valid_indices)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_indices = valid_indices[:train_size]
    val_indices = valid_indices[train_size:train_size + val_size]
    test_indices = valid_indices[train_size + val_size:]
    
    transform_sim = PhiSatTransform(
        mean=config["normalization"]["sim"]["mean"],
        std=config["normalization"]["sim"]["std"],
        max_clip=config["normalization"]["sim"]["max_clip"],
        offset=config["normalization"]["sim"]["offset"]
    )

    transform_real = get_real_transform(baseline_mode, config, train_indices, h5_path, seed)
    
    train_dataset = PairedPhiSatDataset(h5_path, train_indices, split='train', transform_sim=transform_sim, transform_real=transform_real, band_order=band_order)
    val_dataset   = PairedPhiSatDataset(h5_path, val_indices, split='val', transform_sim=transform_sim, transform_real=transform_real, band_order=band_order)
    test_dataset  = PairedPhiSatDataset(h5_path, test_indices, split='test', transform_sim=transform_sim, transform_real=transform_real, band_order=band_order)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    print(f"Data splits initialized -> Train: {train_size}, Val: {val_size}, Test: {test_size}")
    print(f"Running in Baseline Mode: {baseline_mode}")
    
    return train_loader, val_loader, test_loader