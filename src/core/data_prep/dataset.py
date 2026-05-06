import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF
from torchvision.transforms import RandomCrop

import torch

class PhiSatTransform:
    def __init__(self, mean, std, max_clip):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(8, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(8, 1, 1)
        self.max_clip = max_clip

    def __call__(self, img):
        img_tensor = torch.from_numpy(img).to(torch.float32)
        
        img_tensor = torch.clamp(img_tensor, min=0)
        img_tensor = torch.sqrt(img_tensor)
        img_tensor = torch.clamp(img_tensor, max=self.max_clip)
        
        img_tensor = (img_tensor - self.mean) / (self.std + 1e-8)
        
        return img_tensor


class PairedPhiSatDataset(Dataset):
    def __init__(self, h5_path, indices, split='train', transform_sim=None, transform_real=None):
        self.h5_path = h5_path
        self.valid_indices = indices  # Receives pre-filtered and pre-split indices
        self.split = split            # 'train', 'val', or 'test'
        self.transform_sim = transform_sim
        self.transform_real = transform_real
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
            
        return {
            'sim': img_sim,
            'real': img_real,
            'climate': torch.tensor(climate, dtype=torch.long),
            'lat': torch.tensor(lat, dtype=torch.float32),
            'lon': torch.tensor(lon, dtype=torch.float32)
        }
        
    def get_climate_class_distribution(self):
        with h5py.File(self.h5_path, 'r') as f:
            all_labels = f["metadata/koppen_zone"][:]
            
        valid_labels = all_labels[self.valid_indices]
        unique_classes, counts = np.unique(valid_labels, return_counts=True)
        return dict(zip(unique_classes, counts))


def get_dataloaders(config, batch_size=32, num_workers=4, seed=42):
    
    h5_path = config["paths"]["h5_dataset"]
    bad_ids = config["dataset"].get("bad_product_ids", [])
    
    transform_sim = PhiSatTransform(
        mean=config["normalization"]["sim"]["mean"],
        std=config["normalization"]["sim"]["std"],
        max_clip=config["normalization"]["sim"]["max_clip"]
    )
    
    transform_real = PhiSatTransform(
        mean=config["normalization"]["real"]["mean"],
        std=config["normalization"]["real"]["std"],
        max_clip=config["normalization"]["real"]["max_clip"]
    )
    
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
    
    train_dataset = PairedPhiSatDataset(h5_path, train_indices, split='train', transform_sim=transform_sim, transform_real=transform_real)
    val_dataset   = PairedPhiSatDataset(h5_path, val_indices, split='val', transform_sim=transform_sim, transform_real=transform_real)
    test_dataset  = PairedPhiSatDataset(h5_path, test_indices, split='test', transform_sim=transform_sim, transform_real=transform_real)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    print(f"Train ({train_size}), Val ({val_size}), Test ({test_size})")
    
    return train_loader, val_loader, test_loader