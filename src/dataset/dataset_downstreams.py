import os
import zarr
import torch
import numpy as np
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

#TODO
BAND_PERMUTATIONS = {
    'worldfloods': [0, 1, 2, 3, 4, 5, 6, 7],
    'burned_area': [0, 1, 2, 3, 4, 5, 6, 7],
    'lulc':        [0, 1, 2, 3, 4, 5, 6, 7],
    'marine_area': [0, 1, 2, 3, 4, 5, 6, 7],
    'clouds':      [0, 1, 2, 3, 4, 5, 6, 7],
}

#TODO
MEANS = torch.tensor([49.7866, 49.0253, 48.4297, 49.2364, 51.1648, 55.4065, 57.3572, 56.7808]).view(8, 1, 1) 
STDS = torch.tensor([7.2800, 6.5203, 6.9570, 9.0981, 8.3858, 7.9555, 8.3155, 8.3664]).view(8, 1, 1)

class DownstreamDataset(Dataset):
    def __init__(self, root_dir, task_name, split='train', val_ratio=0.1, seed=42, crop_size=224):
        self.task_name = task_name
        self.split = split
        self.crop_size = crop_size
        
        if task_name not in BAND_PERMUTATIONS:
            raise ValueError(f"Task '{task_name}' not found in BAND_PERMUTATIONS.")
        self.permutation = BAND_PERMUTATIONS[task_name]
        
        base_path = Path(root_dir) / f"{task_name}.zarr"
        source_folder = base_path / 'trainval' if split in ['train', 'val'] else base_path / 'test'
            
        def find_max_patch(folder_path):
            low, high, best = 0, 5_000_000, -1
            while low <= high:
                mid = (low + high) // 2
                if (folder_path / f"{mid:07d}").exists():
                    best = mid
                    low = mid + 1
                else:
                    high = mid - 1
            return best + 1

        num_patches = find_max_patch(source_folder)
        if num_patches == 0:
            raise FileNotFoundError(f"No patches found in {source_folder}")
            
        all_patches = [str(source_folder / f"{i:07d}") for i in range(num_patches)]
                
        if split in ['train', 'val']:
            rng = np.random.RandomState(seed)
            rng.shuffle(all_patches)
            
            num_val = int(len(all_patches) * val_ratio)
            self.patches = all_patches[:num_val] if split == 'val' else all_patches[num_val:]
        else:
            self.patches = all_patches
            
        print(f"[INFO] Initialized {task_name} - Split: {split} - Samples: {len(self.patches)}")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch_path = self.patches[idx]
        
        group = zarr.open(patch_path, mode='r', synchronizer=None)
        img_np = group['img'][:]
        label_np = group['label'][:]
        
        img = torch.from_numpy(img_np).float()
        label = torch.from_numpy(label_np).long()
        
        if label.ndim == 3 and label.shape[0] == 1:
            label = label.squeeze(0)

        img = img[self.permutation, :, :]
        
        img = torch.sqrt(img)
        img = torch.clamp(img, max=100.0)
        img = (img - MEANS) / STDS
        
        if self.split == 'train':
            i, j, h, w = torch.randint(0, img.shape[1] - self.crop_size + 1, (1,)).item(), \
                         torch.randint(0, img.shape[2] - self.crop_size + 1, (1,)).item(), \
                         self.crop_size, self.crop_size
            img = TF.crop(img, i, j, h, w)
            label = TF.crop(label, i, j, h, w)
        else:
            img = TF.center_crop(img, [self.crop_size, self.crop_size])
            label = TF.center_crop(label, [self.crop_size, self.crop_size])
            
        return {"img": img, "label": label}

def get_dataloaders(root_dir, task_name, batch_size=32, num_workers=4):
    ds_train = DownstreamDataset(root_dir, task_name, split='train')
    ds_val   = DownstreamDataset(root_dir, task_name, split='val')
    ds_test  = DownstreamDataset(root_dir, task_name, split='test')
    
    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    loader_val   = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    loader_test  = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return loader_train, loader_val, loader_test