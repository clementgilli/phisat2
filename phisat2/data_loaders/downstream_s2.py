from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader, Dataset

from phisat2.tasks.specs import TaskSpec
from phisat2.data_loaders.sensors import PHISAT2_REAL_BANDS, get_norm_tensors
from phisat2.data_loaders.transforms import apply_spatial_transforms, upscale_to_phisat2, normalize_tensor, extract_phisat2_bands

DATASET_NAMES = {
    "clouds": "clouds",
    "floods": "floods",
    "lulc": "lulc",
    "marine": "marine",
    "methane": "methane",
    "burned": "burned",
}

class DownstreamS2Dataset(Dataset):
    def __init__(
        self, 
        dataset_dir: str | Path,
        split: str = "train",
        max_patches: int | None = None,
        is_train: bool = True,
        crop_size: int = 224,
    ) -> None:
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.is_train = is_train
        self.crop_size = crop_size
        
        self.images_dir = self.dataset_dir / split / "images"
        self.labels_dir = self.dataset_dir / split / "labels"
        self.samples = sorted(list(self.images_dir.glob("*_image.tif")))
        
        if max_patches is not None:
            self.samples = self.samples[:max_patches]
            
        self.s2_mean, self.s2_std = get_norm_tensors("s2", PHISAT2_REAL_BANDS)

    def __len__(self) -> int:
        return len(self.samples)

    def _read_tif(self, path: Path, is_mask: bool = False) -> torch.Tensor:
        data = tifffile.imread(path)
        if not is_mask and data.ndim == 3:
            data = data.transpose(2, 0, 1)            
        tensor = torch.from_numpy(data.astype(np.int64 if is_mask else np.float32))
        return tensor.squeeze(0) if tensor.ndim == 3 and tensor.shape[0] == 1 else tensor

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        img_path = self.samples[idx]
        lbl_path = self.labels_dir / img_path.name.replace("_image.tif", "_label.tif")
        
        s2_tensor = self._read_tif(img_path, is_mask=False)[:13]
        if "marine" in str(self.dataset_dir): #offset ESA (post 2022)
            s2_tensor = s2_tensor - 1000
        
        mask_tensor = self._read_tif(lbl_path, is_mask=True)
        
        s2_tensor = upscale_to_phisat2(s2_tensor, is_mask=False)
        mask_tensor = upscale_to_phisat2(mask_tensor, is_mask=True)
        
        s2_tensor = extract_phisat2_bands(s2_tensor)
        
        s2_tensor = normalize_tensor(s2_tensor, self.s2_mean, self.s2_std)
        
        transformed = apply_spatial_transforms(
            [s2_tensor, mask_tensor],
            is_train=self.is_train,
            crop_size=self.crop_size
        )
        
        return {
            "sentinel2_phisat2": transformed[0],
            "mask": transformed[1],
            "image_id": img_path.name
        }

class DownstreamS2DataModule(L.LightningDataModule):
    def __init__(
        self,
        root_dir: str | Path,
        spec: TaskSpec,             
        *,
        batch_size: int,
        num_workers: int = 4,
        fast_dev_run: bool = False,
        crop_size: int = 224,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.spec = spec
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fast_dev_run = fast_dev_run
        self.crop_size = crop_size
        self.input_bands = PHISAT2_REAL_BANDS
        
        dataset_key = self.spec.dataset
        dataset_name = DATASET_NAMES.get(dataset_key, dataset_key)
            
        self.dataset_dir = self.root_dir / dataset_name

    def setup(self, stage: str | None = None) -> None:
        max_patches = self.batch_size if self.fast_dev_run else None
        
        def get_ds(split):
            return DownstreamS2Dataset(
                self.dataset_dir,
                split=split, 
                max_patches=max_patches, 
                is_train=(split=="train"), 
                crop_size=self.crop_size
            )

        if stage == "fit" or stage is None:
            self.train_dataset = get_ds("train")
            self.val_dataset = get_ds("val")
        if stage == "test" or stage is None:
            self.test_dataset = get_ds("test")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, 
                          num_workers=self.num_workers, pin_memory=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, 
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, 
                          num_workers=self.num_workers, pin_memory=True)