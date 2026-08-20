from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader, Dataset

from phisat2.tasks.specs import TaskSpec
from phisat2.data_loaders.sensors import S2_BANDS, get_norm_tensors
from phisat2.data_loaders.transforms import apply_kd_transforms, upscale_to_phisat2, normalize_tensor

import warnings
from rasterio.errors import NotGeoreferencedWarning

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


class SSL4EODataset(Dataset):
    def __init__(
        self, 
        root_dir: str | Path, 
        split: str = "train",
        max_patches: int | None = None,
        is_train: bool = True,
        crop_size: int = 224
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.is_train = is_train
        self.crop_size = crop_size
        
        self.images_dir = self.root_dir / split
        
        self.samples = sorted(list(self.images_dir.glob("*_image.tif")))
        
        if max_patches is not None:
            self.samples = self.samples[:max_patches]
            
        self.s2_mean, self.s2_std = get_norm_tensors("s2", S2_BANDS)

    def __len__(self) -> int:
        return len(self.samples)

    def _read_tif(self, path: Path) -> torch.Tensor:
        with rasterio.open(path) as src:
            data = src.read()
        return torch.from_numpy(data.astype(np.float32))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        img_path = self.samples[idx]
        
        s2_tensor = self._read_tif(img_path)
        
        s2_tensor = upscale_to_phisat2(s2_tensor, is_mask=False)
        
        s2_tensor = normalize_tensor(s2_tensor, self.s2_mean, self.s2_std)
        
        tensor_views = apply_kd_transforms(
            s2_tensor,
            is_train=self.is_train,
            crop_size=self.crop_size,
            p_jitter=0.0,
            p_noise=0.0
        )
        
        return {
            "sentinel2": tensor_views["teacher"],  # (13, 224, 224)
            "sentinel2_augmented": tensor_views["student"],
            "image_id": img_path.name
        }


class SSL4EODataModule(L.LightningDataModule):
    def __init__(
        self,
        root_dir: str | Path,
        spec: TaskSpec | None = None,      
        *,
        batch_size: int,
        num_workers: int = 4,
        fast_dev_run: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.spec = spec
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fast_dev_run = fast_dev_run
        self.input_bands = S2_BANDS
        self.s2_bands = S2_BANDS

    def setup(self, stage: str | None = None) -> None:
        max_patches = self.batch_size if self.fast_dev_run else None
        
        if stage == "fit" or stage is None:
            self.train_dataset = SSL4EODataset(
                self.root_dir, 
                split="train", 
                max_patches=max_patches,
                is_train=True,
                crop_size=224
            )
            self.val_dataset = SSL4EODataset(
                self.root_dir, 
                split="val", 
                max_patches=max_patches,
                is_train=False,
                crop_size=224
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True
        )