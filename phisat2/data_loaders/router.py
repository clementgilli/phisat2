from __future__ import annotations

import os
from pathlib import Path

import rasterio
import tifffile
import numpy as np
import torch
import torch.nn.functional as F
import lightning as L
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict

from phisat2.tasks import TaskSpec
from phisat2.data_loaders.sensors import PHISAT2_REAL_BANDS, get_norm_tensors
from phisat2.data_loaders.transforms import normalize_tensor, upscale_to_phisat2, extract_phisat2_bands, apply_spatial_transforms

ROUTER_CLASSES = ["Fire", "Burned", "Water", "Clouds"]

class RouterDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        spec: TaskSpec,
        *,
        split: str = "test",
        seed: int = 42,
        crop_size: int = 224,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.spec = spec
        self.split = split
        self.crop_size = crop_size

        self.s2_mean, self.s2_std = get_norm_tensors("s2", PHISAT2_REAL_BANDS)
        
        self.img_dir = self.root_dir / "images"
        self.labels_path = self.root_dir / "labels.npz"
        
        self.samples = self._build_stratified_split(
            self.img_dir, self.labels_path, split, seed, train_ratio, val_ratio
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        img_filename, label_array = self.samples[index]
        img_path = self.img_dir / img_filename

        image_array = tifffile.imread(str(img_path))
        
        if image_array.shape[0] < image_array.shape[-1]:
            pass
        else:
            image_array = np.transpose(image_array, (2, 0, 1))

        image = torch.from_numpy(image_array).float()
        
        image = upscale_to_phisat2(image, is_mask=False)
        image = extract_phisat2_bands(image)
        image = normalize_tensor(image, self.s2_mean, self.s2_std)
        
        target = torch.tensor(label_array, dtype=torch.float32)
        
        transformed = apply_spatial_transforms([image], is_train=self.split == "train", crop_size=self.crop_size)

        return {"sentinel2_phisat2": transformed[0], self.spec.target_key: target}

    @staticmethod
    def _build_stratified_split(
        img_dir: Path,
        labels_path: Path,
        split: str, 
        seed: int, 
        train_ratio: float, 
        val_ratio: float
    ) -> list[tuple[str, np.ndarray]]:
        
        data = np.load(labels_path)
        array_name = data.files[0]
        all_labels = data[array_name]
        
        powerset_groups = defaultdict(list)
        
        for idx, label_row in enumerate(all_labels):
            power_key = "".join(label_row.astype(int).astype(str))
            powerset_groups[power_key].append(idx)
            
        rng = np.random.default_rng(seed)
        
        split_indices = []
        
        for power_key, indices in powerset_groups.items():
            rng.shuffle(indices)
            
            n_total = len(indices)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)
            
            if n_train == 0 and n_total > 0:
                n_train = 1
                n_val = 0
            
            if split == "train":
                split_indices.extend(indices[:n_train])
            elif split == "val":
                split_indices.extend(indices[n_train:n_train + n_val])
            elif split == "test":
                split_indices.extend(indices[n_train + n_val:])
            else:
                raise ValueError(f"Unknown split: {split}")
                
        samples_for_split = []
        for idx in split_indices:
            img_filename = f"{idx:06d}.tif"
            samples_for_split.append((img_filename, all_labels[idx]))
            
        samples_for_split.sort(key=lambda x: x[0])
        
        return samples_for_split


class RouterDataModule(L.LightningDataModule):
    def __init__(
        self,
        root_dir: str | Path,
        spec: TaskSpec,
        batch_size: int,
        num_workers: int,
        seed: int,
        crop_size: int = 224
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.spec = spec
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.crop_size = crop_size
        
        self.input_bands = PHISAT2_REAL_BANDS

    def setup(self, stage: str | None = None) -> None:
        if stage in {None, "fit", "validate"}:
            self.train_dataset = RouterDataset(
                self.root_dir, self.spec, split="train", seed=self.seed, crop_size=self.crop_size
            )
            self.val_dataset = RouterDataset(
                self.root_dir, self.spec, split="val", seed=self.seed
            )

        if stage in {None, "test", "predict"}:
            self.test_dataset = RouterDataset(
                self.root_dir, self.spec, split="test", seed=self.seed
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
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )