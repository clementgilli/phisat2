from __future__ import annotations

import os
from pathlib import Path

import rasterio
import numpy as np
import torch
import torch.nn.functional as F
import lightning as L
from torch.utils.data import DataLoader, Dataset

from phisat2.tasks import TaskSpec
from phisat2.data_loaders.sensors import PHISAT2_REAL_BANDS, get_norm_tensors
from phisat2.data_loaders.transforms import normalize_tensor, upscale_to_phisat2, extract_phisat2_bands

EUROSAT_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake"
]
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(EUROSAT_CLASSES)}


class EuroSatDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        spec: TaskSpec,
        *,
        split: str = "test",
        seed: int = 42,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.spec = spec
        self.split = split

        self.s2_mean, self.s2_std = get_norm_tensors("s2", PHISAT2_REAL_BANDS)
        
        self.samples = self._build_stratified_split(
            self.root_dir, split, seed, train_ratio, val_ratio
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        img_path, label = self.samples[index]

        with rasterio.open(img_path) as src:
            image_array = src.read()
            
        image = torch.from_numpy(image_array).float()
        
        image = upscale_to_phisat2(image, is_mask=False)
        
        image = extract_phisat2_bands(image)
        
        image = normalize_tensor(image, self.s2_mean, self.s2_std)
        
        target = torch.tensor(label, dtype=torch.long)

        return {"sentinel2_phisat2": image, self.spec.target_key: target}

    @staticmethod
    def _build_stratified_split(
        root_dir: Path, 
        split: str, 
        seed: int, 
        train_ratio: float, 
        val_ratio: float
    ) -> list[tuple[Path, int]]:
        
        samples_for_split = []
        rng = np.random.default_rng(seed)
        
        for cls_name in EUROSAT_CLASSES:
            cls_dir = root_dir / "eurosat" / cls_name
            
            if not cls_dir.exists():
                continue
                
            files = list(cls_dir.glob("*.tif")) + list(cls_dir.glob("*.tiff"))
            files = sorted(files)
            
            rng.shuffle(files)
            
            n_total = len(files)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)
            
            if split == "train":
                split_files = files[:n_train]
            elif split == "val":
                split_files = files[n_train:n_train + n_val]
            elif split == "test":
                split_files = files[n_train + n_val:]
            else:
                raise ValueError(f"Unknown split: {split}")
                
            label = CLASS_TO_IDX[cls_name]
            samples_for_split.extend([(f, label) for f in split_files])
            
        samples_for_split.sort(key=lambda x: str(x[0]))
        
        return samples_for_split


class EuroSatDataModule(L.LightningDataModule):
    def __init__(
        self,
        root_dir: str | Path,
        spec: TaskSpec,
        batch_size: int,
        num_workers: int,
        seed: int,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.spec = spec
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        
        self.input_bands = PHISAT2_REAL_BANDS

    def setup(self, stage: str | None = None) -> None:
        if stage in {None, "fit", "validate"}:
            self.train_dataset = EuroSatDataset(
                self.root_dir, self.spec, split="train", seed=self.seed
            )
            self.val_dataset = EuroSatDataset(
                self.root_dir, self.spec, split="val", seed=self.seed
            )

        if stage in {None, "test", "predict"}:
            self.test_dataset = EuroSatDataset(
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