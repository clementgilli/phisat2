from __future__ import annotations

import os
from pathlib import Path

import rasterio
import numpy as np
import torch
import lightning as L
from torch.utils.data import DataLoader, Dataset

from phisat2.tasks import TaskSpec
from phisat2.data_loaders.sensors import PHISAT2_REAL_BANDS, get_norm_tensors
from phisat2.data_loaders.transforms import normalize_tensor

EUROSAT_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake"
]
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(EUROSAT_CLASSES)}

EUROSAT_SOURCE_BANDS = [
    "BLUE", "GREEN", "RED", "PAN", "NIR_BROAD", "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3"
]

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
        scaling_factor: float = 1.0,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.spec = spec
        self.split = split
        self.scaling_factor = scaling_factor

        try:
            self.permutation = [EUROSAT_SOURCE_BANDS.index(b) for b in PHISAT2_REAL_BANDS]
        except ValueError as exc:
            raise ValueError(
                f"Cannot map {EUROSAT_SOURCE_BANDS} → {PHISAT2_REAL_BANDS}."
            ) from exc

        self.mean, self.std = get_norm_tensors("phisat2_sim", PHISAT2_REAL_BANDS)
        
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
        
        image = image[self.permutation]
        
        image = image / self.scaling_factor

        image = normalize_tensor(image, self.mean, self.std)
        
        target = torch.tensor(label, dtype=torch.long)

        return {"image": image, self.spec.target_key: target}

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
        scaling_factor: float = 1.0,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.spec = spec
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.scaling_factor = scaling_factor
        self.input_bands = PHISAT2_REAL_BANDS

    def setup(self, stage: str | None = None) -> None:
        if stage in {None, "fit", "validate"}:
            self.train_dataset = EuroSatDataset(
                self.root_dir, self.spec, split="train", seed=self.seed, 
                scaling_factor=self.scaling_factor
            )
            self.val_dataset = EuroSatDataset(
                self.root_dir, self.spec, split="val", seed=self.seed, 
                scaling_factor=self.scaling_factor
            )

        if stage in {None, "test", "predict"}:
            self.test_dataset = EuroSatDataset(
                self.root_dir, self.spec, split="test", seed=self.seed, 
                scaling_factor=self.scaling_factor
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