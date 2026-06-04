from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader, Dataset

from phisat2.tasks.specs import TaskSpec

PHISAT2_REAL_BANDS = [
    "PAN", "BLUE", "GREEN", "RED", 
    "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3", "NIR_BROAD"
]

PHISAT2_SIM_BANDS = [
    "BLUE", "GREEN", "RED", "PAN", 
    "NIR_BROAD", "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3"
]

S2_BANDS = [
    "COASTAL_AEROSOL", "BLUE", "GREEN", "RED", "RED_EDGE_1", "RED_EDGE_2", 
    "RED_EDGE_3", "NIR_BROAD", "NIR_NARROW", "WATER_VAPOR", "CIRRUS", "SWIR_1", "SWIR_2"
]

SIM_TO_REAL_PERMUTATION = [3, 0, 1, 2, 5, 6, 7, 4]


class TripletsDataset(Dataset):
    def __init__(
        self, 
        root_dir: str | Path, 
        split_pairs: list[str] | None = None,
        max_patches: int | None = None
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.pairs_dir = self.root_dir / "pairs"
        self.samples = self._index_files(split_pairs)
        
        if max_patches is not None:
            self.samples = self.samples[:max_patches]

    def _index_files(self, split_pairs: list[str] | None) -> list[dict[str, Path]]:
        samples = []
        if not self.pairs_dir.exists():
            warnings.warn(f"Folder {self.pairs_dir} does not exist.")
            return samples

        all_pair_paths = sorted([p for p in self.pairs_dir.iterdir() if p.is_dir() and p.name.startswith("pair_")])

        for pair_path in all_pair_paths:
            if split_pairs is not None and pair_path.name not in split_pairs:
                continue

            base_tiles = pair_path / "lightglue_coregistration"
            sim_dir = base_tiles / "tiles" / "simulated_phisat2"
            real_dir = base_tiles / "tiles" / "phisat2"
            s2_dir = base_tiles / "tiles" / "sentinel2"
            cloud_dir = base_tiles / "masks" / "phisat2_cloud"
            wc_dir = base_tiles / "masks" / "worldcover"

            if not sim_dir.exists():
                continue

            for sim_tile in sim_dir.glob("*.tif"):
                tile_name = sim_tile.name
                
                real_tile = real_dir / tile_name
                s2_tile = s2_dir / tile_name
                cloud_mask = cloud_dir / tile_name
                wc_mask = wc_dir / tile_name

                if real_tile.exists() and s2_tile.exists():
                    samples.append({
                        "simulated": sim_tile,
                        "real": real_tile,
                        "sentinel2": s2_tile,
                        "cloud": cloud_mask if cloud_mask.exists() else None,
                        "worldcover": wc_mask if wc_mask.exists() else None,
                        "tile_id": f"{pair_path.name}_{tile_name}"
                    })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _read_tif(self, path: Path | None, is_mask: bool = False, ref_shape: tuple[int, int] = (224, 224)) -> torch.Tensor:
        if path is None:
            return torch.zeros(ref_shape, dtype=torch.int64 if is_mask else torch.float32)
            
        with rasterio.open(path) as src:
            data = src.read()
            
        return torch.from_numpy(data.astype(np.float32 if not is_mask else np.int64))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        sample_paths = self.samples[idx]
        
        sim_tensor = self._read_tif(sample_paths["simulated"])
        real_tensor = self._read_tif(sample_paths["real"])
        s2_tensor = self._read_tif(sample_paths["sentinel2"])
        
        if sim_tensor.shape[0] == 8:
            sim_tensor = sim_tensor[SIM_TO_REAL_PERMUTATION]
        
        spatial_shape = tuple(sim_tensor.shape[-2:])
        
        cloud_tensor = self._read_tif(sample_paths["cloud"], is_mask=True, ref_shape=spatial_shape)
        wc_tensor = self._read_tif(sample_paths["worldcover"], is_mask=True, ref_shape=spatial_shape)
        
        return {
            "simulated": sim_tensor,
            "real": real_tensor,
            "sentinel2": s2_tensor,
            "mask_cloud": cloud_tensor,
            "mask_worldcover": wc_tensor,
            "tile_id": sample_paths["tile_id"]
        }


class TripletsDataModule(L.LightningDataModule):
    def __init__(
        self,
        root_dir: str | Path,
        spec: TaskSpec,
        *,
        batch_size: int,
        num_workers: int = 4,
        seed: int = 42,
        val_ratio: float = 0.1,
        fast_dev_run: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.spec = spec
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.val_ratio = val_ratio
        self.fast_dev_run = fast_dev_run
        
        self.input_bands = PHISAT2_REAL_BANDS
        self.s2_bands = S2_BANDS

    def _get_pair_splits(self) -> tuple[list[str], list[str]]:
        pairs_dir = self.root_dir / "pairs"
        if not pairs_dir.exists():
            return [], []
            
        all_pairs = sorted([p.name for p in pairs_dir.iterdir() if p.is_dir() and p.name.startswith("pair_")])
        
        if not all_pairs:
            return [], []

        rng = np.random.default_rng(self.seed)
        val_count = max(1, int(len(all_pairs) * self.val_ratio))
        val_indices = set(rng.choice(len(all_pairs), size=val_count, replace=False).tolist())
        
        train_pairs = [pair for idx, pair in enumerate(all_pairs) if idx not in val_indices]
        val_pairs = [pair for idx, pair in enumerate(all_pairs) if idx in val_indices]
        
        return train_pairs, val_pairs

    def setup(self, stage: str | None = None) -> None:
        max_patches = self.batch_size if self.fast_dev_run else None
        
        if stage == "fit" or stage is None:
            train_pairs, val_pairs = self._get_pair_splits()
            
            if not val_pairs:
                train_pairs = val_pairs = [p.name for p in (self.root_dir / "pairs").iterdir() if p.is_dir()]
            
            self.train_dataset = TripletsDataset(
                self.root_dir, 
                split_pairs=train_pairs, 
                max_patches=max_patches
            )
            self.val_dataset = TripletsDataset(
                self.root_dir, 
                split_pairs=val_pairs, 
                max_patches=max_patches
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=not self.fast_dev_run,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers
        )