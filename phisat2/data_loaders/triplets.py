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
from phisat2.data_loaders.sensors import PHISAT2_SIM_BANDS, PHISAT2_REAL_BANDS, S2_BANDS, get_norm_tensors
from phisat2.data_loaders.transforms import apply_spatial_transforms, normalize_tensor

class TripletsDataset(Dataset):
    def __init__(
        self, 
        root_dir: str | Path, 
        split_pairs: list[str] | None = None,
        max_patches: int | None = None,
        is_train: bool = True,
        crop_size: int = 224
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.pairs_dir = self.root_dir / "pairs"
        self.samples = self._index_files(split_pairs)
        self.is_train = is_train
        self.crop_size = crop_size
        
        if max_patches is not None:
            self.samples = self.samples[:max_patches]
            
        self.sim_mean, self.sim_std = get_norm_tensors("phisat2_sim", PHISAT2_SIM_BANDS)
        self.real_mean, self.real_std = get_norm_tensors("phisat2_real", PHISAT2_REAL_BANDS)
        self.s2_mean, self.s2_std = get_norm_tensors("s2", S2_BANDS)
        self.sim_to_real_idx = [PHISAT2_SIM_BANDS.index(b) for b in PHISAT2_REAL_BANDS]

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

    def _read_tif(self, path: Path | None, is_mask: bool = False, ref_shape: tuple[int, int] = (256, 256)) -> torch.Tensor:
        if path is None:
            shape = (1, *ref_shape) if is_mask else ref_shape
            return torch.zeros(shape, dtype=torch.int64 if is_mask else torch.float32)
            
        with rasterio.open(path) as src:
            data = src.read()
            
        tensor = torch.from_numpy(data.astype(np.float32 if not is_mask else np.int64))
        
        if is_mask and tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
            
        return tensor

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        sample_paths = self.samples[idx]
        
        sim_tensor = self._read_tif(sample_paths["simulated"])
        real_tensor = self._read_tif(sample_paths["real"])
        s2_tensor = self._read_tif(sample_paths["sentinel2"])
        
        sim_tensor = normalize_tensor(sim_tensor, self.sim_mean, self.sim_std)
        real_tensor = normalize_tensor(real_tensor, self.real_mean, self.real_std)
        s2_tensor = normalize_tensor(s2_tensor, self.s2_mean, self.s2_std)
        
        sim_tensor = sim_tensor[self.sim_to_real_idx]
        
        spatial_shape = tuple(sim_tensor.shape[-2:])
        cloud_tensor = self._read_tif(sample_paths["cloud"], is_mask=True, ref_shape=spatial_shape)
        wc_tensor = self._read_tif(sample_paths["worldcover"], is_mask=True, ref_shape=spatial_shape)
        
        transformed = apply_spatial_transforms(
            [sim_tensor, real_tensor, s2_tensor, cloud_tensor, wc_tensor],
            is_train=self.is_train,
            crop_size=self.crop_size
        )
        sim_tensor, real_tensor, s2_tensor, cloud_tensor, wc_tensor = transformed
        
        if cloud_tensor.shape[0] == 1: cloud_tensor = cloud_tensor.squeeze(0)
        if wc_tensor.shape[0] == 1: wc_tensor = wc_tensor.squeeze(0)
        
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
        test_ratio: float = 0.1,
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
        self.test_ratio = test_ratio
        self.fast_dev_run = fast_dev_run
        
        self.input_bands = PHISAT2_REAL_BANDS
        self.s2_bands = S2_BANDS

    def _get_pair_splits(self) -> tuple[list[str], list[str], list[str]]:
        pairs_dir = self.root_dir / "pairs"
        if not pairs_dir.exists():
            return [], [], []
            
        all_pairs = sorted([p.name for p in pairs_dir.iterdir() if p.is_dir() and p.name.startswith("pair_")])
        
        if not all_pairs:
            return [], [], []

        rng = np.random.default_rng(self.seed)
        shuffled_indices = rng.permutation(len(all_pairs))
        
        test_count = max(1, int(len(all_pairs) * self.test_ratio))
        val_count = max(1, int(len(all_pairs) * self.val_ratio))
        
        test_indices = set(shuffled_indices[:test_count])
        val_indices = set(shuffled_indices[test_count : test_count + val_count])
        
        train_pairs = [p for i, p in enumerate(all_pairs) if i not in test_indices and i not in val_indices]
        val_pairs = [p for i, p in enumerate(all_pairs) if i in val_indices]
        test_pairs = [p for i, p in enumerate(all_pairs) if i in test_indices]
        
        return train_pairs, val_pairs, test_pairs

    def setup(self, stage: str | None = None) -> None:
        max_patches = self.batch_size if self.fast_dev_run else None
        
        train_pairs, val_pairs, test_pairs = self._get_pair_splits()
        
        if not val_pairs or not test_pairs:
            train_pairs = val_pairs = test_pairs = [p.name for p in (self.root_dir / "pairs").iterdir() if p.is_dir()]
        
        if stage == "fit" or stage is None:
            self.train_dataset = TripletsDataset(
                self.root_dir, 
                split_pairs=train_pairs, 
                max_patches=max_patches,
                is_train=True,
                crop_size=224
            )
            self.val_dataset = TripletsDataset(
                self.root_dir, 
                split_pairs=val_pairs, 
                max_patches=max_patches,
                is_train=False,
                crop_size=224
            )
            
        if stage == "test" or stage is None:
            self.test_dataset = TripletsDataset(
                self.root_dir, 
                split_pairs=test_pairs, 
                max_patches=max_patches,
                is_train=False,
                crop_size=224
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

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers
        )