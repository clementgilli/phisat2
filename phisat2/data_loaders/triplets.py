from __future__ import annotations

import os
import csv
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
from phisat2.data_loaders.transforms import apply_spatial_transforms, normalize_tensor, extract_phisat2_bands

class TripletsDataset(Dataset):
    def __init__(
        self, 
        root_dir: str | Path, 
        split_csv: Path | None = None,
        max_patches: int | None = None,
        is_train: bool = True,
        crop_size: int = 224
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.pairs_dir = self.root_dir / "pairs"
        self.samples = self._load_from_csv(split_csv)
        self.is_train = is_train
        self.crop_size = crop_size
        
        if max_patches is not None:
            self.samples = self.samples[:max_patches]
            
        self.sim_mean, self.sim_std = get_norm_tensors("phisat2_sim", PHISAT2_SIM_BANDS)
        self.real_mean, self.real_std = get_norm_tensors("phisat2_real", PHISAT2_REAL_BANDS)
        self.s2_mean, self.s2_std = get_norm_tensors("s2", PHISAT2_REAL_BANDS)
        self.real_L0_mean, self.real_L0_std = get_norm_tensors("phisat2_l0", PHISAT2_REAL_BANDS)
        self.sim_to_real_idx = [PHISAT2_SIM_BANDS.index(b) for b in PHISAT2_REAL_BANDS]

    def _load_from_csv(self, split_csv: Path | None) -> list[dict[str, Path | str]]:
        samples = []
        if split_csv is None or not split_csv.exists():
            warnings.warn(f"CSV {split_csv} does not exist.")
            return samples

        with split_csv.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                tile_id = row["tile_id"]
                
                parts = tile_id.split("_")
                tile_name = f"{parts[-2]}_{parts[-1]}"
                pair_id = "_".join(parts[:-2])
                
                base_tiles = self.pairs_dir / pair_id / "lightglue_coregistration"
                
                samples.append({
                    "simulated": base_tiles / "tiles" / "simulated_phisat2" / tile_name,
                    "real": base_tiles / "tiles" / "phisat2" / tile_name,
                    "sentinel2": base_tiles / "tiles" / "sentinel2" / tile_name,
                    "real_L0": base_tiles / "tiles" / "phisat2_L0" / tile_name,
                    "cloud": base_tiles / "masks" / "phisat2_cloud" / tile_name,
                    "worldcover": base_tiles / "masks" / "worldcover" / tile_name,
                    "tile_id": tile_id
                })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _read_tif(self, path: Path | None, is_mask: bool = False, ref_shape: tuple[int, int] = (256, 256)) -> torch.Tensor:
        if path is None or not path.exists():
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
        real_L0_tensor = self._read_tif(sample_paths["real_L0"])
        
        real_tensor = real_tensor[:8, :, :]
        s2_tensor = s2_tensor - 1000
        
        s2_tensor = extract_phisat2_bands(s2_tensor)
        
        sim_tensor = normalize_tensor(sim_tensor, self.sim_mean, self.sim_std)
        real_tensor = normalize_tensor(real_tensor, self.real_mean, self.real_std)
        s2_tensor = normalize_tensor(s2_tensor, self.s2_mean, self.s2_std)
        real_L0_tensor = normalize_tensor(real_L0_tensor, self.real_L0_mean, self.real_L0_std)
        
        sim_tensor = sim_tensor[self.sim_to_real_idx]
        
        spatial_shape = tuple(sim_tensor.shape[-2:])
        cloud_tensor = self._read_tif(sample_paths["cloud"], is_mask=True, ref_shape=spatial_shape)
        wc_tensor = self._read_tif(sample_paths["worldcover"], is_mask=True, ref_shape=spatial_shape)
        
        transformed = apply_spatial_transforms(
            [sim_tensor, real_tensor, s2_tensor, real_L0_tensor, cloud_tensor, wc_tensor],
            is_train=self.is_train,
            crop_size=self.crop_size
        )
        sim_tensor, real_tensor, s2_tensor, real_L0_tensor, cloud_tensor, wc_tensor = transformed
        
        if cloud_tensor.shape[0] == 1: cloud_tensor = cloud_tensor.squeeze(0)
        if wc_tensor.shape[0] == 1: wc_tensor = wc_tensor.squeeze(0)
        
        wc_mapping = {10: 0, 20: 1, 30: 2, 40: 3, 50: 4, 60: 5, 70: 6, 80: 7, 90: 8, 95: 9, 100: 10}
        mapped_wc = torch.zeros_like(wc_tensor)
        for old_val, new_val in wc_mapping.items():
            mapped_wc[wc_tensor == old_val] = new_val
        wc_tensor = mapped_wc
        
        return {
            "simulated": sim_tensor,       # (8, 224, 224) @ 4.75m
            "real": real_tensor,           # (8, 224, 224) @ 4.75m
            "sentinel2": s2_tensor,        # (13, 106, 106) @ 10m
            "real_L0": real_L0_tensor,     # (8, 224, 224) @ 4.75m
            "mask_cloud": cloud_tensor,
            "mask_worldcover": wc_tensor,
            "tile_id": sample_paths["tile_id"]
        }


class TripletsDataModule(L.LightningDataModule):
    def __init__(
        self,
        root_dir: str | Path,
        spec: TaskSpec,             
        csv_dir: str | Path | None = None,
        *,
        batch_size: int,
        num_workers: int = 4,
        fast_dev_run: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.spec = spec
        self.csv_dir = Path(csv_dir) if csv_dir else self.root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fast_dev_run = fast_dev_run
        
        self.input_bands = PHISAT2_REAL_BANDS
        self.s2_bands = S2_BANDS

    def setup(self, stage: str | None = None) -> None:
        max_patches = self.batch_size if self.fast_dev_run else None
        
        train_csv = self.csv_dir / "train.csv"
        val_csv = self.csv_dir / "val.csv"
        test_csv = self.csv_dir / "test.csv"
        
        if stage == "fit" or stage is None:
            self.train_dataset = TripletsDataset(
                self.root_dir, 
                split_csv=train_csv, 
                max_patches=max_patches,
                is_train=True,
                crop_size=224
            )
            self.val_dataset = TripletsDataset(
                self.root_dir, 
                split_csv=val_csv, 
                max_patches=max_patches,
                is_train=False,
                crop_size=224
            )
            
        if stage == "test" or stage is None:
            self.test_dataset = TripletsDataset(
                self.root_dir, 
                split_csv=test_csv, 
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