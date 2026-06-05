from __future__ import annotations

import csv
import os
import time
from functools import lru_cache
from pathlib import Path

import lightning as L
import numpy as np
import torch
from torch.utils.data import Dataset

from phisat2.tasks import TaskSpec
from phisat2.data_loaders.sensors import PHISAT2_REAL_BANDS, PHISAT2_SIM_BANDS, get_norm_tensors
from phisat2.data_loaders.transforms import apply_spatial_transforms, normalize_tensor

ZARR_DATASET_NAMES = {
    "burned": ("burned_area", "burned"),
    "floods": ("worldfloods", "floods"),
    "lc": ("phileo-bench_lc", "lc", "lulc"),
    "lulc": ("phileo-bench_lc", "lulc"),
    "marine": ("marine_area", "marine"),
}

# ORDER OF BANDS IN THE ZARR DATASETS (MAY VARY FROM THE ORDER IN THE TIFF FILES)
ZARR_SOURCE_BANDS = {
    "lulc":   ["BLUE", "GREEN", "RED", "PAN", "NIR_BROAD", "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3"],
    "burned": ["BLUE", "GREEN", "RED", "PAN", "NIR_BROAD", "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3"],
    "floods": ["BLUE", "GREEN", "RED", "PAN", "NIR_BROAD", "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3"],
    "marine": ["BLUE", "GREEN", "RED", "PAN", "NIR_BROAD", "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3"],
}

class DownstreamDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        spec: TaskSpec,
        *,
        split: str,
        seed: int,
        val_ratio: float = 0.1,
        crop_size: int = 224,
        max_patches: int | None = None,
        random_crop: bool = True,
        subset_csv: str | None = None,
    ) -> None:
        if spec.task != "segmentation":
            raise ValueError("zarr_downstream currently supports segmentation datasets.")

        self.spec = spec        
        self.split = split
        self.seed = seed
        self.crop_size = crop_size
        self.is_train = (split == "train" and random_crop)
        
        dataset_names = ZARR_DATASET_NAMES.get(spec.dataset, (spec.dataset,))
        base_name = dataset_names[0]
        source_bands = ZARR_SOURCE_BANDS.get(base_name)
        if source_bands is None:
            raise ValueError(f"Source bands for dataset '{base_name}' are not defined in ZARR_SOURCE_BANDS.")
        try:
            self.permutation = [source_bands.index(band) for band in PHISAT2_REAL_BANDS]
        except ValueError as e:
            raise ValueError(f"Cannot map {source_bands} to {PHISAT2_REAL_BANDS}.") from e

        base_path = self._resolve_base_path(Path(root_dir), dataset_names)
        source_folder = base_path / "trainval" if split in {"train", "val"} else base_path / "test"
        if not source_folder.exists():
            raise FileNotFoundError(f"Expected Zarr split folder at {source_folder}")
            
        self.patches = self._list_patches(source_folder, split, seed, val_ratio, max_patches, subset_csv=subset_csv)
        
        self.mean, self.std = get_norm_tensors("phisat2_sim", PHISAT2_REAL_BANDS) # Using sim stats but real bands order (since we will permute the bands to match the real order)

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        patch_path = Path(self.patches[index])
        
        image_array = self._open_array(patch_path / "img")
        mask_array = self._open_array(patch_path / "label")
        
        image = torch.from_numpy(self._read_array(image_array, slice(None))).float()
        mask = torch.from_numpy(self._read_array(mask_array, slice(None))).long()
        
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        image = image[self.permutation]
        
        image = normalize_tensor(image, self.mean, self.std)

        transformed = apply_spatial_transforms([image, mask], is_train=self.is_train, crop_size=self.crop_size)
        image, mask = transformed[0], transformed[1]
        
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)

        return {"image": image, "mask": mask}

    @staticmethod
    def _resolve_base_path(root_dir: Path, dataset_names: tuple[str, ...]) -> Path:
        if root_dir.suffix == ".zarr":
            return root_dir
        for dataset_name in dataset_names:
            base_path = root_dir / f"{dataset_name}.zarr"
            if base_path.exists():
                return base_path
        return root_dir / f"{dataset_names[0]}.zarr"

    @staticmethod
    def _open_array(array_path: Path):
        import zarr

        try:
            return zarr.open_array(array_path, mode="r", zarr_format=3)
        except (FileNotFoundError, ValueError):
            return zarr.open_array(array_path, mode="r")

    @staticmethod
    def _read_array(array, selection) -> np.ndarray:
        last_error: OSError | None = None
        for attempt in range(3):
            try:
                return array[selection]
            except OSError as exc:
                last_error = exc
                time.sleep(0.5 * (attempt + 1))
        assert last_error is not None
        raise last_error

    @staticmethod
    def _list_patches(
        source_folder: Path,
        split: str,
        seed: int,
        val_ratio: float,
        max_patches: int | None,
        subset_csv: str | None = None,
    ) -> list[str]:
        patch_paths = list(_list_patch_dirs(str(source_folder)))
        if not patch_paths:
            raise FileNotFoundError(f"No Zarr patches found in {source_folder}")
            
        if subset_csv and split in {"train", "val"}:
            with open(subset_csv, 'r') as f:
                reader = csv.DictReader(f)
                subset_ids = {row['sample_id'] for row in reader}
                
            if split == "train":
                selected = [p for p in patch_paths if Path(p).name in subset_ids]
                if not selected:
                    raise ValueError(f"No matching patches found for the IDs in {subset_csv}")
            elif split == "val":
                remaining = [p for p in patch_paths if Path(p).name not in subset_ids]
                rng = np.random.default_rng(seed)
                val_count = max(1, int(len(patch_paths) * val_ratio))
                val_indices = set(rng.choice(len(remaining), size=val_count, replace=False).tolist())
                selected = [path for index, path in enumerate(remaining) if index in val_indices]
                
            if max_patches is not None:
                return selected[:max_patches]
            return selected

        if max_patches is not None:
            return patch_paths[:max_patches]
        if split not in {"train", "val"}:
            return patch_paths
            
        rng = np.random.default_rng(seed)
        val_count = max(1, int(len(patch_paths) * val_ratio))
        val_indices = set(rng.choice(len(patch_paths), size=val_count, replace=False).tolist())
        
        if split == "val":
            return [path for index, path in enumerate(patch_paths) if index in val_indices]
        return [path for index, path in enumerate(patch_paths) if index not in val_indices]


@lru_cache(maxsize=16)
def _list_patch_dirs(source_folder: str) -> tuple[str, ...]:
    with os.scandir(source_folder) as entries:
        return tuple(sorted(entry.path for entry in entries if entry.is_dir()))


class DownstreamDataModule(L.LightningDataModule):
    def __init__(
        self,
        root_dir: str | Path,
        spec: TaskSpec,
        *,
        batch_size: int,
        num_workers: int,
        seed: int,
        crop_size: int = 224,
        fast_dev_run: bool = False,
        subset_csv: str | None = None,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.spec = spec
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.crop_size = crop_size
        self.fast_dev_run = fast_dev_run
        self.subset_csv = subset_csv
        self.input_bands = PHISAT2_REAL_BANDS
        
    def setup(self, stage: str | None = None) -> None:
        max_patches = self.batch_size if self.fast_dev_run else None
        if stage in {None, "fit", "validate"}:
            self.train_dataset = DownstreamDataset(
                self.root_dir,
                self.spec,
                split="train",
                seed=self.seed,
                crop_size=self.crop_size,
                max_patches=max_patches,
                random_crop=not self.fast_dev_run,
                subset_csv=self.subset_csv,
            )
            self.val_dataset = DownstreamDataset(
                self.root_dir,
                self.spec,
                split="val",
                seed=self.seed,
                crop_size=self.crop_size,
                max_patches=max_patches,
                random_crop=not self.fast_dev_run,
                subset_csv=self.subset_csv,
            )
        if stage in {None, "test"}:
            self.test_dataset = DownstreamDataset(
                self.root_dir,
                self.spec,
                split="test",
                seed=self.seed,
                crop_size=self.crop_size,
                max_patches=max_patches,
                random_crop=not self.fast_dev_run,
                subset_csv=None,
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
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
