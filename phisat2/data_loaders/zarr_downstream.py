from __future__ import annotations

import os
import time
from functools import lru_cache
from pathlib import Path

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from phisat2.data_loaders.transforms import normalize_sim_image
from phisat2.tasks import TaskSpec

ZARR_DATASET_NAMES = {
    "burned": ("burned_area", "burned"),
    "floods": ("worldfloods", "floods"),
    "lc": ("phileo-bench_lc", "lc", "lulc"),
    "lulc": ("phileo-bench_lc", "lulc"),
    "marine": ("marine_area", "marine"),
}

BAND_PERMUTATIONS = {
    "worldfloods": [0, 1, 2, 3, 4, 5, 6, 7],
    "burned_area": [0, 1, 2, 3, 4, 5, 6, 7],
    "lulc": [0, 1, 2, 3, 4, 5, 6, 7],
    "marine_area": [0, 1, 2, 3, 4, 5, 6, 7],
    "clouds": [0, 1, 2, 3, 4, 5, 6, 7],
}


class ZarrDownstreamDataset(Dataset):
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
    ) -> None:
        if spec.task != "segmentation":
            raise ValueError("zarr_downstream currently supports segmentation datasets.")

        self.spec = spec
        self.split = split
        self.seed = seed
        self.crop_size = crop_size
        self.random_crop = random_crop
        dataset_names = ZARR_DATASET_NAMES.get(spec.dataset, (spec.dataset,))
        self.permutation = BAND_PERMUTATIONS.get(dataset_names[0], list(range(8)))

        base_path = self._resolve_base_path(Path(root_dir), dataset_names)
        source_folder = base_path / "trainval" if split in {"train", "val"} else base_path / "test"
        if not source_folder.exists():
            raise FileNotFoundError(f"Expected Zarr split folder at {source_folder}")
        self.patches = self._list_patches(source_folder, split, seed, val_ratio, max_patches)

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        patch_path = Path(self.patches[index])
        image_array = self._open_array(patch_path / "img")
        mask_array = self._open_array(patch_path / "label")
        top, left, crop_h, crop_w = self._crop_window(
            image_array.shape[-2:],
            self.crop_size,
            train=self.split == "train" and self.random_crop,
        )
        image_selection = (slice(None), slice(top, top + crop_h), slice(left, left + crop_w))
        mask_selection = (..., slice(top, top + crop_h), slice(left, left + crop_w))
        image = torch.from_numpy(self._read_array(image_array, image_selection)).float()
        mask = torch.from_numpy(self._read_array(mask_array, mask_selection)).long()
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)
        image = normalize_sim_image(image[self.permutation])
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
    def _crop_window(shape: tuple[int, int], crop_size: int, *, train: bool) -> tuple[int, int, int, int]:
        height, width = shape
        crop_h = min(crop_size, height)
        crop_w = min(crop_size, width)
        if train and height > crop_h:
            top = int(torch.randint(0, height - crop_h + 1, (1,)).item())
        else:
            top = max(0, (height - crop_h) // 2)
        if train and width > crop_w:
            left = int(torch.randint(0, width - crop_w + 1, (1,)).item())
        else:
            left = max(0, (width - crop_w) // 2)
        return top, left, crop_h, crop_w

    @staticmethod
    def _list_patches(
        source_folder: Path,
        split: str,
        seed: int,
        val_ratio: float,
        max_patches: int | None,
    ) -> list[str]:
        patch_paths = list(_list_patch_dirs(str(source_folder)))
        if not patch_paths:
            raise FileNotFoundError(f"No Zarr patches found in {source_folder}")
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


class ZarrDownstreamDataModule(L.LightningDataModule):
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
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.spec = spec
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.crop_size = crop_size
        self.fast_dev_run = fast_dev_run

    def setup(self, stage: str | None = None) -> None:
        max_patches = self.batch_size if self.fast_dev_run else None
        if stage in {None, "fit", "validate"}:
            self.train_dataset = ZarrDownstreamDataset(
                self.root_dir,
                self.spec,
                split="train",
                seed=self.seed,
                crop_size=self.crop_size,
                max_patches=max_patches,
                random_crop=not self.fast_dev_run,
            )
            self.val_dataset = ZarrDownstreamDataset(
                self.root_dir,
                self.spec,
                split="val",
                seed=self.seed,
                crop_size=self.crop_size,
                max_patches=max_patches,
                random_crop=not self.fast_dev_run,
            )
        if stage in {None, "test"}:
            self.test_dataset = ZarrDownstreamDataset(
                self.root_dir,
                self.spec,
                split="test",
                seed=self.seed,
                crop_size=self.crop_size,
                max_patches=max_patches,
                random_crop=not self.fast_dev_run,
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
