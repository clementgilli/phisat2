from __future__ import annotations

import csv
import os
import time
from functools import lru_cache
from pathlib import Path

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from phisat2.tasks import TaskSpec
from phisat2.data_loaders.sensors import PHISAT2_REAL_BANDS, PHISAT2_SIM_BANDS, get_norm_tensors
from phisat2.data_loaders.transforms import apply_spatial_transforms, normalize_tensor

ZARR_DATASET_NAMES = {
    "burned":   ("burned_area_dataset", "burned"),
    "floods":   ("floods_dataset",      "floods"),
    "lulc":     ("phileo-bench_lc",     "lulc"),
    "clouds":   ("clouds_dataset",      "clouds"),
    "roads":    ("phileo-bench_roads",  "roads"),
    "building": ("phileo-bench_building", "building"),
    "router":     ("router_dataset",        "router"),
}

ZARR_SOURCE_BANDS = {
    "phileo-bench_lc":        ["BLUE", "GREEN", "RED", "PAN", "NIR_BROAD", "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3"],
    "burned_area_dataset":    ["BLUE", "GREEN", "RED", "PAN", "NIR_BROAD", "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3"],
    "floods_dataset":         ["BLUE", "GREEN", "RED", "PAN", "NIR_BROAD", "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3"],
    "clouds_dataset":         ["BLUE", "GREEN", "RED", "PAN", "NIR_BROAD", "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3"],
    "phileo-bench_roads":     ["BLUE", "GREEN", "RED", "PAN", "NIR_BROAD", "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3"],
    "phileo-bench_building":  ["BLUE", "GREEN", "RED", "PAN", "NIR_BROAD", "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3"],
    "router_dataset":         ["BLUE", "GREEN", "RED", "PAN", "NIR_BROAD", "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3"],
}

ZARR_SCALING_FACTORS = {
    "floods":   1.0,
    "lulc":     10000.0,
    "roads":    10000.0,
    "building": 10000.0,
    "router":   1.0,
    "burned":   1.0,
    "clouds":   10000.0,
}

# Clouds: original patches were 512×512, stored as 4096×4096 (factor 8).
#         A 2048×2048 window → downsample ×8 → true 256×256 patch.
ZARR_DOWNSAMPLE_FACTORS: dict[str, int] = {
    "clouds": 8,
}


class DownstreamDataset(Dataset):
    
    # Zarr is read in load_size × load_size windows.
    # crop_size ≤ load_size; apply_spatial_transforms handles the final crop.
    load_size: int = 256

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

        self.spec      = spec
        self.split     = split
        self.seed      = seed
        self.crop_size = crop_size
        self.is_train  = (split == "train" and random_crop)

        dataset_names = ZARR_DATASET_NAMES.get(spec.dataset, (spec.dataset,))
        base_name     = dataset_names[0]
        source_bands  = ZARR_SOURCE_BANDS.get(base_name)
        if source_bands is None:
            raise ValueError(
                f"Source bands for dataset '{base_name}' are not defined in ZARR_SOURCE_BANDS."
            )
        try:
            self.permutation = [source_bands.index(b) for b in PHISAT2_REAL_BANDS]
        except ValueError as exc:
            raise ValueError(
                f"Cannot map {source_bands} → {PHISAT2_REAL_BANDS}."
            ) from exc

        base_path     = self._resolve_base_path(Path(root_dir), dataset_names)
        source_folder = base_path / ("trainval" if split in {"train", "val"} else "test")
        if not source_folder.exists():
            raise FileNotFoundError(f"Expected Zarr split folder at {source_folder}")

        self.scaling_factor = ZARR_SCALING_FACTORS.get(spec.dataset, 1.0)
        self.downsample     = ZARR_DOWNSAMPLE_FACTORS.get(spec.dataset, 1)
        self.patches        = self._list_patches(
            source_folder, split, seed, val_ratio, max_patches, subset_csv=subset_csv
        )
        self.mean, self.std = get_norm_tensors("phisat2_sim", PHISAT2_REAL_BANDS)

        self.samples: list[tuple] = []
        read_size = self.load_size * self.downsample

        for patch_path in self.patches:
            patch_path = Path(patch_path)
            
            if patch_path.name.startswith(".") or not (patch_path / "img").exists():
                continue
                
            img_arr = self._open_array(patch_path / "img")
            H, W    = img_arr.shape[-2:]

            if H <= read_size and W <= read_size:
                self.samples.append((patch_path, 0, 0))

            elif self.is_train:
                self.samples.append((patch_path, None, H, W))

            else:
                n_h = H // read_size
                n_w = W // read_size
                for ty in range(n_h):
                    for tx in range(n_w):
                        self.samples.append(
                            (patch_path, ty * read_size, tx * read_size)
                        )

    # ── Length ────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    # ── Item loading ──────────────────────────────────────────────────────────

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample     = self.samples[index]
        patch_path = sample[0]

        # ── Determine crop window ─────────────────────────────────────────────
        read_size = self.load_size * self.downsample  # e.g. 2048 for clouds

        if len(sample) == 4:
            _, _, H, W = sample
            y_start = int(torch.randint(0, max(1, H - read_size + 1), (1,)).item())
            x_start = int(torch.randint(0, max(1, W - read_size + 1), (1,)).item())
        else:
            _, y_start, x_start = sample

        slice_y = slice(y_start, y_start + read_size)
        slice_x = slice(x_start, x_start + read_size)

        # ── Read image ────────────────────────────────────────────────────────
        image_array = self._open_array(patch_path / "img")
        image = torch.from_numpy(
            self._read_array(image_array, (slice(None), slice_y, slice_x))
        ).float()
        image = image[self.permutation] / self.scaling_factor

        if self.downsample > 1:
            image = torch.nn.functional.interpolate(
                image.unsqueeze(0),
                size=(self.load_size, self.load_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        image = normalize_tensor(image, self.mean, self.std)

        # ── Read label ────────────────────────────────────────────────────────
        target_array = self._open_array(patch_path / "label")
        target_is_scalar = (target_array.ndim == 0)

        if target_is_scalar:
            target = torch.tensor(target_array[...])
        else:
            tgt_slice = (
                (slice_y, slice_x)
                if target_array.ndim == 2
                else (slice(None), slice_y, slice_x)
            )
            target = torch.from_numpy(self._read_array(target_array, tgt_slice))

            if self.downsample > 1 and not target_is_scalar and target.ndim >= 2:
                t = target.unsqueeze(0) if target.ndim == 2 else target   # (1,H,W) or (C,H,W)
                t = torch.nn.functional.interpolate(
                    t.unsqueeze(0).float(),
                    size=(self.load_size, self.load_size),
                    mode="nearest",
                ).squeeze(0)
                target = (t.squeeze(0) if target.ndim == 2 else t).to(target.dtype)

        # ── Task-specific preprocessing ───────────────────────────────────────
        task = self.spec.task

        if task == "segmentation":
            if target.ndim == 3:
                target = (
                    target.argmax(0) if target.shape[0] > 1 else target.squeeze(0)
                )
            target = target.long()

            target_f  = target.unsqueeze(0).float()           # (1, H, W)
            out       = apply_spatial_transforms(
                [image, target_f], is_train=self.is_train, crop_size=self.crop_size
            )
            image     = out[0]
            target    = out[1].squeeze(0).long()               # (H, W)
            if self.spec.dataset == "clouds":
                target = target - 1

        elif task == "pixel_regression":
            if target.ndim == 2:
                target = target.unsqueeze(0)                   # (1, H, W)
            target = target.float()
            out    = apply_spatial_transforms(
                [image, target], is_train=self.is_train, crop_size=self.crop_size
            )
            image, target = out[0], out[1]

        else:
            
            target = (
                target.long() if task == "classification" else target.float()
            )
            out   = apply_spatial_transforms(
                [image], is_train=self.is_train, crop_size=self.crop_size
            )
            image = out[0]
            if not target_is_scalar and target.ndim > 1:
                target = target.squeeze()

        return {"image": image, self.spec.target_key: target}

    # ── Static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_base_path(root_dir: Path, dataset_names: tuple[str, ...]) -> Path:
        if root_dir.suffix == ".zarr":
            return root_dir
        for name in dataset_names:
            p = root_dir / f"{name}.zarr"
            if p.exists():
                return p
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
        last_err: OSError | None = None
        for attempt in range(3):
            try:
                return array[selection]
            except OSError as exc:
                last_err = exc
                time.sleep(0.5 * (attempt + 1))
        raise last_err  # type: ignore[misc]

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
            with open(subset_csv) as f:
                subset_ids = {row["sample_id"] for row in csv.DictReader(f)}

            if split == "train":
                selected = [p for p in patch_paths if Path(p).name in subset_ids]
                if not selected:
                    raise ValueError(
                        f"No matching patches for the IDs in {subset_csv}"
                    )
            else:   # val
                remaining   = [p for p in patch_paths if Path(p).name not in subset_ids]
                rng         = np.random.default_rng(seed)
                val_count   = max(1, int(len(patch_paths) * val_ratio))
                val_indices = set(
                    rng.choice(len(remaining), size=val_count, replace=False).tolist()
                )
                selected = [p for i, p in enumerate(remaining) if i in val_indices]

            return selected[:max_patches] if max_patches else selected

        if max_patches is not None:
            return patch_paths[:max_patches]
        if split not in {"train", "val"}:
            return patch_paths

        rng         = np.random.default_rng(seed)
        val_count   = max(1, int(len(patch_paths) * val_ratio))
        val_indices = set(
            rng.choice(len(patch_paths), size=val_count, replace=False).tolist()
        )
        if split == "val":
            return [p for i, p in enumerate(patch_paths) if i in val_indices]
        return [p for i, p in enumerate(patch_paths) if i not in val_indices]


@lru_cache(maxsize=16)
def _list_patch_dirs(source_folder: str) -> tuple[str, ...]:
    with os.scandir(source_folder) as entries:
        return tuple(sorted(e.path for e in entries if e.is_dir()))


# ─────────────────────────────────────────────────────────────────────────────
# DataModule
# ─────────────────────────────────────────────────────────────────────────────

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
        self.root_dir     = root_dir
        self.spec         = spec
        self.batch_size   = batch_size
        self.num_workers  = num_workers
        self.seed         = seed
        self.crop_size    = crop_size
        self.fast_dev_run = fast_dev_run
        self.subset_csv   = subset_csv
        self.input_bands  = PHISAT2_REAL_BANDS

    def setup(self, stage: str | None = None) -> None:
        max_p = self.batch_size if self.fast_dev_run else None

        if stage in {None, "fit", "validate"}:
            self.train_dataset = DownstreamDataset(
                self.root_dir, self.spec,
                split="train", seed=self.seed, crop_size=self.crop_size,
                max_patches=max_p, random_crop=not self.fast_dev_run,
                subset_csv=self.subset_csv,
            )
            self.val_dataset = DownstreamDataset(
                self.root_dir, self.spec,
                split="val", seed=self.seed, crop_size=self.crop_size,
                max_patches=max_p, random_crop=not self.fast_dev_run,
                subset_csv=self.subset_csv,
            )

        if stage in {None, "test"}:
            self.test_dataset = DownstreamDataset(
                self.root_dir, self.spec,
                split="test", seed=self.seed, crop_size=self.crop_size,
                max_patches=max_p, random_crop=False,
                subset_csv=None,    # always evaluate on the full test set
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