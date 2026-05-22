from __future__ import annotations

from pathlib import Path

import h5py
import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from phisat2.data_loaders.transforms import crop_pair, normalize_sim_image
from phisat2.tasks import TaskSpec

BAD_PRODUCT_IDS = {
    1296,
    1342,
    1385,
    1397,
    1420,
    1460,
    1497,
    1647,
    1854,
    2223,
    2246,
    2259,
    2373,
    2631,
    2640,
    2743,
    2834,
    2853,
    3374,
    3619,
    4071,
    4693,
    4813,
    4942,
    2352,
    2882,
    3322,
    3914,
    4702,
    1333,
    1466,
    1615,
    2460,
    2729,
    2763,
}


class H5PairsDataset(Dataset):
    def __init__(
        self,
        h5_path: str | Path,
        spec: TaskSpec,
        indices: np.ndarray,
        *,
        split: str,
        crop_size: int,
    ) -> None:
        self.h5_path = Path(h5_path)
        self.spec = spec
        self.indices = indices
        self.split = split
        self.crop_size = crop_size

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        h5_index = int(self.indices[index])
        with h5py.File(self.h5_path, "r") as handle:
            image = torch.from_numpy(handle["sim/images"][h5_index].astype(np.float32))
            batch = {"image": normalize_sim_image(image)}
            if self.spec.task == "classification":
                label = int(handle["metadata/koppen_zone"][h5_index])
                batch["label"] = torch.tensor(label, dtype=torch.long)
            elif self.spec.task == "global_regression":
                lat = float(handle["metadata/center_lat"][h5_index])
                lon = float(handle["metadata/center_lon"][h5_index])
                batch["target"] = encode_geolocation(lat, lon)
            elif self.spec.task == "pixel_regression":
                target = torch.from_numpy(handle["sim/images"][h5_index].astype(np.float32))
                batch["target"] = normalize_sim_image(target)
            else:
                raise ValueError("h5_pairs does not provide segmentation masks.")

        if self.spec.task == "pixel_regression":
            batch["image"], batch["target"] = crop_pair(
                batch["image"], batch["target"], self.crop_size, train=self.split == "train"
            )
        elif self.spec.task in {"classification", "global_regression"}:
            dummy = torch.empty(0)
            batch["image"], _ = crop_pair(batch["image"], dummy, self.crop_size, train=self.split == "train")
        return batch


class H5PairsDataModule(L.LightningDataModule):
    def __init__(
        self,
        root_dir: str | Path,
        spec: TaskSpec,
        *,
        batch_size: int,
        num_workers: int,
        seed: int,
        crop_size: int = 224,
    ) -> None:
        super().__init__()
        self.h5_path = resolve_h5_path(root_dir)
        self.spec = spec
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.crop_size = crop_size

    def setup(self, stage: str | None = None) -> None:
        train_indices, val_indices, test_indices = split_h5_indices(self.h5_path, self.seed)
        self.train_dataset = H5PairsDataset(
            self.h5_path, self.spec, train_indices, split="train", crop_size=self.crop_size
        )
        self.val_dataset = H5PairsDataset(self.h5_path, self.spec, val_indices, split="val", crop_size=self.crop_size)
        self.test_dataset = H5PairsDataset(
            self.h5_path, self.spec, test_indices, split="test", crop_size=self.crop_size
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


def resolve_h5_path(root_dir: str | Path) -> Path:
    root = Path(root_dir)
    if root.is_file():
        return root
    candidates = [
        root / "phisat2_s2b_dataset_v1.h5",
        root / "phisat2_s2b_dataset.h5",
        root / "dataset.h5",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find an HDF5 dataset under {root}")


def split_h5_indices(h5_path: Path, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as handle:
        total = handle["sim/images"].shape[0]
        if "metadata/product_id" in handle:
            product_ids = handle["metadata/product_id"][:]
            indices = np.where(~np.isin(product_ids, list(BAD_PRODUCT_IDS)))[0]
        else:
            indices = np.arange(total)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    train_end = int(0.7 * len(indices))
    val_end = train_end + int(0.15 * len(indices))
    return indices[:train_end], indices[train_end:val_end], indices[val_end:]


def encode_geolocation(lat: float, lon: float) -> torch.Tensor:
    lat_tensor = torch.tensor(lat / 90.0, dtype=torch.float32)
    lon_tensor = torch.tensor(lon / 180.0, dtype=torch.float32)
    return torch.stack(
        [
            torch.sin(torch.pi * lat_tensor),
            torch.cos(torch.pi * lat_tensor),
            torch.sin(torch.pi * lon_tensor),
            torch.cos(torch.pi * lon_tensor),
        ]
    )
