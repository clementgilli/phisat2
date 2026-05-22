from __future__ import annotations

from pathlib import Path

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from phisat2.data_loaders.transforms import crop_pair, normalize_sim_image
from phisat2.tasks import TaskSpec

ZARR_DATASET_NAMES = {
    "burned": "burned_area",
    "floods": "worldfloods",
    "marine": "marine_area",
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
    ) -> None:
        if spec.task != "segmentation":
            raise ValueError("zarr_downstream currently supports segmentation datasets.")

        self.spec = spec
        self.split = split
        self.seed = seed
        self.crop_size = crop_size
        dataset_name = ZARR_DATASET_NAMES.get(spec.dataset, spec.dataset)
        self.permutation = BAND_PERMUTATIONS.get(dataset_name, list(range(8)))

        base_path = Path(root_dir) / f"{dataset_name}.zarr"
        source_folder = base_path / "trainval" if split in {"train", "val"} else base_path / "test"
        if not source_folder.exists():
            raise FileNotFoundError(f"Expected Zarr split folder at {source_folder}")
        self.patches = self._list_patches(source_folder, split, seed, val_ratio)

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        import zarr

        group = zarr.open(self.patches[index], mode="r")
        image = torch.from_numpy(group["img"][:]).float()
        mask = torch.from_numpy(group["label"][:]).long()
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)
        image = normalize_sim_image(image[self.permutation])
        image, mask = crop_pair(image, mask, self.crop_size, train=self.split == "train")
        return {"image": image, "mask": mask}

    @staticmethod
    def _list_patches(source_folder: Path, split: str, seed: int, val_ratio: float) -> list[str]:
        patch_paths = sorted(str(path) for path in source_folder.iterdir() if path.is_dir())
        if not patch_paths:
            raise FileNotFoundError(f"No Zarr patches found in {source_folder}")
        if split not in {"train", "val"}:
            return patch_paths
        rng = np.random.default_rng(seed)
        shuffled = patch_paths[:]
        rng.shuffle(shuffled)
        val_count = max(1, int(len(shuffled) * val_ratio))
        return shuffled[:val_count] if split == "val" else shuffled[val_count:]


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
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.spec = spec
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.crop_size = crop_size

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = ZarrDownstreamDataset(
            self.root_dir, self.spec, split="train", seed=self.seed, crop_size=self.crop_size
        )
        self.val_dataset = ZarrDownstreamDataset(
            self.root_dir, self.spec, split="val", seed=self.seed, crop_size=self.crop_size
        )
        self.test_dataset = ZarrDownstreamDataset(
            self.root_dir, self.spec, split="test", seed=self.seed, crop_size=self.crop_size
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
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
