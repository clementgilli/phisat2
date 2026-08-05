from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import lightning as L

from phisat2.data_loaders.triplets import TripletsDataModule
from phisat2.data_loaders.synthetic import SyntheticDataModule
from phisat2.data_loaders.downstream import DownstreamDataModule
from phisat2.data_loaders.eurosat import EuroSatDataModule
from phisat2.tasks import TaskSpec

DataModuleBuilder = Callable[..., L.LightningDataModule]

@dataclass(frozen=True)
class DataLoaderEntry:
    name: str
    description: str
    builder: DataModuleBuilder


REGISTRY: dict[str, DataLoaderEntry] = {
    "downstream": DataLoaderEntry(
        "downstream",
        "Downstream segmentation datasets with image/mask batches.",
        DownstreamDataModule,
        
    ),
    "triplets": DataLoaderEntry(
        "triplets",
        "Triplet datasets for training and evaluation.",
        TripletsDataModule,
    ),
    "synthetic": DataLoaderEntry(
        "synthetic",
        "Small random dataloader used by smoke tests and CI.",
        SyntheticDataModule,
    ),
    "eurosat": DataLoaderEntry(
        "eurosat",
        "Full EuroSAT simulated dataset for 1-NN feature extraction.",
        EuroSatDataModule,
    ),
}


def list_dataloaders() -> list[DataLoaderEntry]:
    return [REGISTRY[name] for name in sorted(REGISTRY)]


def build_datamodule(
    name: str,
    *,
    root_dir: str | Path,
    spec: TaskSpec,
    batch_size: int,
    num_workers: int,
    seed: int,
    crop_size: int = 224,
    fast_dev_run: bool = False,
    subset_csv: str | None = None,
) -> L.LightningDataModule:
    try:
        entry = REGISTRY[name]
    except KeyError as exc:
        valid = ", ".join(sorted(REGISTRY))
        raise ValueError(f"Unknown dataloader '{name}'. Expected one of: {valid}.") from exc
        
    if name == "synthetic":
        return entry.builder(spec=spec, batch_size=batch_size, num_workers=num_workers, seed=seed)
        
    if name == "downstream":
        return entry.builder(
            root_dir=root_dir,
            spec=spec,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            crop_size=crop_size,
            fast_dev_run=fast_dev_run,
            subset_csv=subset_csv,
        )
        
    if name == "eurosat":
        return entry.builder(
            root_dir=root_dir,
            spec=spec,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
        )
        
    return entry.builder(
        root_dir=root_dir,
        spec=spec,
        csv_dir=Path(root_dir) / "splits",
        batch_size=batch_size,
        num_workers=num_workers,
        fast_dev_run=fast_dev_run,
        crop_size=crop_size,
    )