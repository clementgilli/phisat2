from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import lightning as L

from phisat2.data_loaders.h5_pairs import H5PairsDataModule
from phisat2.data_loaders.synthetic import SyntheticDataModule
from phisat2.data_loaders.zarr_downstream import ZarrDownstreamDataModule
from phisat2.tasks import TaskSpec

DataModuleBuilder = Callable[..., L.LightningDataModule]


@dataclass(frozen=True)
class DataLoaderEntry:
    name: str
    description: str
    builder: DataModuleBuilder


REGISTRY: dict[str, DataLoaderEntry] = {
    "zarr_downstream": DataLoaderEntry(
        "zarr_downstream",
        "Zarr downstream segmentation datasets with image/mask batches.",
        ZarrDownstreamDataModule,
    ),
    "h5_pairs": DataLoaderEntry(
        "h5_pairs",
        "Paired PhiSat-2/Sentinel HDF5 dataset for climate, geolocation, and reconstruction.",
        H5PairsDataModule,
    ),
    "synthetic": DataLoaderEntry(
        "synthetic",
        "Small random dataloader used by smoke tests and CI.",
        SyntheticDataModule,
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
) -> L.LightningDataModule:
    try:
        entry = REGISTRY[name]
    except KeyError as exc:
        valid = ", ".join(sorted(REGISTRY))
        raise ValueError(f"Unknown dataloader '{name}'. Expected one of: {valid}.") from exc
    if name == "synthetic":
        return entry.builder(spec=spec, batch_size=batch_size, num_workers=num_workers, seed=seed)
    return entry.builder(root_dir=root_dir, spec=spec, batch_size=batch_size, num_workers=num_workers, seed=seed)
