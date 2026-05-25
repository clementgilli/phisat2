from __future__ import annotations

from dataclasses import dataclass

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset

from phisat2.tasks import TaskSpec


@dataclass(frozen=True)
class SyntheticSettings:
    length: int = 12
    image_size: int = 32
    channels: int = 8


class SyntheticDataset(Dataset):
    def __init__(self, spec: TaskSpec, settings: SyntheticSettings, seed: int) -> None:
        self.spec = spec
        self.settings = settings
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return self.settings.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image = torch.rand(
            self.settings.channels,
            self.settings.image_size,
            self.settings.image_size,
            generator=self.generator,
        )
        batch: dict[str, torch.Tensor] = {"image": image}
        if self.spec.task == "segmentation":
            batch["mask"] = torch.randint(
                0,
                self.spec.num_outputs,
                (self.settings.image_size, self.settings.image_size),
                generator=self.generator,
            )
        elif self.spec.task == "classification":
            batch["label"] = torch.randint(0, self.spec.num_outputs, (), generator=self.generator)
        elif self.spec.task == "global_regression":
            batch["target"] = torch.rand(self.spec.num_outputs, generator=self.generator)
        else:
            batch["target"] = torch.rand(
                self.spec.num_outputs,
                self.settings.image_size,
                self.settings.image_size,
                generator=self.generator,
            )
        return batch


class SyntheticDataModule(L.LightningDataModule):
    def __init__(
        self,
        spec: TaskSpec,
        *,
        batch_size: int,
        num_workers: int,
        seed: int,
        image_size: int = 32,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.settings = SyntheticSettings(image_size=image_size)

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = SyntheticDataset(self.spec, self.settings, self.seed)
        self.val_dataset = SyntheticDataset(self.spec, self.settings, self.seed + 1)
        self.test_dataset = SyntheticDataset(self.spec, self.settings, self.seed + 2)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
