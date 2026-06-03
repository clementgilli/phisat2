from __future__ import annotations

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn

from phisat2.tasks import TaskSpec

class SSLPretrainModule(L.LightningModule):
    def __init__(self, model: nn.Module, spec: TaskSpec, *, lr: float) -> None:
        super().__init__()
        self.model = model
        self.spec = spec
        self.lr = lr
        
        self.save_hyperparameters({"task": spec.task, "dataset": spec.dataset, "lr": lr})

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self._shared_step(batch, "train")
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)

    def _shared_step(self, batch: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
        image = batch["image"]
        
        reconstruction = self(image)
        
        if reconstruction.ndim >= 3 and image.ndim >= 3 and reconstruction.shape[-2:] != image.shape[-2:]:
            reconstruction = F.interpolate(reconstruction, size=image.shape[-2:], mode="bilinear", align_corners=False)
            
        loss = F.mse_loss(reconstruction, image)
        
        self.log(f"{prefix}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss