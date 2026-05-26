from __future__ import annotations

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn

from phisat2.tasks import TaskSpec


class PhiSat2LightningModule(L.LightningModule):
    def __init__(self, model: nn.Module, spec: TaskSpec, *, lr: float) -> None:
        super().__init__()
        self.model = model
        self.spec = spec
        self.lr = lr
        self.save_hyperparameters({"task": spec.task, "dataset": spec.dataset, "lr": lr})
        self._freeze_encoder()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if self.training:
            encoder = getattr(self.model, "encoder", None)
            adapter = getattr(self.model, "adapter", None)
            head = getattr(self.model, "head", None)
            if encoder is not None and adapter is not None and head is not None:
                self._freeze_encoder()
                with torch.no_grad():
                    features = encoder(image)
                pyramid = adapter(features, image.shape[-2:])
                return head(pyramid)
        return self.model(image)

    def train(self, mode: bool = True):
        module = super().train(mode)
        self._freeze_encoder()
        return module

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self._shared_step(batch, "train")
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        trainable_params = [param for param in self.parameters() if param.requires_grad]
        return torch.optim.AdamW(trainable_params, lr=self.lr, weight_decay=1e-4)

    def _freeze_encoder(self) -> None:
        encoder = getattr(self.model, "encoder", None)
        if encoder is None:
            return
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False

    def _shared_step(self, batch: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
        image = batch["image"]
        prediction = self(image)
        target = batch[self.spec.target_key]
        loss = self._loss(prediction, target)
        self.log(f"{prefix}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def _loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.spec.task == "segmentation":
            if prediction.shape[-2:] != target.shape[-2:]:
                prediction = F.interpolate(prediction, size=target.shape[-2:], mode="bilinear", align_corners=False)
            return F.cross_entropy(prediction, target.long())
        if self.spec.task == "classification":
            return F.cross_entropy(prediction, target.long())
        if prediction.ndim == 4 and target.ndim == 4 and prediction.shape[-2:] != target.shape[-2:]:
            prediction = F.interpolate(prediction, size=target.shape[-2:], mode="bilinear", align_corners=False)
        return F.mse_loss(prediction, target.float())
