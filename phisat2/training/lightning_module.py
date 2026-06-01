from __future__ import annotations

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn

from phisat2.tasks import TaskSpec
from phisat2.evaluation.metrics import build_metrics


class PhiSat2LightningModule(L.LightningModule):
    def __init__(self, model: nn.Module, spec: TaskSpec, *, lr: float) -> None:
        super().__init__()
        self.model = model
        self.spec = spec
        self.lr = lr
        self.save_hyperparameters({"task": spec.task, "dataset": spec.dataset, "lr": lr})
        self.val_metrics = build_metrics(spec, prefix="val")
        self.test_metrics = build_metrics(spec, prefix="test")
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
        loss, _, _ = self._shared_step(batch, "train")
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        loss, preds, targets = self._shared_step(batch, "val")
        self.val_metrics.update(preds, targets)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        loss, preds, targets = self._shared_step(batch, "test")
        self.test_metrics.update(preds, targets)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, prog_bar=True)

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

    def _shared_step(self, batch: dict[str, torch.Tensor], prefix: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image = batch["image"]
        prediction = self(image)
        target = batch[self.spec.target_key]
        
        if prediction.ndim >= 3 and target.ndim >= 3 and prediction.shape[-2:] != target.shape[-2:]:
            prediction = F.interpolate(prediction, size=target.shape[-2:], mode="bilinear", align_corners=False)
            
        loss = self._loss(prediction, target)
        self.log(f"{prefix}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss, prediction, target

    def _loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.spec.task in ["segmentation", "classification"]:
            return F.cross_entropy(prediction, target.long())
        return F.mse_loss(prediction, target.float())