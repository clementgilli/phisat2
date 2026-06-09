from __future__ import annotations

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from phisat2.tasks import TaskSpec

class DomainAdaptationModule(L.LightningModule):
    
    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        spec: TaskSpec,
        *,
        lr: float,
        loss_weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.student = student_model
        self.teacher = teacher_model
        self.spec = spec
        self.lr = lr

        self.loss_weights = loss_weights or {
            "enc_0": 1.0,
            "enc_1": 1.0,
            "enc_2": 1.0,
            "bottleneck": 1.0
        }

        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.save_hyperparameters(ignore=["student_model", "teacher_model"])

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        return self

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.student(image)

    def _shared_step(self, batch: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
        img_real = batch["real"]
        img_sim  = batch["simulated"]

        with torch.no_grad():
            feat_sim = self.teacher(img_sim)

        feat_real = self.student(img_real)

        losses = []
        
        if isinstance(feat_sim, list) and isinstance(feat_real, list):
            for i, (layer_name, weight) in enumerate(self.loss_weights.items()):
                if i < len(feat_sim) and i < len(feat_real):
                    layer_loss = F.mse_loss(feat_real[i], feat_sim[i])
                    losses.append(layer_loss * weight)
                    
                    self.log(
                        f"{prefix}_{layer_name}_loss", 
                        layer_loss, 
                        on_step=False, 
                        on_epoch=True, 
                        sync_dist=True
                    )

        if not losses:
            actual_type = type(feat_sim)
            actual_len = len(feat_sim) if isinstance(feat_sim, (list, dict)) else "N/A"
            raise RuntimeError(
                f"CRITICAL ERROR: Could not map features for Domain Adaptation!\n"
                f"Expected mapping: {list(self.loss_weights.keys())}\n"
                f"Actual encoder output type: {actual_type} (length: {actual_len})"
            )

        total_loss = sum(losses)

        self.log(
            f"{prefix}_loss",
            total_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=img_real.shape[0]
        )

        return total_loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.student.parameters(), 
            lr=self.lr, 
            weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.trainer.max_epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}