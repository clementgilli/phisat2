from __future__ import annotations

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from phisat2.tasks import TaskSpec

def sliced_wasserstein_distance(
    x: torch.Tensor, 
    y: torch.Tensor, 
    num_projections: int = 128,
    dense_alignment: bool = False,
    p: int = 2
) -> torch.Tensor:
    """
    Computes the Sliced Wasserstein-p Distance between two CNN feature maps.
    Expects inputs strictly of shape (B, C, H, W).
    """
    x_flat = x.mean(dim=[2, 3])
    y_flat = y.mean(dim=[2, 3])

    dim = x_flat.size(1)
    
    projections = torch.randn(dim, num_projections, device=x.device)
    projections = F.normalize(projections, p=2, dim=0)
    
    x_proj  = (x_flat @ projections).sort(dim=0).values
    y_proj  = (y_flat @ projections).sort(dim=0).values
    
    if p == 1:
        return F.l1_loss(x_proj, y_proj)
    elif p == 2:
        return F.mse_loss(x_proj, y_proj)
    else:
        return torch.mean(torch.abs(x_proj - y_proj) ** p) ** (1.0 / p)

class DomainAdaptationModule(L.LightningModule):
    
    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        spec: TaskSpec,
        *,
        lr: float,
        weight_decay: float,
        loss_weights: dict[str, float] | None = None,
        lambda_swd: float = 0.0,
    ) -> None:
        super().__init__()
        self.student = student_model
        self.teacher = teacher_model
        self.spec = spec
        self.lr = lr
        self.weight_decay = weight_decay
        self.lambda_swd = lambda_swd

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
                    layer_mse = F.mse_loss(feat_real[i], feat_sim[i])
                    layer_swd = sliced_wasserstein_distance(feat_real[i], feat_sim[i])
                    layer_loss = layer_mse + self.lambda_swd * layer_swd
                    losses.append(layer_loss * weight)
                    
                    self.log_dict({
                        f"{prefix}_{layer_name}_mse": layer_mse,
                        f"{prefix}_{layer_name}_swd": layer_swd,
                        f"{prefix}_{layer_name}_total_loss": layer_loss,
                    }, on_step=False, on_epoch=True, sync_dist=True)

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
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.trainer.max_epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}