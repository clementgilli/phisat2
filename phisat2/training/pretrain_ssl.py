from __future__ import annotations

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn

from phisat2.tasks import TaskSpec

class SSLPretrainModule(L.LightningModule):
    def __init__(
        self, 
        model: nn.Module, 
        spec: TaskSpec, 
        *, 
        lr: float, 
        patch_size: int = 32, 
        mask_ratio: float = 0.6
    ) -> None:
        super().__init__()
        self.model = model
        self.spec = spec
        self.lr = lr
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        
        self.mask_token = nn.Parameter(torch.zeros(1, spec.num_outputs, 1, 1))
        
        self.save_hyperparameters({"task": spec.task, "dataset": spec.dataset, "lr": lr})

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image)

    def generate_mask(self, batch_size: int, h: int, w: int, device: torch.device) -> torch.Tensor:
        grid_h, grid_w = h // self.patch_size, w // self.patch_size
        num_patches = grid_h * grid_w
        num_masked = int(num_patches * self.mask_ratio)
        
        noise = torch.rand(batch_size, num_patches, device=device)
        indices = torch.argsort(noise, dim=1)
        
        binary_mask = torch.zeros(batch_size, num_patches, device=device)
        for i in range(batch_size):
            binary_mask[i, indices[i, :num_masked]] = 1.0
            
        binary_mask = binary_mask.view(batch_size, 1, grid_h, grid_w)
        full_mask = F.interpolate(binary_mask, size=(h, w), mode="nearest")
        return full_mask

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self._shared_step(batch, "train")
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.trainer.max_epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def _shared_step(self, batch: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
        image = batch["simulated"]
        B, C, H, W = image.shape
        
        mask = self.generate_mask(B, H, W, image.device)
        
        masked_image = image * (1 - mask) + self.mask_token * mask
        
        reconstruction = self(masked_image)
        
        if reconstruction.ndim >= 3 and image.ndim >= 3 and reconstruction.shape[-2:] != image.shape[-2:]:
            reconstruction = F.interpolate(reconstruction, size=image.shape[-2:], mode="bilinear", align_corners=False)
            
        loss_matrix = F.mse_loss(reconstruction, image, reduction="none")
        
        num_masked_elements = mask.sum() * C
        loss = (loss_matrix * mask).sum() / (num_masked_elements + 1e-8)
        
        self.log(
            f"{prefix}_loss", 
            loss, 
            prog_bar=True, 
            on_step=False, 
            on_epoch=True, 
            sync_dist=True,
            batch_size=B
        )
        
        return loss