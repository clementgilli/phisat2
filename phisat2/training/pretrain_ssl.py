from __future__ import annotations

import math
import os

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from phisat2.tasks import TaskSpec
from phisat2.utils.weights import _strip_compile_prefix


class SSLPretrainModule(L.LightningModule):

    def __init__(
        self,
        model: nn.Module,
        spec: TaskSpec,
        *,
        lr: float,
        weight_decay: float,
        patch_size: int = 16,
        mask_ratio: float = 0.6,
        masking_strategy: str = "block",   # "random" | "block"
    ) -> None:
        super().__init__()
        self.model = model
        self.spec = spec
        self.lr = lr
        self.weight_decay = weight_decay
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.masking_strategy = masking_strategy

        print(
            f"SSLPretrainModule | patch_size={patch_size} "
            f"| mask_ratio={mask_ratio} | masking={masking_strategy}"
        )

        self.mask_token = nn.Parameter(torch.zeros(1, spec.num_outputs, 1, 1))

        self.save_hyperparameters(
            {
                "task": spec.task,
                "dataset": spec.dataset,
                "lr": lr,
                "weight_decay": weight_decay,
                "masking_strategy": masking_strategy,
            }
        )

    # ------------------------------------------------------------------ #
    #  Forward                                                             #
    # ------------------------------------------------------------------ #

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image)

    # ------------------------------------------------------------------ #
    #  Mask generation                                                     #
    # ------------------------------------------------------------------ #

    def generate_mask(
        self, batch_size: int, h: int, w: int, device: torch.device
    ) -> torch.Tensor:
        """Return a (B, 1, H, W) binary mask — 1 = masked, 0 = visible."""
        if self.masking_strategy == "block":
            return self._block_mask(batch_size, h, w, device)
        return self._random_mask(batch_size, h, w, device)

    def _random_mask(
        self, batch_size: int, h: int, w: int, device: torch.device
    ) -> torch.Tensor:
        
        grid_h, grid_w = h // self.patch_size, w // self.patch_size
        num_patches = grid_h * grid_w
        num_masked  = int(num_patches * self.mask_ratio)

        noise       = torch.rand(batch_size, num_patches, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)                  # (B, P)

        binary_mask = torch.zeros(batch_size, num_patches, device=device)
        binary_mask.scatter_(1, ids_shuffle[:, :num_masked], 1.0)  # ← no for-loop

        binary_mask = binary_mask.view(batch_size, 1, grid_h, grid_w)
        return F.interpolate(binary_mask, size=(h, w), mode="nearest")

    def _block_mask(
        self, batch_size: int, h: int, w: int, device: torch.device
    ) -> torch.Tensor:
        
        grid_h  = h // self.patch_size
        grid_w  = w // self.patch_size
        target  = int(grid_h * grid_w * self.mask_ratio)
        max_bh  = max(2, grid_h // 3)
        max_bw  = max(2, grid_w // 3)

        masks = []
        for _ in range(batch_size):
            m = torch.zeros(grid_h, grid_w)   # CPU tensor, moved to device below
            for _ in range(200):              # safety upper-bound on iterations
                bh   = torch.randint(1, max_bh + 1, (1,)).item()
                bw   = torch.randint(1, max_bw + 1, (1,)).item()
                top  = torch.randint(0, grid_h - bh + 1, (1,)).item()
                left = torch.randint(0, grid_w - bw + 1, (1,)).item()
                m[top: top + bh, left: left + bw] = 1.0
                if m.sum() >= target:
                    break
            masks.append(m)

        binary_mask = torch.stack(masks).unsqueeze(1).to(device)   # (B, 1, gh, gw)
        return F.interpolate(binary_mask, size=(h, w), mode="nearest")

    # ------------------------------------------------------------------ #
    #  Loss                                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _masked_band_loss(reconstruction, target, mask):
        loss_map   = F.mse_loss(reconstruction, target, reduction="none")  # (B, C, H, W)
        masked_sum = (loss_map * mask).sum()
        n_elements = mask.sum() * target.shape[1]
        return masked_sum / (n_elements + 1e-8)

    # ------------------------------------------------------------------ #
    #  Lightning steps                                                     #
    # ------------------------------------------------------------------ #

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> None:
        self._shared_step(batch, "val")

    def test_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> None:
        image = batch["simulated"]
        B, C, H, W = image.shape
        mask           = self.generate_mask(B, H, W, image.device)
        masked_image   = image * (1 - mask) + self.mask_token * mask
        reconstruction = self(masked_image)
        loss = self._masked_band_loss(reconstruction, image, mask)
        self.log("test_loss", loss, sync_dist=True, batch_size=B)
        self._visualize_and_save(image, masked_image, reconstruction, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def _shared_step(
        self, batch: dict[str, torch.Tensor], prefix: str
    ) -> torch.Tensor:
        image = batch["sentinel2"]
        B, C, H, W = image.shape

        mask           = self.generate_mask(B, H, W, image.device)
        masked_image   = image * (1 - mask) + self.mask_token * mask
        reconstruction = self(masked_image)

        loss = self._masked_band_loss(reconstruction, image, mask)
        self.log(
            f"{prefix}_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=B,
        )
        return loss
    
    def on_load_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["state_dict"] = _strip_compile_prefix(
            checkpoint.get("state_dict", {})
        )


    # ------------------------------------------------------------------ #
    #  Visualisation helpers                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _percentile_stretch(
        t: torch.Tensor, ref_t: torch.Tensor = None, lo: float = 2.0, hi: float = 98.0
    ) -> np.ndarray:
        if ref_t is None:
            ref_t = t
            
        flat = ref_t.reshape(-1).float()
        v_lo = torch.quantile(flat, lo / 100.0)
        v_hi = torch.quantile(flat, hi / 100.0)
        
        return ((t.float() - v_lo) / (v_hi - v_lo + 1e-6)).clamp(0, 1).numpy()

    @staticmethod
    def _to_falsecolor(
        t: torch.Tensor,
        ref_t: torch.Tensor = None,
        rgb_idx: tuple[int, int, int] = (3, 2, 1),
    ) -> np.ndarray:
        if ref_t is None:
            ref_t = t
            
        C = t.shape[0]
        idx = [c for c in rgb_idx if c < C]
        
        if len(idx) < 3:
            gray = SSLPretrainModule._percentile_stretch(t[0], ref_t[0])
            return np.stack([gray, gray, gray], axis=-1)
            
        channels = [SSLPretrainModule._percentile_stretch(t[c], ref_t[c]) for c in idx]
        return np.stack(channels, axis=-1)

    def _visualize_and_save(
        self,
        image: torch.Tensor,
        masked_image: torch.Tensor,
        reconstruction: torch.Tensor,
        batch_idx: int,
        max_samples: int = 5,
    ) -> None:
        
        if batch_idx % 1 != 0:
            return

        n = min(max_samples, image.shape[0])
        
        fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n), squeeze=False)
        fig.suptitle(
            f"Reconstruction (RGB)\n"
            f"strategy={self.masking_strategy}   ratio={self.mask_ratio}",
            fontsize=14,
        )
        
        col_titles = [
            "Original",
            "Masked Input",
            "Reconstruction",
        ]
        
        for ax, title in zip(axes[0], col_titles):
            ax.set_title(title, fontsize=11)

        for i in range(n):
            orig  = image[i].detach().cpu()
            mskd  = masked_image[i].detach().cpu()
            recon = reconstruction[i].detach().cpu()

            axes[i, 0].imshow(self._to_falsecolor(orig, ref_t=orig))
            axes[i, 1].imshow(self._to_falsecolor(mskd, ref_t=orig))
            axes[i, 2].imshow(self._to_falsecolor(recon, ref_t=orig))

            for ax in axes[i]:
                ax.axis("off")

        os.makedirs(self.trainer.default_root_dir, exist_ok=True)
        path = os.path.join(self.trainer.default_root_dir, f"reconstruction_debug_batch_{batch_idx}.png")
        plt.tight_layout()
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[Viz] saved → {path}")