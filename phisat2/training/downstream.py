from __future__ import annotations

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
    
from phisat2.tasks import TaskSpec
from phisat2.evaluation.metrics import build_metrics
from phisat2.utils.visualization import mask_to_rgb
from phisat2.utils.weights import _strip_compile_prefix

class DownstreamModule(L.LightningModule):
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
        encoder = getattr(self.model, "encoder", None)
        adapter  = getattr(self.model, "adapter",  None)
        head     = getattr(self.model, "head",     None)

        if encoder is not None and adapter is not None and head is not None:
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
        
        if self.spec.task == "segmentation":
            self._visualize_and_save_segmentation(batch, preds, targets, batch_idx)
            
        if self.spec.task == "pixel_regression":
            self._visualize_and_save_pixel_regression(batch, preds, targets, batch_idx)

    def configure_optimizers(self):
        trainable_params = [param for param in self.parameters() if param.requires_grad]
        
        optimizer = torch.optim.AdamW(trainable_params, lr=self.lr, weight_decay=1e-4)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.trainer.max_epochs
        )
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def on_load_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["state_dict"] = _strip_compile_prefix(
            checkpoint.get("state_dict", {})
        )

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

    @staticmethod
    def _percentile_stretch(
        t: torch.Tensor, lo: float = 2.0, hi: float = 98.0
    ) -> np.ndarray:
        flat = t.reshape(-1).float()
        v_lo = torch.quantile(flat, lo / 100.0)
        v_hi = torch.quantile(flat, hi / 100.0)
        return ((t.float() - v_lo) / (v_hi - v_lo + 1e-6)).clamp(0, 1).numpy()

    @staticmethod
    def _to_falsecolor(
        t: torch.Tensor,
        rgb_idx: tuple[int, int, int] = (3, 2, 1),
    ) -> np.ndarray:
        C = t.shape[0]
        idx = [c for c in rgb_idx if c < C]
        if len(idx) < 3:
            gray = DownstreamModule._percentile_stretch(t[0])
            return np.stack([gray, gray, gray], axis=-1)
        channels = [DownstreamModule._percentile_stretch(t[c]) for c in idx]
        return np.stack(channels, axis=-1)

    def _visualize_and_save_segmentation(self, batch, preds, targets, batch_idx, max_samples=5) -> None:
        if batch_idx % 100 != 0:
            return
            
        image = batch["image"]
        dataset_name = self.spec.dataset
        is_lulc = (dataset_name == "lulc")
        
        if preds.ndim == 4:
            preds = preds.argmax(dim=1)
            
        targets_np = targets.detach().cpu().numpy()
        preds_np = preds.detach().cpu().numpy()
        n = min(max_samples, image.shape[0])
        
        num_cols = 5 if is_lulc else 3
        fig, axes = plt.subplots(n, num_cols, figsize=(5 * num_cols, 4 * n), squeeze=False, constrained_layout=True)
        fig.suptitle(f"Debug Segmentation ({dataset_name}) - Batch {batch_idx}", fontsize=14, fontweight='bold')
        
        if is_lulc:
            col_titles = ["Original (RGB)", "Micro GT", "Micro Pred", "Macro GT", "Macro Pred"]
        else:
            col_titles = ["Original (RGB)", "Ground Truth", "Predictions"]
            
        for ax, title in zip(axes[0], col_titles):
            ax.set_title(title, fontsize=12)
        
        current_meta_micro = None 
        current_meta_macro = None
        
        for i in range(n):
            orig = image[i].detach().cpu()
            axes[i, 0].imshow(self._to_falsecolor(orig, rgb_idx=(3, 2, 1)))
            axes[i, 0].axis("off")
            
            gt_micro_rgb, current_meta_micro = mask_to_rgb(targets_np[i], dataset_name)
            pred_micro_rgb, _ = mask_to_rgb(preds_np[i], dataset_name)
            
            axes[i, 1].imshow(gt_micro_rgb)
            axes[i, 1].axis("off")
            
            axes[i, 2].imshow(pred_micro_rgb)
            axes[i, 2].axis("off")
            
            if is_lulc:
                gt_macro_rgb, current_meta_macro = mask_to_rgb(targets_np[i], "lulc_macro")
                pred_macro_rgb, _ = mask_to_rgb(preds_np[i], "lulc_macro")
                
                axes[i, 3].imshow(gt_macro_rgb)
                axes[i, 3].axis("off")
                
                axes[i, 4].imshow(pred_macro_rgb)
                axes[i, 4].axis("off")
            
        if current_meta_micro is not None:
            patches_micro = [
                mpatches.Patch(color=np.array(color)/255.0, label=name)
                for class_idx, (name, color) in current_meta_micro.items()
            ]
            
            if is_lulc:
                fig.legend(handles=patches_micro, loc='upper center', bbox_to_anchor=(0.35, 0), ncol=6, title="Micro Classes", fontsize=10)
                
                if current_meta_macro is not None:
                    unique_macro = {name: color for name, color in current_meta_macro.values()}
                    patches_macro = [
                        mpatches.Patch(color=np.array(color)/255.0, label=name)
                        for name, color in unique_macro.items()
                    ]
                    fig.legend(handles=patches_macro, loc='upper center', bbox_to_anchor=(0.8, 0), ncol=4, title="Macro Classes", fontsize=10)
            else:
                fig.legend(handles=patches_micro, loc='upper center', bbox_to_anchor=(0.5, 0), ncol=len(current_meta_micro), fontsize=11)
            
        os.makedirs(self.trainer.default_root_dir, exist_ok=True)
        save_path = os.path.join(self.trainer.default_root_dir, f"segmentation_debug_batch_{batch_idx}.png")
        
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[Viz] saved → {save_path}")
        
        
    def _visualize_and_save_pixel_regression(
        self, batch, preds, targets, batch_idx, max_samples=5
    ) -> None:
        
        if batch_idx % 100 != 0:
            return
            
        image = batch["image"]
        
        if preds.ndim == 4 and preds.shape[1] == 1:
            preds = preds.squeeze(1)
        if targets.ndim == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)

        if preds.ndim == 4:
            preds = preds[:, 0, ...]
        if targets.ndim == 4:
            targets = targets[:, 0, ...]

        targets_np = targets.detach().cpu().float().numpy()
        preds_np = preds.detach().cpu().float().numpy()
        
        vmin = min(np.percentile(targets_np, 2), np.percentile(preds_np, 2))
        vmax = max(np.percentile(targets_np, 98), np.percentile(preds_np, 98))
            
        n = min(max_samples, image.shape[0])
        
        fig, axes = plt.subplots(n, 3, figsize=(15, 4 * n), squeeze=False, constrained_layout=True)
        fig.suptitle(f"Debug Pixel Regression ({self.spec.dataset}) - Batch {batch_idx}\nScale: [{vmin:.3f}, {vmax:.3f}]", fontsize=14)
        
        col_titles = ["Original (RGB)", "Ground Truth", "Predictions"]
        for ax, title in zip(axes[0], col_titles):
            ax.set_title(title, fontsize=11)
        
        for i in range(n):
            orig = image[i].detach().cpu()
            
            axes[i, 0].imshow(self._to_falsecolor(orig, rgb_idx=(3, 2, 1)))
            axes[i, 0].axis("off")
            
            axes[i, 1].imshow(targets_np[i], cmap="viridis", vmin=vmin, vmax=vmax)
            axes[i, 1].axis("off")
            
            im_pred = axes[i, 2].imshow(preds_np[i], cmap="viridis", vmin=vmin, vmax=vmax)
            axes[i, 2].axis("off")
            
        cbar = fig.colorbar(im_pred, ax=axes[:, 2], shrink=0.8, pad=0.02)
        cbar.ax.tick_params(labelsize=8)
                
        os.makedirs(self.trainer.default_root_dir, exist_ok=True)
        save_path = os.path.join(
            self.trainer.default_root_dir, 
            f"pixel_regression_debug_batch_{batch_idx}.png"
        )
        
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[Viz] saved → {save_path}")