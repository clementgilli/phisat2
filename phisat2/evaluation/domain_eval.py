from __future__ import annotations

import os
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from torchmetrics.functional import jaccard_index
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

from phisat2.utils.visualization import mask_to_rgb, plot_tsne

class DomainEvalModule(L.LightningModule):
    def __init__(
        self,
        teacher_encoder: nn.Module,
        student_encoder: nn.Module,
        decoders: nn.ModuleDict,
    ) -> None:
        super().__init__()
        self.teacher = teacher_encoder
        self.student = student_encoder
        self.decoders = decoders
        
        self.feature_layers = ['enc_0', 'enc_1', 'enc_2', 'bottleneck']
        
        self.eval()
        self.requires_grad_(False)

        self.stored_features_sim = {layer: [] for layer in self.feature_layers}
        self.stored_features_real = {layer: [] for layer in self.feature_layers}

    # ==========================================
    # UTILS VISUALISATION
    # ==========================================
    @staticmethod
    def _percentile_stretch(t: torch.Tensor, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
        flat = t.reshape(-1).float()
        v_lo = torch.quantile(flat, lo / 100.0)
        v_hi = torch.quantile(flat, hi / 100.0)
        return ((t.float() - v_lo) / (v_hi - v_lo + 1e-6)).clamp(0, 1).numpy()

    @staticmethod
    def _to_falsecolor(t: torch.Tensor, rgb_idx: tuple[int, int, int] = (3, 2, 1)) -> np.ndarray:
        C = t.shape[0]
        idx = [c for c in rgb_idx if c < C]
        if len(idx) < 3:
            gray = DomainEvalModule._percentile_stretch(t[0])
            return np.stack([gray, gray, gray], axis=-1)
        channels = [DomainEvalModule._percentile_stretch(t[c]) for c in idx]
        return np.stack(channels, axis=-1)

    def _visualize_consistency_segmentation(self, img_sim, img_real, preds_sim, preds_real, task_name, batch_idx, max_samples=4) -> None:
        # Récupération du vrai nom de dataset depuis le dictionnaire (ex: task_name='lulc' -> dataset_name='lulc')
        dataset_name = task_name 
        
        if preds_sim.ndim == 4:
            preds_sim = preds_sim.argmax(dim=1)
        if preds_real.ndim == 4:
            preds_real = preds_real.argmax(dim=1)
            
        preds_sim_np = preds_sim.detach().cpu().numpy()
        preds_real_np = preds_real.detach().cpu().numpy()
        n = min(max_samples, img_sim.shape[0])
        
        fig, axes = plt.subplots(n, 4, figsize=(20, 4 * n), squeeze=False, constrained_layout=True)
        fig.suptitle(f"Consistency Segmentation ({task_name}) - Batch {batch_idx}", fontsize=16, fontweight='bold')
        
        col_titles = ["Image Simulated", "Prediction Simulated", "Image Real", "Prediction Real"]
        for ax, title in zip(axes[0], col_titles):
            ax.set_title(title, fontsize=13)
        
        current_meta = None 
        
        for i in range(n):
            # SIMULATED
            orig_sim = img_sim[i].detach().cpu()
            axes[i, 0].imshow(self._to_falsecolor(orig_sim, rgb_idx=(3, 2, 1)))
            axes[i, 0].axis("off")
            
            pred_sim_rgb, current_meta = mask_to_rgb(preds_sim_np[i], dataset_name)
            axes[i, 1].imshow(pred_sim_rgb)
            axes[i, 1].axis("off")
            
            # REAL
            orig_real = img_real[i].detach().cpu()
            axes[i, 2].imshow(self._to_falsecolor(orig_real, rgb_idx=(3, 2, 1)))
            axes[i, 2].axis("off")
            
            pred_real_rgb, _ = mask_to_rgb(preds_real_np[i], dataset_name)
            axes[i, 3].imshow(pred_real_rgb)
            axes[i, 3].axis("off")
            
        if current_meta is not None:
            legend_patches = [
                mpatches.Patch(color=np.array(color)/255.0, label=name)
                for class_idx, (name, color) in current_meta.items()
            ]
            fig.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=len(current_meta), fontsize=12)
            
        save_path = os.path.join(self.trainer.default_root_dir, f"consistency_seg_{task_name}_batch_{batch_idx}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        
    def _visualize_consistency_regression(self, img_sim, img_real, preds_sim, preds_real, task_name, batch_idx, max_samples=4) -> None:
        if preds_sim.ndim == 4 and preds_sim.shape[1] == 1: preds_sim = preds_sim.squeeze(1)
        if preds_real.ndim == 4 and preds_real.shape[1] == 1: preds_real = preds_real.squeeze(1)
        if preds_sim.ndim == 4: preds_sim = preds_sim[:, 0, ...]
        if preds_real.ndim == 4: preds_real = preds_real[:, 0, ...]

        preds_sim_np = preds_sim.detach().cpu().float().numpy()
        preds_real_np = preds_real.detach().cpu().float().numpy()
        
        vmin = min(np.percentile(preds_sim_np, 2), np.percentile(preds_real_np, 2))
        vmax = max(np.percentile(preds_sim_np, 98), np.percentile(preds_real_np, 98))
            
        n = min(max_samples, img_sim.shape[0])
        
        fig, axes = plt.subplots(n, 4, figsize=(20, 4 * n), squeeze=False, constrained_layout=True)
        fig.suptitle(f"Consistency Regression ({task_name}) - Batch {batch_idx}\nScale: [{vmin:.3f}, {vmax:.3f}]", fontsize=16, fontweight='bold')
        
        col_titles = ["Image Simulated", "Prediction Simulated", "Image Real", "Prediction Real"]
        for ax, title in zip(axes[0], col_titles):
            ax.set_title(title, fontsize=13)
        
        for i in range(n):
            # SIMULATED
            orig_sim = img_sim[i].detach().cpu()
            axes[i, 0].imshow(self._to_falsecolor(orig_sim, rgb_idx=(3, 2, 1)))
            axes[i, 0].axis("off")
            
            im_pred_s = axes[i, 1].imshow(preds_sim_np[i], cmap="viridis", vmin=vmin, vmax=vmax)
            axes[i, 1].axis("off")
            
            # REAL
            orig_real = img_real[i].detach().cpu()
            axes[i, 2].imshow(self._to_falsecolor(orig_real, rgb_idx=(3, 2, 1)))
            axes[i, 2].axis("off")
            
            im_pred_r = axes[i, 3].imshow(preds_real_np[i], cmap="viridis", vmin=vmin, vmax=vmax)
            axes[i, 3].axis("off")
            
        cbar = fig.colorbar(im_pred_r, ax=axes[:, [1, 3]], shrink=0.8, pad=0.02)
        cbar.ax.tick_params(labelsize=10)
                
        save_path = os.path.join(self.trainer.default_root_dir, f"consistency_reg_{task_name}_batch_{batch_idx}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


    # ==========================================
    # TEST STEP
    # ==========================================
    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        img_real, img_sim = batch["real"], batch["simulated"]

        feat_sim = self.teacher(img_sim)
        feat_real = self.student(img_real)

        if isinstance(feat_sim, list):
            feat_sim = {name: feat_sim[i] for i, name in enumerate(self.feature_layers) if i < len(feat_sim)}
            feat_real = {name: feat_real[i] for i, name in enumerate(self.feature_layers) if i < len(feat_real)}

        for layer in self.feature_layers:
            if layer not in feat_sim: continue
            f_s = F.adaptive_avg_pool2d(feat_sim[layer], 1).flatten(1)
            f_r = F.adaptive_avg_pool2d(feat_real[layer], 1).flatten(1)
            cos_sim = F.cosine_similarity(f_s, f_r, dim=1).mean()
            self.log(f"latent/cosine_sim_{layer}", cos_sim, on_step=False, on_epoch=True)
            self.stored_features_sim[layer].append(f_s.cpu().numpy())
            self.stored_features_real[layer].append(f_r.cpu().numpy())

        for task_name, decoder in self.decoders.items():
            feat_sim_list = [feat_sim[layer] for layer in self.feature_layers if layer in feat_sim]
            feat_real_list = [feat_real[layer] for layer in self.feature_layers if layer in feat_real]
            
            logits_sim = decoder(feat_sim_list)
            logits_real = decoder(feat_real_list)
            
            is_spatial = logits_sim.ndim == 4      
            num_channels = logits_sim.shape[1]     

            # ---- A. SEGMENTATION ----
            if is_spatial and num_channels > 1:
                preds_sim = torch.argmax(logits_sim, dim=1)
                preds_real = torch.argmax(logits_real, dim=1)
                
                miou = jaccard_index(preds_real, preds_sim, task="multiclass", num_classes=num_channels, average='macro')
                self.log(f"consistency/{task_name}_miou", miou, on_step=False, on_epoch=True)
                
                p_sim, log_p_sim, log_p_real = F.softmax(logits_sim, dim=1), F.log_softmax(logits_sim, dim=1), F.log_softmax(logits_real, dim=1)
                kl_div = torch.sum(p_sim * (log_p_sim - log_p_real), dim=1).mean()
                self.log(f"consistency/{task_name}_kl", kl_div, on_step=False, on_epoch=True)
                
                # Plot (seulement tous les 100 batchs pour ne pas spammer le disque)
                if batch_idx % 100 == 0:
                    self._visualize_consistency_segmentation(img_sim, img_real, logits_sim, logits_real, task_name, batch_idx)

            # ---- B. CLASSIFICATION GLOBALE ----
            elif not is_spatial and num_channels > 1:
                preds_sim = torch.argmax(logits_sim, dim=1)
                preds_real = torch.argmax(logits_real, dim=1)
                
                acc = (preds_real == preds_sim).float().mean()
                self.log(f"consistency/{task_name}_acc", acc, on_step=False, on_epoch=True)
                
                p_sim, log_p_sim, log_p_real = F.softmax(logits_sim, dim=1), F.log_softmax(logits_sim, dim=1), F.log_softmax(logits_real, dim=1)
                kl_div = torch.sum(p_sim * (log_p_sim - log_p_real), dim=1).mean()
                self.log(f"consistency/{task_name}_kl", kl_div, on_step=False, on_epoch=True)

            # ---- C. RÉGRESSION PIXEL / GLOBALE ----
            else:
                mse = F.mse_loss(logits_real, logits_sim)
                self.log(f"consistency/{task_name}_mse", mse, on_step=False, on_epoch=True)
                
                # Plot Regression Pixel (si spatial)
                if batch_idx % 100 == 0 and is_spatial:
                    self._visualize_consistency_regression(img_sim, img_real, logits_sim, logits_real, task_name, batch_idx)


    def on_test_epoch_end(self) -> None:
        save_dir = self.trainer.default_root_dir
        os.makedirs(save_dir, exist_ok=True)
        MAX_TSNE_SAMPLES = 4000

        for layer in self.feature_layers:
            if not self.stored_features_sim[layer]: continue
                
            X_sim = np.concatenate(self.stored_features_sim[layer], axis=0)
            X_real = np.concatenate(self.stored_features_real[layer], axis=0)
            
            X_combined = np.vstack([X_sim, X_real])
            labels = np.array([0] * len(X_sim) + [1] * len(X_real))

            clf = make_pipeline(StandardScaler(), LinearSVC(C=0.01, dual=False, max_iter=10000))
            accuracies = cross_val_score(clf, X_combined, labels, cv=5, scoring='accuracy', n_jobs=-1)
            
            pad_score = 2 * (1 - 2 * min(1.0 - np.mean(accuracies), 0.5))
            self.log(f"latent/pad_score_{layer}", pad_score)

            if len(X_sim) > MAX_TSNE_SAMPLES:
                idx_sim = np.random.choice(len(X_sim), MAX_TSNE_SAMPLES, replace=False)
                idx_real = np.random.choice(len(X_real), MAX_TSNE_SAMPLES, replace=False)
                X_sim_plot, X_real_plot = X_sim[idx_sim], X_real[idx_real]
            else:
                X_sim_plot, X_real_plot = X_sim, X_real
                
            tsne_filepath = os.path.join(save_dir, f"tsne_{layer}.png")
            plot_tsne(X_sim_plot, X_real_plot, save_path=tsne_filepath)