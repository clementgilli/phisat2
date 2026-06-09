from __future__ import annotations

import os
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torchmetrics.functional import jaccard_index
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

from src.utils import plot_tsne

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

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        img_real, img_sim = batch["real"], batch["simulated"]

        # ==========================================
        # 1. FORWARD ENCODERS
        # ==========================================
        feat_sim = self.teacher(img_sim)
        feat_real = self.student(img_real)

        if isinstance(feat_sim, list):
            feat_sim = {name: feat_sim[i] for i, name in enumerate(self.feature_layers) if i < len(feat_sim)}
            feat_real = {name: feat_real[i] for i, name in enumerate(self.feature_layers) if i < len(feat_real)}

        # ==========================================
        # 2. LATENT DOMAIN GAP
        # ==========================================
        for layer in self.feature_layers:
            if layer not in feat_sim: continue
                
            f_s = F.adaptive_avg_pool2d(feat_sim[layer], 1).flatten(1)
            f_r = F.adaptive_avg_pool2d(feat_real[layer], 1).flatten(1)

            cos_sim = F.cosine_similarity(f_s, f_r, dim=1).mean()
            self.log(f"latent/cosine_sim_{layer}", cos_sim, on_step=False, on_epoch=True)

            self.stored_features_sim[layer].append(f_s.cpu().numpy())
            self.stored_features_real[layer].append(f_r.cpu().numpy())

        # ==========================================
        # 3. DOWNSTREAM CONSISTENCY
        # ==========================================
        for task_name, decoder in self.decoders.items():
            feat_sim_list = [feat_sim[layer] for layer in self.feature_layers if layer in feat_sim]
            feat_real_list = [feat_real[layer] for layer in self.feature_layers if layer in feat_real]
            
            logits_sim = decoder(feat_sim_list)
            logits_real = decoder(feat_real_list)
            
            is_spatial = logits_sim.ndim == 4      # [B, C, H, W]
            num_channels = logits_sim.shape[1]     # C

            # ---- A. SEGMENTATION (Spatial, >1 classes) ----
            if is_spatial and num_channels > 1:
                preds_sim = torch.argmax(logits_sim, dim=1)
                preds_real = torch.argmax(logits_real, dim=1)
                
                # Hard: mIoU
                miou = jaccard_index(preds_real, preds_sim, task="multiclass", num_classes=num_channels, average='macro')
                self.log(f"consistency/{task_name}_miou", miou, on_step=False, on_epoch=True)
                
                # Soft: KL Divergence
                p_sim, log_p_sim, log_p_real = F.softmax(logits_sim, dim=1), F.log_softmax(logits_sim, dim=1), F.log_softmax(logits_real, dim=1)
                kl_div = torch.sum(p_sim * (log_p_sim - log_p_real), dim=1).mean()
                self.log(f"consistency/{task_name}_kl", kl_div, on_step=False, on_epoch=True)

            # ---- B. CLASSIFICATION GLOBALE (1D, >1 classes) ----
            elif not is_spatial and num_channels > 1:
                preds_sim = torch.argmax(logits_sim, dim=1)
                preds_real = torch.argmax(logits_real, dim=1)
                
                # Hard: Accuracy
                acc = (preds_real == preds_sim).float().mean()
                self.log(f"consistency/{task_name}_acc", acc, on_step=False, on_epoch=True)
                
                # Soft: KL Divergence
                p_sim, log_p_sim, log_p_real = F.softmax(logits_sim, dim=1), F.log_softmax(logits_sim, dim=1), F.log_softmax(logits_real, dim=1)
                kl_div = torch.sum(p_sim * (log_p_sim - log_p_real), dim=1).mean()
                self.log(f"consistency/{task_name}_kl", kl_div, on_step=False, on_epoch=True)

            # ---- C. RÉGRESSION PIXEL / GLOBALE (1 canal) ----
            else:
                mse = F.mse_loss(logits_real, logits_sim)
                self.log(f"consistency/{task_name}_mse", mse, on_step=False, on_epoch=True)

    # ==========================================
    # 4. END EPOCH (PAD & t-SNE)
    # ==========================================
    def on_test_epoch_end(self) -> None:
        print("\n" + "="*50)
        print("COMPUTING PAD SCORES & t-SNE")
        print("="*50)
        
        save_dir = self.trainer.default_root_dir
        os.makedirs(save_dir, exist_ok=True)
        
        MAX_TSNE_SAMPLES = 4000

        for layer in self.feature_layers:
            if not self.stored_features_sim[layer]: continue
                
            X_sim = np.concatenate(self.stored_features_sim[layer], axis=0)
            X_real = np.concatenate(self.stored_features_real[layer], axis=0)
            
            # --- 1. PAD Score ---
            X_combined = np.vstack([X_sim, X_real])
            labels = np.array([0] * len(X_sim) + [1] * len(X_real))

            clf = make_pipeline(StandardScaler(), LinearSVC(C=0.01, dual=False, max_iter=10000))
            accuracies = cross_val_score(clf, X_combined, labels, cv=5, scoring='accuracy', n_jobs=-1)
            
            pad_score = 2 * (1 - 2 * min(1.0 - np.mean(accuracies), 0.5))
            self.log(f"latent/pad_score_{layer}", pad_score)
            print(f"Layer: {layer:<10} | PAD Score: {pad_score:.4f}")

            # --- 2. t-SNE ---
            if len(X_sim) > MAX_TSNE_SAMPLES:
                idx_sim = np.random.choice(len(X_sim), MAX_TSNE_SAMPLES, replace=False)
                idx_real = np.random.choice(len(X_real), MAX_TSNE_SAMPLES, replace=False)
                X_sim_plot, X_real_plot = X_sim[idx_sim], X_real[idx_real]
            else:
                X_sim_plot, X_real_plot = X_sim, X_real
                
            tsne_filepath = os.path.join(save_dir, f"tsne_{layer}.png")
            plot_tsne(X_sim_plot, X_real_plot, save_path=tsne_filepath)

        print("="*50 + "\n")