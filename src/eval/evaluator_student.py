import torch
import torch.nn.functional as F
import numpy as np
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.functional.image import structural_similarity_index_measure as ssim_func
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr_func
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from torchmetrics.functional import jaccard_index
from src.core.utils import AverageMeter, set_seed
import json
import os
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

class MultiTaskEvaluator:
    def __init__(self, config, device='cuda', use_lpips=False):
        """
        Initializes the paired evaluation pipeline for Domain Adaptation.
        Upgraded for Multi-Scale feature extraction (enc_0, enc_1, enc_2, bottleneck).
        """
        self.device = device
        self.config = config
        self.use_lpips = use_lpips
        
        self.climate_weights = self.config["training"].get("climate_loss_weights", None)
        if self.climate_weights is not None:
            self.climate_weights = torch.tensor(self.climate_weights, dtype=torch.float32).to(self.device)
        
        # --- Shared Metric Objects ---
        if self.use_lpips:
            self.lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(self.device)
        
        num_climates = 31
        self.acc_metric = MulticlassAccuracy(num_classes=num_climates, average='micro').to(self.device)
        self.f1_metric = MulticlassF1Score(num_classes=num_climates, average='macro').to(self.device)
        
        # --- Domain-Specific Meters ---
        self.sim_meters = self._create_meters()
        self.real_meters = self._create_meters()
        
        # --- Cross-Domain Multi-Scale Meters & Storage ---
        self.feature_layers = ['enc_0', 'enc_1', 'enc_2', 'bottleneck']
        self.metric_cos_sim = {layer: AverageMeter(f'Latent_Cosine_Sim_{layer}') for layer in self.feature_layers}
        
        # Storage for PAD SVM and t-SNE
        self.saved_features = {
            'sim': {layer: [] for layer in self.feature_layers},
            'real': {layer: [] for layer in self.feature_layers}
        }
        self.pad_scores = {}
        
        # --- Macro Climate Metrics (6 Classes) ---
        self.num_macro_climates = 6
        self.macro_mapping = torch.tensor([
            0,                                               # 0: Water/NoData
            1, 1, 1,                                         # 1: Tropical 
            2, 2, 2, 2,                                      # 2: Arid 
            3, 3, 3, 3, 3, 3, 3, 3, 3,                       # 3: Temperate 
            4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,              # 4: Cold 
            5, 5                                             # 5: Polar 
        ], dtype=torch.long, device=self.device)
        
        self.macro_acc_metric = MulticlassAccuracy(num_classes=self.num_macro_climates, average='micro').to(self.device)
        self.macro_f1_metric = MulticlassF1Score(num_classes=self.num_macro_climates, average='macro').to(self.device)
        
        self.seg_tasks = {
            'lc': 11, 'anomaly_detection': 9, 'burned_area': 4, 
            'clouds': 2, 'fire': 3, 'worldfloods': 3
        }
        
        self.consistency_meters = {}
        for task in self.seg_tasks.keys():
            self.consistency_meters[f'{task}_consistency_miou'] = AverageMeter(f'Consistency_{task}_mIoU')
            self.consistency_meters[f'{task}_consistency_kl'] = AverageMeter(f'Consistency_{task}_KL')

    def _create_meters(self):
        """Helper to initialize a complete set of meters for a single domain."""
        meters = {
            'loss_climate_ce': AverageMeter('Climate_CE'),
            'loss_geo_mse': AverageMeter('Geo_MSE'),
            'metric_ssim': AverageMeter('Recon_SSIM'),
            'metric_psnr': AverageMeter('Recon_PSNR'),
            'metric_acc': AverageMeter('Climate_Acc'),
            'metric_f1': AverageMeter('Climate_F1'),
            'metric_dist': AverageMeter('Geo_Dist_km'),
            'metric_macro_acc': AverageMeter('Climate_Acc_Macro'),
            'metric_macro_f1': AverageMeter('Climate_F1_Macro'),
            'lc_consistency_miou': AverageMeter('Consistency_LC_mIoU'),
            'lc_consistency_kl': AverageMeter('Consistency_LC_KL_Div')
        }
        if self.use_lpips:
            meters['metric_lpips'] = AverageMeter('Recon_LPIPS_RGB')
        return meters

    def reset(self):
        """Resets all meters, feature storages, and cross-domain metrics."""
        for meters in [self.sim_meters, self.real_meters]:
            for m in meters.values():
                m.reset()
                
        for layer in self.feature_layers:
            self.metric_cos_sim[layer].reset()
            self.saved_features['sim'][layer] = []
            self.saved_features['real'][layer] = []
            
        self.pad_scores = {}
        
        for m in self.consistency_meters.values(): m.reset()

    def haversine_distance(self, pred_coords, true_lat, true_lon):
        """Approximates distance in kilometers between predicted coordinates and ground truth."""
        pred_pi_u = torch.atan2(pred_coords[:, 0], pred_coords[:, 1])
        pred_pi_v = torch.atan2(pred_coords[:, 2], pred_coords[:, 3])
        
        pred_u = pred_pi_u / np.pi
        pred_v = pred_pi_v / np.pi
        
        pred_lat_deg = pred_u * 90.0
        pred_lon_deg = pred_v * 180.0
        
        pred_lat_rad = torch.deg2rad(pred_lat_deg)
        pred_lon_rad = torch.deg2rad(pred_lon_deg)
        
        true_lat_rad = torch.deg2rad(true_lat)
        true_lon_rad = torch.deg2rad(true_lon)
        
        dlat = true_lat_rad - pred_lat_rad
        dlon = true_lon_rad - pred_lon_rad
        
        a = torch.sin(dlat/2)**2 + torch.cos(pred_lat_rad) * torch.cos(true_lat_rad) * torch.sin(dlon/2)**2
        c = 2 * torch.arcsin(torch.sqrt(torch.clamp(a, 0, 1))) 
        earth_radius = 6371.0 
        
        return c * earth_radius

    def get_gt_coords_encoded(self, lat, lon):
        """Encodes Ground Truth degrees into [sin, cos, sin, cos] targets."""
        u, v = lat / 90.0, lon / 180.0
        return torch.stack([
            torch.sin(np.pi * u), torch.cos(np.pi * u), 
            torch.sin(np.pi * v), torch.cos(np.pi * v)
        ], dim=1)
        
    def compute_PAD_scores(self):
        """
        Computes the Proxy-A-Distance (PAD) score for each layer using an SVM.
        Must be called at the end of the epoch, after all batches are evaluated.
        """
        for layer in self.feature_layers:
            if len(self.saved_features['sim'][layer]) == 0:
                continue
                
            X_sim = np.concatenate(self.saved_features['sim'][layer], axis=0)
            X_real = np.concatenate(self.saved_features['real'][layer], axis=0)
            
            X_combined = np.vstack([X_sim, X_real])
            labels = np.array([0] * len(X_sim) + [1] * len(X_real))

            clf = make_pipeline(
                StandardScaler(),
                LinearSVC(C=0.01, dual=False, max_iter=10000)
            )

            accuracies = cross_val_score(clf, X_combined, labels, cv=5, scoring='accuracy', n_jobs=-1)
            
            err = 1.0 - np.mean(accuracies)
            err = min(err, 0.5)
            
            pad_score = 2 * (1 - 2 * err)
            self.pad_scores[layer] = pad_score
            
        return self.pad_scores

    def get_tsne_features(self, layer):
        """
        Returns the combined feature matrix X and labels y for a specific layer.
        Perfect for feeding directly into sklearn's TSNE.
        Labels: 0 = SIM, 1 = REAL.
        """
        if layer not in self.feature_layers:
            raise ValueError(f"Layer {layer} not found. Choose from {self.feature_layers}")
            
        X_sim = np.concatenate(self.saved_features['sim'][layer], axis=0)
        X_real = np.concatenate(self.saved_features['real'][layer], axis=0)
        X_combined = np.vstack([X_sim, X_real])
        labels = np.array([0] * len(X_sim) + [1] * len(X_real))
        
        return X_combined, labels

    @torch.no_grad()
    def evaluate_paired_batch(self, batch, preds_sim, preds_real, feat_sim, feat_real, seed=42):
        """
        Evaluates both SIM and REAL domains.
        Expects dictionaries for predictions (climate, coords, reconstruction) 
        and dictionaries for features (enc_0, enc_1, enc_2, bottleneck).
        """
        set_seed(seed) 
        
        B = batch['sim'].size(0)
        
        true_climate = batch['climate'].to(self.device)
        true_lat = batch['lat'].to(self.device)
        true_lon = batch['lon'].to(self.device)
        true_coords_enc = self.get_gt_coords_encoded(true_lat, true_lon).to(self.device)
        
        sim_imgs = batch['sim'].to(self.device)
        real_imgs = batch['real'].to(self.device)
        
        # 1. Update Domain-Specific Task Metrics
        self._update_domain_metrics(
            preds_sim, sim_imgs, true_climate, true_lat, true_lon, true_coords_enc, self.sim_meters, B
        )
        self._update_domain_metrics(
            preds_real, real_imgs, true_climate, true_lat, true_lon, true_coords_enc, self.real_meters, B
        )
        
        # 2. Update Multi-Scale Feature Metrics (Cosine + Save for PAD)
        for layer in self.feature_layers:
            # Global Average Pooling + Flattening
            f_s = F.adaptive_avg_pool2d(feat_sim[layer], 1).flatten(1)
            f_r = F.adaptive_avg_pool2d(feat_real[layer], 1).flatten(1)
            
            cos_sim_val = F.cosine_similarity(f_s, f_r, dim=1).mean().item()
            self.metric_cos_sim[layer].update(cos_sim_val, B)
            
            # Store numpy arrays for SVM / t-SNE later
            self.saved_features['sim'][layer].append(f_s.detach().cpu().numpy())
            self.saved_features['real'][layer].append(f_r.detach().cpu().numpy())
            
        # 3. Update Cross-Domain Consistency Metrics for Land Cover
        for task_name, num_classes in self.seg_tasks.items():
            self._update_consistency_segmentation(preds_sim, preds_real, B, task_name, num_classes)

    def _update_domain_metrics(self, preds, true_imgs, true_climate, true_lat, true_lon, true_coords_enc, meters, B):
        """Internal helper to calculate and track metrics for a specific domain dictionary."""
        # Climate
        if self.climate_weights is not None:
            l_climate = F.cross_entropy(preds['climate'], true_climate, weight=self.climate_weights)
        else:
            l_climate = F.cross_entropy(preds['climate'], true_climate)
        meters['loss_climate_ce'].update(l_climate.item(), B)
        
        # Geolocation
        l_geo = F.mse_loss(preds['coords'], true_coords_enc)
        meters['loss_geo_mse'].update(l_geo.item(), B)

        # Reconstruction Metrics (Dynamic Data Range)
        current_data_range = true_imgs.max() - true_imgs.min()
        if current_data_range < 1e-4:
             current_data_range = torch.tensor(1e-4, device=self.device)
             
        ssim_val = ssim_func(preds['reconstruction'], true_imgs, data_range=current_data_range.item())
        psnr_val = psnr_func(preds['reconstruction'], true_imgs, data_range=current_data_range.item())
        
        meters['metric_ssim'].update(ssim_val.item(), B)
        meters['metric_psnr'].update(psnr_val.item(), B)
        
        if self.use_lpips:
            rgb_indices = [2, 1, 0] # R, G, B order
            rgb_pred = torch.clamp(preds['reconstruction'][:, rgb_indices, :, :] / 3.0, -1.0, 1.0)
            rgb_true = torch.clamp(true_imgs[:, rgb_indices, :, :] / 3.0, -1.0, 1.0)
            lpips_val = self.lpips_metric(rgb_pred, rgb_true)
            meters['metric_lpips'].update(lpips_val.item(), B)
        
        # Task Metrics
        acc_val = self.acc_metric(preds['climate'], true_climate)
        f1_val = self.f1_metric(preds['climate'], true_climate)
        meters['metric_acc'].update(acc_val.item(), B)
        meters['metric_f1'].update(f1_val.item(), B)
        
        pred_classes = torch.argmax(preds['climate'], dim=1)
        macro_preds = self.macro_mapping[pred_classes]
        macro_true = self.macro_mapping[true_climate]
        
        macro_acc_val = self.macro_acc_metric(macro_preds, macro_true)
        macro_f1_val = self.macro_f1_metric(macro_preds, macro_true)
        
        meters['metric_macro_acc'].update(macro_acc_val.item(), B)
        meters['metric_macro_f1'].update(macro_f1_val.item(), B)
        
        dist_km = self.haversine_distance(preds['coords'], true_lat, true_lon).mean()
        meters['metric_dist'].update(dist_km.item(), B)

    @torch.no_grad()
    def _update_consistency_segmentation(self, preds_sim, preds_real, batch_size, task_name, num_classes):
        
        if task_name not in preds_sim or task_name not in preds_real:
            return
            
        logits_sim = preds_sim[task_name]
        logits_real = preds_real[task_name]
        
        # --- Hard Consistency (mIoU) ---
        labels_sim = torch.argmax(logits_sim, dim=1)
        labels_real = torch.argmax(logits_real, dim=1)
        
        miou_consistency = jaccard_index(
            preds=labels_real,
            target=labels_sim,
            task="multiclass",
            num_classes=num_classes,
            average='macro'
        )
        self.consistency_meters[f'{task_name}_consistency_miou'].update(miou_consistency.item(), batch_size)
        
        # --- Soft Consistency (KL Divergence) ---
        p_sim = F.softmax(logits_sim, dim=1)
        log_p_sim = F.log_softmax(logits_sim, dim=1)
        log_p_real = F.log_softmax(logits_real, dim=1)
        
        kl_pixel_wise = torch.sum(p_sim * (log_p_sim - log_p_real), dim=1)
        kl = kl_pixel_wise.mean()
        
        self.consistency_meters[f'{task_name}_consistency_kl'].update(kl.item(), batch_size)
    
    def print_report(self, title="Multi-Domain Evaluation Report"):
        """Prints a unified side-by-side style report for Domain Adaptation."""
        print(f"\n{title.center(70)}")
        print(f"{'='*70}")
        
        print(f"\n[ Domain Gap (Multi-Scale) ]")
        for layer in self.feature_layers:
            pad = f"{self.pad_scores.get(layer, 'N/A'):.4f}" if layer in self.pad_scores else "N/A"
            print(f"  {layer.upper():<12} | Cosine Sim: {self.metric_cos_sim[layer].avg:.4f} | PAD Score: {pad}")
        
        domains = [("SOURCE (SIM)", self.sim_meters), ("TARGET (REAL)", self.real_meters)]
        
        for dom_name, meters in domains:
            print(f"\n--- {dom_name} ---")
            print("  Reconstruction:")
            print(f"    SSIM: {meters['metric_ssim'].avg:.4f} | PSNR: {meters['metric_psnr'].avg:.2f} dB")
            if self.use_lpips:
                print(f"    LPIPS (RGB): {meters['metric_lpips'].avg:.4f}")
                
            print("  Climate Zone:")
            print(f"    CE Loss: {meters['loss_climate_ce'].avg:.4f}")
            print(f"    Fine (31)  -> Acc: {meters['metric_acc'].avg:.4f} | F1: {meters['metric_f1'].avg:.4f}")
            print(f"    Macro (6)  -> Acc: {meters['metric_macro_acc'].avg:.4f} | F1: {meters['metric_macro_f1'].avg:.4f}")
            
            print("  Geolocation:")
            print(f"    MSE Loss: {meters['loss_geo_mse'].avg:.4f} | Mean Error: {meters['metric_dist'].avg:.2f} km")

        if hasattr(self, 'consistency_meters') and len(self.consistency_meters) > 0:
            print(f"\n[ Cross-Domain Consistency (SIM vs REAL) ]")
            for task in self.seg_tasks.keys():
                miou_meter = self.consistency_meters.get(f'{task}_consistency_miou')
                kl_meter = self.consistency_meters.get(f'{task}_consistency_kl')
                
                if miou_meter and miou_meter.count > 0:
                    print(f"  {task.upper().replace('_', ' ')} Target:")
                    print(f"    mIoU (Hard)   : {miou_meter.avg:.4f}")
                    print(f"    KL Div (Soft) : {kl_meter.avg:.4f}")

        print(f"{'='*70}\n")
        
    def to_dict(self):
        report = {
            "domain_gap": {
                "cosine_similarity": {layer: float(self.metric_cos_sim[layer].avg) for layer in self.feature_layers},
            },
            "source_sim": self._extract_domain_metrics(self.sim_meters),
            "target_real": self._extract_domain_metrics(self.real_meters)
        }
        
        if self.pad_scores:
            report["domain_gap"]["pad_scores"] = {layer: float(score) for layer, score in self.pad_scores.items()}
            
        if hasattr(self, 'consistency_meters') and len(self.consistency_meters) > 0:
            report["cross_domain_consistency"] = {
                k: float(v.avg) for k, v in self.consistency_meters.items() if v.count > 0
            }
            
        return report

    def _extract_domain_metrics(self, meters):
        domain_dict = {
            "reconstruction": {
                "ssim": float(meters['metric_ssim'].avg),
                "psnr_db": float(meters['metric_psnr'].avg)
            },
            "climate_zone": {
                "ce_loss": float(meters['loss_climate_ce'].avg),
                "fine_31_acc": float(meters['metric_acc'].avg),
                "fine_31_f1": float(meters['metric_f1'].avg),
                "macro_6_acc": float(meters['metric_macro_acc'].avg),
                "macro_6_f1": float(meters['metric_macro_f1'].avg)
            },
            "geolocation": {
                "mse_loss": float(meters['loss_geo_mse'].avg),
                "mean_error_km": float(meters['metric_dist'].avg)
            }
        }
        if self.use_lpips:
            domain_dict["reconstruction"]["lpips_rgb"] = float(meters['metric_lpips'].avg)
            
        return domain_dict

    def save_report(self, filepath):
        import os
        import json
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        report_dict = self.to_dict()
        with open(filepath, 'w') as f:
            json.dump(report_dict, f, indent=4)
        print(f"[INFO] Report successfully saved to: {filepath}")