import torch
import torch.nn.functional as F
import numpy as np
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.functional.image import structural_similarity_index_measure as ssim_func
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr_func
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from src.core.utils import AverageMeter, set_seed
import json
import os
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

class MultiTaskEvaluator:
    def __init__(self, config, device='cuda', use_lpips=False):
        """
        Initializes the paired evaluation pipeline for Domain Adaptation.
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
        
        # --- Cross-Domain Meters ---
        self.metric_cos_sim = AverageMeter('Latent_Cosine_Sim')
        
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
        }
        if self.use_lpips:
            meters['metric_lpips'] = AverageMeter('Recon_LPIPS_RGB')
        return meters

    def reset(self):
        """Resets all meters across both domains and cross-domain metrics."""
        for meters in [self.sim_meters, self.real_meters]:
            for m in meters.values():
                m.reset()
        self.metric_cos_sim.reset()

    def haversine_distance(self, pred_coords, true_lat, true_lon):
        """
        Approximates distance in kilometers between predicted coordinates and ground truth.
        """
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
        
    def compute_PAD_score(self, X_sim, X_real):
        X_combined = np.vstack([X_sim, X_real])
        labels = np.array([0] * len(X_sim) + [1] * len(X_real))

        X_train, X_test, y_train, y_test = train_test_split(X_combined, labels, test_size=0.2, random_state=42, stratify=labels)

        cls = LinearSVC(dual="auto", max_iter=10000)
        cls.fit(X_train, y_train)

        y_pred = cls.predict(X_test)
        err = np.mean(y_pred != y_test)
        
        pad_score = 2 * (1 - 2 * err)
        self.pad_score = pad_score
        
        return pad_score

    @torch.no_grad()
    def evaluate_paired_batch(self, model_sim, batch, model_real=None, return_features=False, seed=42):
        """
        Evaluates both SIM and REAL domains simultaneously, computes Domain Gap metrics.
        Can accept a single model for both, or distinct models for asymmetric inference.
        """
        set_seed(seed) 
        
        if model_real is None:
            model_real = model_sim
            
        model_sim.eval()
        model_real.eval()
        
        B = batch['sim'].size(0)
        
        true_climate = batch['climate'].to(self.device)
        true_lat = batch['lat'].to(self.device)
        true_lon = batch['lon'].to(self.device)
        true_coords_enc = self.get_gt_coords_encoded(true_lat, true_lon).to(self.device)
        
        sim_imgs = batch['sim'].to(self.device)
        preds_sim = model_sim(sim_imgs)
        
        self._update_domain_metrics(
            preds_sim, sim_imgs, true_climate, true_lat, true_lon, true_coords_enc, self.sim_meters, B
        )
        
        real_imgs = batch['real'].to(self.device)
        preds_real = model_real(real_imgs)
        
        self._update_domain_metrics(
            preds_real, real_imgs, true_climate, true_lat, true_lon, true_coords_enc, self.real_meters, B
        )
        
        feat_sim = preds_sim.get('features', None)
        feat_real = preds_real.get('features', None)
        
        if feat_sim is not None and feat_real is not None:
            cos_sim_val = F.cosine_similarity(feat_sim, feat_real, dim=1).mean().item()
            self.metric_cos_sim.update(cos_sim_val, B)
            
        if return_features:
            return feat_sim, feat_real

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
            rgb_indices = [2, 1, 0] # corresponding to bands 1, 2, 3 in the original order (R, G, B)
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

    def print_report(self, title="Multi-Domain Evaluation Report"):
        """Prints a unified side-by-side style report for Domain Adaptation."""
        print(f"\n{title.center(70)}")
        print(f"{'='*70}")
        
        print(f"\n[ Domain Gap ]")
        print(f"  Latent Cosine Similarity : {self.metric_cos_sim.avg:.4f}")
        
        if hasattr(self, 'pad_score'):
            print(f"  PAD Score (SVM)          : {self.pad_score:.4f}")
        
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
            
        print(f"{'='*70}\n")
        
        
    def to_dict(self):
        report = {
            "domain_gap": {
                "latent_cosine_similarity": float(self.metric_cos_sim.avg)
            },
            "source_sim": self._extract_domain_metrics(self.sim_meters),
            "target_real": self._extract_domain_metrics(self.real_meters)
        }
        
        if hasattr(self, 'pad_score'):
            report["domain_gap"]["pad_score"] = float(self.pad_score)
            
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
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        report_dict = self.to_dict()
        with open(filepath, 'w') as f:
            json.dump(report_dict, f, indent=4)
            
        print(f"[INFO] Report successfully saved to: {filepath}")