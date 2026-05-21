import os
import yaml
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def set_seed(seed: int = 42):
    """
    Sets the seed for reproducibility across all libraries.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_yaml_config(config_path="configs/config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
class AverageMeter(object):
    """
    Computes and stores the average and current value of a metric.
    Extremely useful for tracking losses during the training loop.
    """
    def __init__(self, name='Metric'):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"

def compute_climate_weights(class_distribution_dict, num_classes=31):
    """
    Computes class weights inversely proportional to the square root of class frequencies.
    Safely handles missing classes in the dataset (assigns them a weight of 0.0).
    
    Args:
        class_distribution_dict (dict): Dictionary mapping class index to count.
        num_classes (int): Total expected number of classes by the model.
        
    Returns:
        torch.Tensor: Weights tensor of shape [num_classes] for CrossEntropyLoss.
    """
    counts = np.zeros(num_classes, dtype=np.float32)
    
    for class_idx, count in class_distribution_dict.items():
        if class_idx < num_classes:
            counts[class_idx] = count
            
    total_samples = np.sum(counts)
    valid_mask = counts > 0
    
    frequencies = np.zeros(num_classes, dtype=np.float32)
    frequencies[valid_mask] = counts[valid_mask] / total_samples
    
    weights = np.zeros(num_classes, dtype=np.float32)
    weights[valid_mask] = 1.0 / np.sqrt(frequencies[valid_mask])
    
    mean_weight = np.mean(weights[valid_mask])
    weights[valid_mask] = weights[valid_mask] / mean_weight
    
    return torch.tensor(weights, dtype=torch.float32)

def extract_latent_features(model, batch, device='cuda', pool=True):
    """
    Extracts latent features from the model's encoder for a given batch of SIM and REAL images.
    """
    model.eval()
    
    sim_imgs = batch['sim'].to(device)
    real_imgs = batch['real'].to(device)
    
    with torch.no_grad():
        # --- SIM (Source) ---
        sim_stem = model.stem(sim_imgs)
        sim_bottom, _ = model.encoder(sim_stem)
        sim_bottom_feats = model.bridge(sim_bottom)
        
        # --- REAL (Target) ---
        real_stem = model.stem(real_imgs)
        real_bottom, _ = model.encoder(real_stem)
        real_bottom_feats = model.bridge(real_bottom)
        
        if pool:
            feat_sim = model.pool_feats(sim_bottom_feats)
            feat_real = model.pool_feats(real_bottom_feats)
        else:
            feat_sim = sim_bottom_feats
            feat_real = real_bottom_feats
            
    return feat_sim, feat_real

def denormalize_tensor(tensor, mean_list, std_list):
    """
    Reverses the Z-score normalization for visualization purposes.
    
    Args:
        tensor (np.ndarray): The normalized tensor (C, H, W).
        mean_list (list): The list of channel means.
        std_list (list): The list of channel standard deviations.
        
    Returns:
        np.ndarray: The denormalized tensor.
    """
    mean = np.array(mean_list).reshape(-1, 1, 1)
    std = np.array(std_list).reshape(-1, 1, 1)
    
    denorm = (tensor * std) + mean
    denorm = np.square(np.maximum(denorm, 0))
    return denorm

import torch
import numpy as np
import matplotlib.pyplot as plt

def denormalize_tensor(tensor, mean_list, std_list):
    """
    Reverses the Z-score normalization for visualization purposes.
    
    Args:
        tensor (np.ndarray): The normalized tensor (C, H, W).
        mean_list (list): The list of channel means.
        std_list (list): The list of channel standard deviations.
        
    Returns:
        np.ndarray: The denormalized tensor.
    """
    mean = np.array(mean_list).reshape(-1, 1, 1)
    std = np.array(std_list).reshape(-1, 1, 1)
    
    denorm = (tensor * std) + mean
    denorm = np.square(np.maximum(denorm, 0))
    return denorm

def visualize_batch(batch, config, baseline_mode=1.0, model_sim=None, model_real=None, device='cuda', num_samples=3, band=0, show_histograms=False, show_reconstructions=False):
    """
    Visualizes a pair of SIM and REAL images from a DataLoader batch, 
    with optional reconstructions and histograms, taking into account the normalization baseline.
    
    Args:
        batch (dict): A batch dictionary generated by PairedPhiSatDataset.
        config (dict): The loaded YAML configuration dictionary.
        baseline_mode (float): Normalization mode used (0.0, 0.5, or 1.0).
        model_sim (torch.nn.Module, optional): Model used for forward pass on SIM data if show_reconstructions=True.
        model_real (torch.nn.Module, optional): Model used for forward pass on REAL data if show_reconstructions=True.
        device (str): Compute device ('cuda' or 'cpu').
        num_samples (int): Number of image pairs to display from the batch.
        band (int): The spectral band index to display.
        show_histograms (bool): Whether to plot pixel distributions.
        show_reconstructions (bool): Whether to display reconstructed images.
    """
    # 1. Determine dynamic layout dimensions
    num_cols = 2
    if show_reconstructions:
        assert model_sim is not None and model_real is not None, "Both model_sim and model_real must be provided to show reconstructions."
        num_cols += 2
    if show_histograms:
        num_cols += 1
        
    batch_size = batch['sim'].size(0)
    num_samples = min(num_samples, batch_size)
    
    # 2. Forward pass for reconstructions (processed batched for efficiency)
    if show_reconstructions:
        model_sim.eval()
        model_real.eval()
        with torch.no_grad():
            sim_imgs = batch['sim'].to(device)
            real_imgs = batch['real'].to(device)
            
            preds_sim = model_sim(sim_imgs)
            preds_real = model_real(real_imgs)
            
            sim_rec_batch = preds_sim['reconstruction'].cpu().numpy()
            real_rec_batch = preds_real['reconstruction'].cpu().numpy()

    # 3. STATS ROUTING: Determine correct stats based on baseline mode
    band_order = config["band_order"]
    
    sim_mean_orig = config["normalization"]["sim"]["mean"]
    sim_std_orig = config["normalization"]["sim"]["std"]
    
    sim_mean = np.array(sim_mean_orig)[band_order].tolist()
    sim_std = np.array(sim_std_orig)[band_order].tolist()
    
    if baseline_mode == 0.5:
        # Domain-based: REAL uses its own stats (also reordered)
        real_mean_orig = config["normalization"]["real"]["mean"]
        real_std_orig = config["normalization"]["real"]["std"]
        real_mean = np.array(real_mean_orig)[band_order].tolist()
        real_std = np.array(real_std_orig)[band_order].tolist()
    else:
        # Source-based (0.0) or OT (1.0): REAL is normalized with SIM stats
        real_mean = sim_mean
        real_std = sim_std

    # 4. Setup Figure
    fig, axes = plt.subplots(num_samples, num_cols, figsize=(4 * num_cols, 4 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)
        
    for i in range(num_samples):
        # --- Extract Inputs ---
        sim_np = batch['sim'][i].cpu().numpy()
        real_np = batch['real'][i].cpu().numpy()
        climate_gt = batch['climate'][i].item()
        
        # --- Denormalize Inputs with routed stats ---
        sim_vis = denormalize_tensor(sim_np, sim_mean, sim_std)
        real_vis = denormalize_tensor(real_np, real_mean, real_std)
        
        sim_band_vis = sim_vis[band]
        real_band_vis = real_vis[band]
        
        col_idx = 0
        
        # ---------------------------------------------------
        # A. SIM (Source) Input
        # ---------------------------------------------------
        ax_sim = axes[i, col_idx]
        vmin_sim, vmax_sim = np.percentile(sim_band_vis, [2, 98])
        ax_sim.imshow(sim_band_vis, cmap='gray', vmin=vmin_sim, vmax=vmax_sim)
        ax_sim.set_title(f"SIM (Source) - Band {band}\nClimate: {climate_gt}", fontsize=11)
        ax_sim.axis('off')
        col_idx += 1
        
        # ---------------------------------------------------
        # B. SIM Reconstruction (Optional)
        # ---------------------------------------------------
        if show_reconstructions:
            sim_rec_np = sim_rec_batch[i]
            sim_rec_vis = denormalize_tensor(sim_rec_np, sim_mean, sim_std)
            sim_rec_band_vis = sim_rec_vis[band]
            
            ax_sim_rec = axes[i, col_idx]
            ax_sim_rec.imshow(sim_rec_band_vis, cmap='gray', vmin=vmin_sim, vmax=vmax_sim)
            ax_sim_rec.set_title(f"SIM Recon - Band {band}", fontsize=11)
            ax_sim_rec.axis('off')
            col_idx += 1
            
        # ---------------------------------------------------
        # C. REAL (Target) Input
        # ---------------------------------------------------
        ax_real = axes[i, col_idx]
        vmin_real, vmax_real = np.percentile(real_band_vis, [2, 98])
        ax_real.imshow(real_band_vis, cmap='gray', vmin=vmin_real, vmax=vmax_real)
        ax_real.set_title(f"REAL (Target) - Band {band}\nClimate: {climate_gt}", fontsize=11)
        ax_real.axis('off')
        col_idx += 1
        
        # ---------------------------------------------------
        # D. REAL Reconstruction (Optional)
        # ---------------------------------------------------
        if show_reconstructions:
            real_rec_np = real_rec_batch[i]
            real_rec_vis = denormalize_tensor(real_rec_np, real_mean, real_std)
            real_rec_band_vis = real_rec_vis[band]
            
            ax_real_rec = axes[i, col_idx]
            ax_real_rec.imshow(real_rec_band_vis, cmap='gray', vmin=vmin_real, vmax=vmax_real)
            ax_real_rec.set_title(f"REAL Recon - Band {band}", fontsize=11)
            ax_real_rec.axis('off')
            col_idx += 1
            
        # ---------------------------------------------------
        # E. Histograms (Optional)
        # ---------------------------------------------------
        if show_histograms:
            ax_hist = axes[i, col_idx]
            
            sim_flat = sim_np[band].flatten()
            real_flat = real_np[band].flatten()
            
            min_val = min(np.min(sim_flat), np.min(real_flat))
            max_val = max(np.max(sim_flat), np.max(real_flat))
            bins = np.linspace(min_val, max_val, 50)
            
            # Plot Input Distributions (Filled)
            ax_hist.hist(sim_flat, bins=bins, alpha=0.4, color='red', label='SIM In', density=True)
            ax_hist.hist(real_flat, bins=bins, alpha=0.4, color='blue', label='REAL In', density=True)
            
            if show_reconstructions:
                sim_rec_flat = sim_rec_batch[i][band].flatten()
                real_rec_flat = real_rec_batch[i][band].flatten()
                ax_hist.hist(sim_rec_flat, bins=bins, histtype='step', color='red', label='SIM Rec', density=True, linewidth=1.5)
                ax_hist.hist(real_rec_flat, bins=bins, histtype='step', color='blue', label='REAL Rec', density=True, linewidth=1.5)
            
            ax_hist.set_title("Z-score Distribution", fontsize=11)
            ax_hist.set_xlabel("Normalized Value")
            ax_hist.set_ylabel("Density")
            ax_hist.legend(fontsize=8, loc='upper right')
            ax_hist.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()
    
def plot_tsne(X_sim, X_real, figsize=(10, 8), save_path=None):
    """
    Plots a t-SNE visualization of the latent features from SIM and REAL datasets.
    
    Args:        
        X_sim (np.ndarray): Latent features from the SIM dataset.
        X_real (np.ndarray): Latent features from the REAL dataset.
    """
    
    X_combined = np.vstack([X_sim, X_real])
    labels = np.array([0] * len(X_sim) + [1] * len(X_real))
    
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_combined)
    
    plt.figure(figsize=figsize)
    plt.scatter(X_tsne[labels==0, 0], X_tsne[labels==0, 1], alpha=0.5, label='SIM', color='red')
    plt.scatter(X_tsne[labels==1, 0], X_tsne[labels==1, 1], alpha=0.5, label='REAL', color='blue')
    plt.legend()
    plt.title("t-SNE of Latent Features")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()