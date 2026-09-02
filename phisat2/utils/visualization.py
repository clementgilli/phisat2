import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE

SEGMENTATION_METADATA = {
    "clouds": {
        0:  ("Clear",          (46, 139, 87)),
        1:  ("Thick Cloud",    (255, 255, 255)),
        2:  ("Thin Cloud",     (0, 255, 255)),  
        3:  ("Cloud Shadow",   (255, 0, 255)),
        99: ("No Data",        (0, 0, 0)),
    },
    
    "floods": {
        0: ("Invalid",        (0, 0, 0)), 
        1: ("Land",           (139, 69, 19)),
        2: ("Water",          (30, 144, 255)),
        3: ("Clouds",         (211, 211, 211)),
    },
    
    "lulc": {
        0:  ("No Data",            (0, 0, 0)),    
        1:  ("Water",              (0, 0, 255)),   
        2:  ("Trees",              (34, 139, 34)),   
        3:  ("Grass",              (124, 252, 0)),   
        4:  ("Flooded Vegetation", (0, 139, 139)),    
        5:  ("Crops",              (255, 215, 0)),  
        6:  ("Scrub",              (184, 134, 11)), 
        7:  ("Built Area",         (255, 0, 0)),     
        8:  ("Bare Ground",        (210, 180, 140)),  
        9:  ("Snow/Ice",           (240, 248, 255)),  
        10: ("Cloud",              (211, 211, 211)),
    },
    
    "lulc_macro": {
        0:  ("No Data / Cloud", (0, 0, 0)),       # 0: No Data
        1:  ("Water",           (0, 0, 255)),     # 1: Water
        2:  ("Vegetation",      (34, 139, 34)),   # 2: Trees
        3:  ("Vegetation",      (34, 139, 34)),   # 3: Grass
        4:  ("Water",           (0, 0, 255)),     # 4: Flooded Vegetation
        5:  ("Vegetation",      (34, 139, 34)),   # 5: Crops
        6:  ("Vegetation",      (34, 139, 34)),   # 6: Scrub
        7:  ("Built Area",      (255, 0, 0)),     # 7: Built Area
        8:  ("Bare/Ice",        (210, 180, 140)), # 8: Bare Ground
        9:  ("Bare/Ice",        (210, 180, 140)), # 9: Snow/Ice
        10: ("No Data / Cloud", (0, 0, 0)),       # 10: Cloud
    },
    
    "marine": {
        0: ("No Data",      (0, 0, 0)),
        1: ("Clear Water",  (30, 144, 255)),
        2: ("Turbid Water", (189, 183, 107)),
        3: ("Land",         (139, 69, 19)),
        4: ("Plastic",      (255, 20, 147)),  
        5: ("Oil",          (47, 79, 79)),   
        6: ("Algae",        (50, 205, 50)),  
        7: ("Sediments",    (210, 180, 140)),
        8: ("Cloud",        (255, 255, 255)),
    },
    
    "methane": {
        0: ("No Plume", (0, 0, 0)),
        1: ("Plume",    (255, 69, 0)),
    },
    
    "burned": {
        0:   ("Clear",        (34, 139, 34)),
        1:   ("Fresh Burn",   (255, 0, 0)),
        2:   ("Old Burn",     (139, 69, 19)),
        3:   ("Cloud",        (255, 255, 255)),
        4:   ("Cloud Shadow", (105, 105, 105)),
        5:   ("Water",        (30, 144, 255)),
        255: ("No Data",      (0, 0, 0))
    },
    
    "eurosat": {
        0: ("AnnualCrop",           (230, 230, 0)),   
        1: ("Forest",               (34, 139, 34)),   
        2: ("HerbaceousVegetation", (144, 238, 144)), 
        3: ("Highway",              (105, 105, 105)), 
        4: ("Industrial",           (220, 20, 60)),     
        5: ("Pasture",              (173, 255, 47)),   
        6: ("PermanentCrop",        (218, 165, 32)),  
        7: ("Residential",          (139, 69, 19)),   
        8: ("River",                (0, 191, 255)),   
        9: ("SeaLake",              (0, 0, 128)),     
    },
    
    "router": {
        0: ("Safe",   (34, 139, 34)),
        1: ("Fire",   (255, 69, 0)),     
        2: ("Burnt",  (105, 105, 105)), 
        3: ("Water",  (30, 144, 255)), 
        4: ("Clouds", (220, 220, 220)),
    }
}

def mask_to_rgb(mask_2d: np.ndarray, dataset_name: str):
    meta = SEGMENTATION_METADATA.get(dataset_name, {})
    rgb_img = np.zeros((mask_2d.shape[0], mask_2d.shape[1], 3), dtype=np.uint8)
    for class_idx, (name, color) in meta.items():
        rgb_img[mask_2d == class_idx] = color
    return rgb_img, meta

def plot_tsne(X_sim, X_real, figsize=(10, 8), save_path=None):
    
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
    
def get_text_color(rgb_tuple):
    lum = 0.299 * rgb_tuple[0] + 0.587 * rgb_tuple[1] + 0.114 * rgb_tuple[2]
    return "black" if lum > 150 else "white"

def _extract_rgb(image_array, rgb_idx=(3, 2, 1)):
    rgb = image_array[list(rgb_idx), :, :].transpose(1, 2, 0)
    p2, p98 = np.percentile(rgb, (2, 98))
    if p98 > p2:
        rgb = np.clip((rgb - p2) / (p98 - p2), 0, 1)
    else:
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    return rgb

def visualize_downstream_demo(images, targets, preds, task_type, dataset_name, max_samples=5, rgb_idx=(3, 2, 1)):
    
    n = min(max_samples, images.shape[0])
    has_gt = targets is not None
    
    if task_type == "segmentation":
        is_lulc = (dataset_name == "lulc")
        
        if is_lulc:
            num_cols = 5 if has_gt else 3
            col_titles = ["Original (RGB)", "Micro GT", "Micro Pred", "Macro GT", "Macro Pred"] if has_gt else ["Original (RGB)", "Micro Pred", "Macro Pred"]
        else:
            num_cols = 3 if has_gt else 2
            col_titles = ["Original (RGB)", "Ground Truth", "Prediction"] if has_gt else ["Original (RGB)", "Prediction"]
            
        fig, axes = plt.subplots(n, num_cols, figsize=(5 * num_cols, 4 * n), squeeze=False, constrained_layout=True)
        
        current_meta_micro, current_meta_macro = None, None
        
        for i in range(n):
            axes[i, 0].imshow(_extract_rgb(images[i], rgb_idx))
            
            pr_rgb, current_meta_micro = mask_to_rgb(preds[i], dataset_name)
            
            if has_gt:
                gt_rgb, _ = mask_to_rgb(targets[i], dataset_name)
                axes[i, 1].imshow(gt_rgb)
                axes[i, 2].imshow(pr_rgb)
                
                if is_lulc:
                    gt_ma, current_meta_macro = mask_to_rgb(targets[i], "lulc_macro")
                    pr_ma, _ = mask_to_rgb(preds[i], "lulc_macro")
                    axes[i, 3].imshow(gt_ma)
                    axes[i, 4].imshow(pr_ma)
            else:
                axes[i, 1].imshow(pr_rgb)
                if is_lulc:
                    pr_ma, current_meta_macro = mask_to_rgb(preds[i], "lulc_macro")
                    axes[i, 2].imshow(pr_ma)
        
        if current_meta_micro:
            p_micro = [mpatches.Patch(color=np.array(c)/255.0, label=n_) for _, (n_, c) in current_meta_micro.items()]
            fig.legend(handles=p_micro, loc='lower center', bbox_to_anchor=(0.3 if is_lulc else 0.5, -0.05), ncol=5, title="Micro Classes")
            if is_lulc and current_meta_macro:
                unique_ma = {n_: c for n_, c in current_meta_macro.values()}
                p_macro = [mpatches.Patch(color=np.array(c)/255.0, label=n_) for n_, c in unique_ma.items()]
                fig.legend(handles=p_macro, loc='lower center', bbox_to_anchor=(0.75, -0.05), ncol=4, title="Macro Classes")

    elif task_type == "pixel_regression":
        num_cols = 3 if has_gt else 2
        col_titles = ["Original (RGB)", "Ground Truth", "Prediction"] if has_gt else ["Original (RGB)", "Prediction"]
        
        if has_gt:
            vmin = min(np.percentile(targets, 2), np.percentile(preds, 2))
            vmax = max(np.percentile(targets, 98), np.percentile(preds, 98))
        else:
            vmin, vmax = np.percentile(preds, 2), np.percentile(preds, 98)
            
        fig, axes = plt.subplots(n, num_cols, figsize=(5 * num_cols, 4 * n), squeeze=False, constrained_layout=True)
        
        for i in range(n):
            axes[i, 0].imshow(_extract_rgb(images[i], rgb_idx))
            if has_gt:
                axes[i, 1].imshow(targets[i], vmin=vmin, vmax=vmax, cmap="viridis")
                im = axes[i, 2].imshow(preds[i], vmin=vmin, vmax=vmax, cmap="viridis")
            else:
                im = axes[i, 1].imshow(preds[i], vmin=vmin, vmax=vmax, cmap="viridis")
                
        fig.colorbar(im, ax=axes[:, num_cols - 1], shrink=0.7)

    elif task_type == "classification":
        num_cols = 3 if has_gt else 2
        col_titles = ["Original (RGB)", "Ground Truth", "Prediction"] if has_gt else ["Original (RGB)", "Prediction"]
        
        fig, axes = plt.subplots(n, num_cols, figsize=(4 * num_cols, 4 * n), squeeze=False, constrained_layout=True)
        
        meta = None
        for i in range(n):
            axes[i, 0].imshow(_extract_rgb(images[i], rgb_idx))
            
            cols_to_plot = [(1, int(targets[i])), (2, int(preds[i]))] if has_gt else [(1, int(preds[i]))]
            
            for col, val in cols_to_plot:
                img, meta = mask_to_rgb(np.full((128,128), val), dataset_name)
                axes[i, col].imshow(img)
                name, color = meta.get(val, ("?", (0,0,0)))
                axes[i, col].text(64, 64, name, ha="center", va="center", fontweight="bold", color=get_text_color(color))
        
        if meta:
            p = [mpatches.Patch(color=np.array(c)/255.0, label=n_) for _, (n_, c) in meta.items()]
            fig.legend(handles=p, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=len(meta))

    for ax_row in axes:
        for ax, title in zip(ax_row, col_titles):
            ax.set_axis_off()
            if ax_row is axes[0]: ax.set_title(title, fontweight="bold", fontsize=14)
            
    suffix = "" if has_gt else "(Inference Only)"
    fig.suptitle(f"Client Demo: {dataset_name.upper()} {suffix}", fontsize=16, fontweight="bold")
    
    return fig

def visualize_triplet_batch(
    batch: dict[str, torch.Tensor | list], 
    idx: int = 0, 
    show_cloud: bool = True, 
    show_worldcover: bool = True, 
    vis_indices: tuple[int, ...] = (3, 2, 1),
    ir_indices: tuple[int, int, int, int] = (4, 5, 6, 7), # RE1, RE2, RE3, NIR
    zoom_box: tuple[int, int, int, int] | None = None
) -> None:
    """
    zoom_box: (xmin, ymin, xmax, ymax)
    """
    
    modalities = [
        ("Sentinel-2", 'sentinel2'),
        ("Simulated PhiSat-2", 'simulated'), 
        ("Real PhiSat-2", 'real'),
        ("Real PhiSat-2 (L0)", 'real_L0')
    ]
    
    n_cols = len(modalities) + int(show_cloud) + int(show_worldcover)
    fig, axes = plt.subplots(nrows=3, ncols=n_cols, figsize=(4 * n_cols, 12))
    
    if axes.ndim == 1:
        axes = np.expand_dims(axes, axis=0)

    tile_id = batch['tile_id'][idx] if 'tile_id' in batch else "Unknown"
    ref_shape = batch['simulated'][idx].shape[-2:] if 'simulated' in batch else (224, 224)

    def crop_tensor(tensor: torch.Tensor, key: str) -> torch.Tensor:
        if zoom_box is None:
            return tensor
        xmin, ymin, xmax, ymax = zoom_box
        
        if key == 'sentinel2':
            h_s2, w_s2 = tensor.shape[-2:]
            sy, sx = h_s2 / ref_shape[0], w_s2 / ref_shape[1]
            xmin, xmax = int(xmin * sx), int(xmax * sx)
            ymin, ymax = int(ymin * sy), int(ymax * sy)
            
        if tensor.ndim == 3:
            return tensor[:, ymin:ymax, xmin:xmax]
        return tensor[ymin:ymax, xmin:xmax]

    def get_stretched_image(tensor: torch.Tensor) -> tuple[np.ndarray, bool]:
        img_data = tensor[list(vis_indices)].detach().cpu().float().numpy()
        
        if len(vis_indices) == 3:
            img_data = np.transpose(img_data, (1, 2, 0))
            p2, p98 = np.percentile(img_data, (2, 98), axis=(0, 1))
            is_gray = False
        else: 
            img_data = img_data[0] # (1, H, W) -> (H, W)
            p2, p98 = np.percentile(img_data, (2, 98))
            is_gray = True
            
        p98 = np.maximum(p98, p2 + 1e-5) 
        img_stretched = np.clip((img_data - p2) / (p98 - p2), 0, 1)
        return img_stretched, is_gray

    def plot_custom_histogram(ax, tensor: torch.Tensor, indices: list[int], colors: list[str], labels: list[str], title: str) -> None:
        for idx_band, color, label in zip(indices, colors, labels):
            data = tensor[idx_band].detach().cpu().float().numpy().flatten()
            ax.hist(
                data, 
                bins=50, 
                color=color, 
                alpha=0.4, 
                histtype='stepfilled', 
                label=label
            )
            
        ax.set_title(title, fontsize=11)
        ax.set_yticks([])
        ax.legend(loc='upper right', fontsize=8)

    for row in range(1, 3):
        for col in range(len(modalities), n_cols):
            axes[row, col].axis("off")

    for col, (title, key) in enumerate(modalities):
        ax_img = axes[0, col]
        if key in batch:
            cropped_tensor = crop_tensor(batch[key][idx], key)
            img_disp, is_gray = get_stretched_image(cropped_tensor)
            
            if is_gray:
                ax_img.imshow(img_disp, cmap='gray')
            else:
                ax_img.imshow(img_disp)
        else:
            ax_img.text(0.5, 0.5, 'Missing\nModality', ha='center', va='center', color='gray')
            
        ax_img.set_title(title, fontsize=12, fontweight="bold")
        ax_img.axis("off")

    current_col = len(modalities)
    
    if show_cloud and 'mask_cloud' in batch:
        ax_cloud = axes[0, current_col]
        cloud_mask = crop_tensor(batch['mask_cloud'][idx].squeeze(), 'mask_cloud').detach().cpu().numpy()
        
        cloud_rgb, cloud_meta = mask_to_rgb(cloud_mask, "clouds")
        ax_cloud.imshow(cloud_rgb)
        ax_cloud.set_title("Cloud Mask", fontsize=12, fontweight="bold")
        ax_cloud.axis("off")
        
        unique_classes = np.unique(cloud_mask)
        patches = [
            mpatches.Patch(color=np.array(cloud_meta[c][1])/255.0, label=cloud_meta[c][0]) 
            for c in unique_classes if c in cloud_meta
        ]
        if patches:
            axes[1, current_col].legend(handles=patches, loc='center', fontsize=10)
            axes[1, current_col].set_title("Clouds Legend", fontsize=11)
            
        current_col += 1

    if show_worldcover and 'mask_worldcover' in batch:
        ax_wc = axes[0, current_col]
        wc_mask = crop_tensor(batch['mask_worldcover'][idx].squeeze(), 'mask_worldcover').detach().cpu().numpy()
        
        wc_to_dw = torch.tensor([2, 6, 3, 5, 7, 8, 9, 1, 4, 4, 0])
        wc_mask = wc_to_dw[wc_mask]
        
        wc_rgb, wc_meta = mask_to_rgb(wc_mask, "lulc")
        ax_wc.imshow(wc_rgb)
        ax_wc.set_title("WorldCover Mask", fontsize=12, fontweight="bold")
        ax_wc.axis("off")
        
        unique_classes = np.unique(wc_mask)
        patches = [
            mpatches.Patch(color=np.array(wc_meta[c][1])/255.0, label=wc_meta[c][0]) 
            for c in unique_classes if c in wc_meta
        ]
        if patches:
            axes[1, current_col].legend(handles=patches, loc='center', fontsize=10)
            axes[1, current_col].set_title("WorldCover Legend", fontsize=11)
            
        current_col += 1

    for col, (title, key) in enumerate(modalities):
        ax_hist = axes[1, col]
        if key in batch:
            cropped_tensor = crop_tensor(batch[key][idx], key)
            
            if len(vis_indices) == 3:
                plot_custom_histogram(
                    ax_hist, cropped_tensor, 
                    indices=list(vis_indices), 
                    colors=['red', 'green', 'blue'], 
                    labels=['Red', 'Green', 'Blue'], 
                    title=f"Vis Hist: {title}"
                )
            else:
                plot_custom_histogram(
                    ax_hist, cropped_tensor, 
                    indices=list(vis_indices), 
                    colors=['gray'], 
                    labels=[f'Band {vis_indices[0]}'], 
                    title=f"Band {vis_indices[0]} Hist: {title}"
                )

    for col, (title, key) in enumerate(modalities):
        ax_hist = axes[2, col]
        if key in batch:
            cropped_tensor = crop_tensor(batch[key][idx], key)
            plot_custom_histogram(
                ax_hist, cropped_tensor, 
                indices=list(ir_indices), 
                colors=['orange', 'magenta', 'purple', 'black'], 
                labels=['RE1', 'RE2', 'RE3', 'NIR'], 
                title=f"IR Hist: {title}"
            )

    plt.tight_layout()
    plt.show()