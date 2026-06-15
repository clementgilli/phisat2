import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

SEGMENTATION_METADATA = {
    "floods": {
        0: ("Cloud",      (211, 211, 211)), 
        1: ("Clear Land", (139, 69, 19)),
        2: ("Water",      (30, 144, 255)),   
    },
    "lulc": {
        0:  ("Tree Cover",          (34, 139, 34)),    
        1:  ("Shrubland",           (184, 134, 11)),   
        2:  ("Grassland",           (124, 252, 0)),   
        3:  ("Cropland",            (255, 215, 0)),   
        4:  ("Built-up",            (255, 0, 0)),    
        5:  ("Bare/Sparse Veg",     (210, 180, 140)),  
        6:  ("Snow and Ice",        (255, 255, 255)), 
        7:  ("Permanent Water",     (0, 0, 255)),     
        8:  ("Herbaceous Wetland",  (0, 139, 139)),  
        9:  ("Mangroves",           (32, 178, 170)),  
        10: ("Moss and Lichen",     (240, 230, 140)),
    },
    "lulc_macro": {
        0:  ("Vegetation", (34, 139, 34)),    # Tree Cover
        1:  ("Vegetation", (34, 139, 34)),    # Shrubland
        2:  ("Vegetation", (34, 139, 34)),    # Grassland
        3:  ("Vegetation", (34, 139, 34)),    # Cropland
        10: ("Vegetation", (34, 139, 34)),    # Moss and Lichen
        4:  ("Built-up",   (255, 0, 0)),      # Built-up        
        5:  ("Bare/Ice",   (210, 180, 140)),  # Bare/Sparse Veg
        6:  ("Bare/Ice",   (210, 180, 140)),  # Snow and Ice        
        7:  ("Water",      (0, 0, 255)),      # Permanent Water
        8:  ("Water",      (0, 0, 255)),      # Herbaceous Wetland
        9:  ("Water",      (0, 0, 255)),      # Mangroves
    },
    "burned": {
        0: ("Background",      (0, 0, 0)), 
        1: ("Burned Area", (255, 0, 0)),
        2: ("Clouds",      (211, 211, 211)),
        3: ("Waterbodies",      (30, 144, 255)),      
    },
    "clouds": {
        0: ("Background",      (0, 0, 0)), 
        1: ("Burned Area", (255, 0, 0)),
        2: ("Clouds",      (211, 211, 211)),
        3: ("Waterbodies",      (30, 144, 255)),      
    },
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