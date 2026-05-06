import math
import json
import pandas as pd
import numpy as np
import rasterio
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ==========================================
# CONFIGURATION
# ==========================================
BAND_REAL = 3 #RED
BAND_SIM = 2 #RED          #yes be carefull with the order of bands in the .tif files for both Sentinel-2B and simulated PhiSat-2 (see main.py fine_registration for details)
BAND_S2B = 4 #RED

# Colormaps
cmap_clouds_s2b = mcolors.ListedColormap(['none', 'red', 'orange', 'cyan'])
bounds_s2b = [-0.5, 0.5, 1.5, 2.5, 3.5]
norm_s2b = mcolors.BoundaryNorm(bounds_s2b, cmap_clouds_s2b.N)

# PhiSat-2 Cloud Mask Colormap (Assuming 1=Thick, 2=Thin)
cmap_clouds_phi = mcolors.ListedColormap(['none', 'magenta', 'pink'])
bounds_phi = [-0.5, 0.5, 1.5, 2.5]
norm_phi = mcolors.BoundaryNorm(bounds_phi, cmap_clouds_phi.N)

# ==========================================
# FUNCTIONS
# ==========================================
def load_band(path, band_idx, flip_h=False):
    with rasterio.open(path) as src:
        data = src.read(band_idx).astype(np.float32)
    if flip_h:
        data = cv2.flip(data, 1)
    return data

def to_vis(img):
    v = np.percentile(img[img > 0], (2, 98))
    if v[1] == v[0]: return np.zeros_like(img)
    return np.clip((img - v[0]) / (v[1] - v[0]), 0, 1)

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in kilometers
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    
    a = (math.sin(d_lat / 2)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon / 2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c

def generate_geojson_polygon(df, product_id):
    row = df[df['product_id'] == product_id]
    
    if len(row) == 0:
        print(f"Error: No entry found for product_id {product_id}. Cannot generate GeoJSON.")
        return
    
    r = row.iloc[0]
    
    # Coordinates for polygons
    new_coords = [
        [float(r['ul_lon']), float(r['ul_lat'])],
        [float(r['ur_lon']), float(r['ur_lat'])],
        [float(r['lr_lon']), float(r['lr_lat'])],
        [float(r['ll_lon']), float(r['ll_lat'])],
        [float(r['ul_lon']), float(r['ul_lat'])]  
    ]
    
    orig_coords = [
        [float(r['phisat2_orig_ul_lon']), float(r['phisat2_orig_ul_lat'])],
        [float(r['phisat2_orig_ur_lon']), float(r['phisat2_orig_ur_lat'])],
        [float(r['phisat2_orig_lr_lon']), float(r['phisat2_orig_lr_lat'])],
        [float(r['phisat2_orig_ll_lon']), float(r['phisat2_orig_ll_lat'])],
        [float(r['phisat2_orig_ul_lon']), float(r['phisat2_orig_ul_lat'])]  
    ]
    
    new_center = [float(r['center_lon']), float(r['center_lat'])]
    orig_center = [float(r['phisat2_orig_center_lon']), float(r['phisat2_orig_center_lat'])]
    
    shift_km = haversine_distance(orig_center[1], orig_center[0], new_center[1], new_center[0])
    
    print(f"Product ID: {product_id}")
    print(f"Pointing Error (Shift Distance): {shift_km:.2f} km\n")
    
    geojson_dict = {
      "type": "FeatureCollection",
      "features": [
        {
          "type": "Feature",
          "properties": {
            "title": "NEW - Corrected Footprint",
            "product_id": int(r['product_id']),
            "stroke": "#0000FF",  
            "stroke-width": 3,
            "fill": "#0000FF",
            "fill-opacity": 0.3
          },
          "geometry": {
            "type": "Polygon",
            "coordinates": [new_coords]
          }
        },
        {
          "type": "Feature",
          "properties": {
            "title": "ORIGINAL - PhiSat-2 Footprint",
            "product_id": int(r['product_id']),
            "stroke": "#FF0000", 
            "stroke-width": 3,
            "fill": "#FF0000",
            "fill-opacity": 0.1
          },
          "geometry": {
            "type": "Polygon",
            "coordinates": [orig_coords]
          }
        }
      ]
    }
    
    print(json.dumps(geojson_dict, indent=2))
    
    
def visualize_product(df, product_id, generate_geojson=False):
    
    product = df[df['product_id'] == product_id].iloc[0]
    print(f"Visualizing Product ID: {product_id}")

    real_path = product['phisat2_path']
    sim_path = product['aligned_phisat2sim_path']
    s2b_path = product['aligned_s2b_path']
    mask_sim_path = product['aligned_mask_path']
    mask_real_path = product['phisat2_mask_path']

    # Load Images
    img_real = load_band(real_path, BAND_REAL, flip_h=True)
    h_ref, w_ref = img_real.shape
    img_sim = load_band(sim_path, BAND_SIM)
    img_s2b = load_band(s2b_path, BAND_S2B)

    # Load S2B Mask
    with rasterio.open(mask_sim_path) as src_mask:
        mask_s2b_data = src_mask.read(1)

    mask_s2b_mapped = np.zeros_like(mask_s2b_data, dtype=np.uint8)
    mask_s2b_mapped[(mask_s2b_data == 8) | (mask_s2b_data == 9)] = 1
    mask_s2b_mapped[mask_s2b_data == 10] = 2
    mask_s2b_mapped[mask_s2b_data == 3] = 3

    # Load PhiSat-2 Mask (Assuming OCM format)
    with rasterio.open(mask_real_path) as src_mask_phi:
        mask_phi_data_raw = src_mask_phi.read(1)
        
    mask_phi_data = cv2.resize(mask_phi_data_raw, (w_ref, h_ref), interpolation=cv2.INTER_NEAREST)
    mask_phi_data = cv2.flip(mask_phi_data, 1)

    mask_phi_mapped = np.zeros_like(mask_phi_data, dtype=np.uint8)
    mask_phi_mapped[mask_phi_data == 1] = 1 
    mask_phi_mapped[mask_phi_data == 2] = 2

    # ==========================================
    # PLOTTING
    # ==========================================
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    plt.subplots_adjust(wspace=0.05, hspace=0.1)

    # --- Top Row: Images ---
    axes[0, 0].imshow(to_vis(img_real), cmap='gray')
    axes[0, 0].set_title(f"Real: PhiSat-2 (RED) ({w_ref}x{h_ref})", fontsize=14)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(to_vis(img_sim), cmap='gray')
    axes[0, 1].set_title("Simulated: S2B -> PhiSat-2 (RED)", fontsize=14)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(to_vis(img_s2b), cmap='gray')
    axes[0, 2].set_title("Original: S2B Aligned (RED)", fontsize=14)
    axes[0, 2].axis('off')

    # --- Bottom Row: Masks & Check ---
    # Bottom Left: PhiSat-2 Mask Overlay
    axes[1, 0].imshow(to_vis(img_real), cmap='gray')
    axes[1, 0].imshow(mask_phi_mapped, cmap=cmap_clouds_phi, norm=norm_phi, alpha=0.5, interpolation='nearest')
    axes[1, 0].set_title("PhiSat-2 Cloud Mask (Orig)", fontsize=14)
    axes[1, 0].axis('off')

    # Bottom Center: S2B Mask Overlay
    axes[1, 1].imshow(to_vis(img_sim), cmap='gray')
    axes[1, 1].imshow(mask_s2b_mapped, cmap=cmap_clouds_s2b, norm=norm_s2b, alpha=0.5, interpolation='nearest')
    axes[1, 1].set_title("S2B Cloud Mask (Aligned)", fontsize=14)
    axes[1, 1].axis('off')

    # Bottom Right: False RGB Alignment Check
    vis_real = to_vis(img_real)
    vis_sim = to_vis(img_sim)
    vis_s2b = to_vis(img_s2b)
    false_rgb = np.stack([vis_real, vis_sim, vis_s2b], axis=2)

    axes[1, 2].imshow(false_rgb)
    axes[1, 2].set_title("Alignment Check (R=Real, G=Sim, B=S2B)", fontsize=14)
    axes[1, 2].axis('off')

    plt.suptitle(f"Co-registered Triplet & Masks for Product {product_id}", fontsize=20, y=0.95)
    plt.show()

    if generate_geojson:
        generate_geojson_polygon(df, product_id)