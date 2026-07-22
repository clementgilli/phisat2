import os
import pandas as pd
import json
import rasterio
from rasterio.warp import transform as warp_transform
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
CSV_PHISAT2_META = "/shared/projects/phisat2/data/index/phisat2_metadata.csv"
CSV_PHISAT2_S2B = "/shared/projects/phisat2/data/index/phisat2_to_s2b.csv"
CSV_PHISAT2_CLOUD = "/shared/projects/phisat2/data/index/phisat2_cloud_stats_detailed.csv"
CSV_ALIGNMENT_LOG = "/shared/projects/phisat2/data/index/s2b_simulated_aligned_log.csv"

OUTPUT_MASTER_CSV = "/shared/projects/phisat2/data/processed/dataset_master_images_metadata.csv"

# ==========================================
# FONCTION
# ==========================================
def get_phisat2_orig_footprint(tif_path):
    
    keys = [
        'phisat2_orig_ul_lon', 'phisat2_orig_ul_lat', 
        'phisat2_orig_ur_lon', 'phisat2_orig_ur_lat', 
        'phisat2_orig_lr_lon', 'phisat2_orig_lr_lat', 
        'phisat2_orig_ll_lon', 'phisat2_orig_ll_lat',
        'phisat2_orig_center_lon', 'phisat2_orig_center_lat'
    ]
    
    try:
        with rasterio.open(tif_path) as src:
            w, h = src.width, src.height
            t = src.transform
            crs = src.crs

            xs = [t.c, t.c + w * t.a, t.c + w * t.a + h * t.b, t.c + h * t.b]
            ys = [t.f, t.f + w * t.d, t.f + w * t.d + h * t.e, t.f + h * t.e]

            lons, lats = warp_transform(crs, 'EPSG:4326', xs, ys)
            
            center_lon = sum(lons) / 4.0
            center_lat = sum(lats) / 4.0

            return {
                'phisat2_orig_ul_lon': round(lons[0], 6), 'phisat2_orig_ul_lat': round(lats[0], 6),
                'phisat2_orig_ur_lon': round(lons[1], 6), 'phisat2_orig_ur_lat': round(lats[1], 6),
                'phisat2_orig_lr_lon': round(lons[2], 6), 'phisat2_orig_lr_lat': round(lats[2], 6),
                'phisat2_orig_ll_lon': round(lons[3], 6), 'phisat2_orig_ll_lat': round(lats[3], 6),
                'phisat2_orig_center_lon': round(center_lon, 6), 'phisat2_orig_center_lat': round(center_lat, 6)
            }
    except Exception as e:
        print(f"Error for {tif_path} : {e}")
        return {k: None for k in keys}

# ==========================================
# MAIN
# ==========================================
def create_master_csv():
    df_align = pd.read_csv(CSV_ALIGNMENT_LOG)
    df_meta = pd.read_csv(CSV_PHISAT2_META)
    df_s2b = pd.read_csv(CSV_PHISAT2_S2B)
    df_cloud = pd.read_csv(CSV_PHISAT2_CLOUD)

    df_master = df_align[df_align['status'] == 'SUCCESS'].copy()
    
    cols_to_keep_align = ['product_id', 'inliers', 'aligned_phisat2sim_path', 'aligned_mask_path', 'aligned_s2b_path']
    df_master = df_master[cols_to_keep_align]
    
    df_master = pd.merge(df_master, df_meta[['product_id', 'folder_path', 'start_date']], on='product_id', how='left')
    df_master = pd.merge(df_master, df_s2b[['product_id', 's2_day', 'delta_days']], on='product_id', how='left')
    df_master = pd.merge(df_master, df_cloud[['product_id', 'thick_cloud_pct', 'thin_cloud_pct']], on='product_id', how='left')

    df_master = df_master.rename(columns={'start_date': 'phisat2_date', 's2_day': 's2b_date'})

    footprint_data = []
    
    for idx, row in tqdm(df_master.iterrows(), total=len(df_master), desc="Processing images"):
        product_id = row['product_id']
        
        # On va chercher les coords de l'alignement
        align_row = df_align[df_align['product_id'] == product_id].iloc[0]
        coords = align_row[["ul_lon", "ul_lat", "ur_lon", "ur_lat", "lr_lon", "lr_lat", "ll_lon", "ll_lat", "center_lon", "center_lat"]].to_dict()
        
        phisat2_real_path = os.path.join(row['folder_path'], "bands", "scene_0_BC_multiband.tiff")
        phisat2_mask_path = f"/shared/projects/phisat2/data/interim/phisat2_cloud_masks/{product_id}_phisat2_OCM_v1_7_1.tif"
        
        orig_coords = get_phisat2_orig_footprint(phisat2_real_path)
        
        # On met TOUT dans le dictionnaire
        coords.update(orig_coords)
        coords['product_id'] = product_id
        coords['phisat2_path'] = phisat2_real_path
        coords['phisat2_mask_path'] = phisat2_mask_path
        
        footprint_data.append(coords)

    # Création du DataFrame avec toutes nos nouvelles données calculées
    df_footprint = pd.DataFrame(footprint_data)
    
    # 🚨 LE FIX EST ICI : On merge PROPREMENT par product_id. Pas d'affectation directe !
    df_master = pd.merge(df_master, df_footprint, on='product_id', how='left')
    
    df_master = df_master.drop(columns=['folder_path'])

    final_columns = [
        'product_id', 
        'phisat2_date', 's2b_date', 'delta_days', 
        'center_lat', 'center_lon', 
        'ul_lat', 'ul_lon', 'ur_lat', 'ur_lon', 
        'lr_lat', 'lr_lon', 'll_lat', 'll_lon', 
        'phisat2_orig_center_lat', 'phisat2_orig_center_lon',
        'phisat2_orig_ul_lat', 'phisat2_orig_ul_lon', 'phisat2_orig_ur_lat', 'phisat2_orig_ur_lon', 
        'phisat2_orig_lr_lat', 'phisat2_orig_lr_lon', 'phisat2_orig_ll_lat', 'phisat2_orig_ll_lon', 
        'thick_cloud_pct', 'thin_cloud_pct', 'inliers', 
        'phisat2_path', 'phisat2_mask_path',  
        'aligned_phisat2sim_path', 'aligned_s2b_path', 'aligned_mask_path' 
    ]
    df_master = df_master[final_columns]

    os.makedirs(os.path.dirname(OUTPUT_MASTER_CSV), exist_ok=True)
    df_master.to_csv(OUTPUT_MASTER_CSV, index=False)
    
    print("\n" + "="*50)
    print(f"MASTER CSV GENERATED : {OUTPUT_MASTER_CSV}")
    print("="*50)


create_master_csv()