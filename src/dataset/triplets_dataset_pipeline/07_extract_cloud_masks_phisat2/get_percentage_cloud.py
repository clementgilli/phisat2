import os
import rasterio as rio
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def calculate_full_cloud_stats(row, mask_dir):
    product_id = row['product_id']
    
    mask_filename = f"{product_id}_phisat2_OCM_v1_7_1.tif"
    mask_path = os.path.join(mask_dir, mask_filename)
    
    stats = {
        'product_id': product_id,
        'clear_pct': np.nan,
        'thick_cloud_pct': np.nan,
        'thin_cloud_pct': np.nan,
        'shadow_pct': np.nan,
        'status': 'PENDING'
    }

    if not os.path.exists(mask_path):
        stats['status'] = 'MASK_NOT_FOUND'
        return stats

    try:
        with rio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR'):
            with rio.open(mask_path) as src:
                mask = src.read(1)
                total_pixels = mask.size
                
                counts = np.bincount(mask.flatten(), minlength=4)
                
                stats.update({
                    'clear_pct': round((counts[0] / total_pixels) * 100.0, 2),
                    'thick_cloud_pct': round((counts[1] / total_pixels) * 100.0, 2),
                    'thin_cloud_pct': round((counts[2] / total_pixels) * 100.0, 2),
                    'shadow_pct': round((counts[3] / total_pixels) * 100.0, 2),
                    'status': 'SUCCESS'
                })
        
        return stats
        
    except Exception as e:
        stats['status'] = f"ERROR: {str(e)}"
        return stats

if __name__ == "__main__":
    mask_dir = "/shared/projects/phisat2/data/interim/phisat2_cloud_masks"
    input_csv = "/shared/projects/phisat2/data/index/phisat2_metadata.csv"
    output_csv = "/shared/projects/phisat2/data/index/phisat2_cloud_stats_detailed.csv"
    
    df = pd.read_csv(input_csv)
    rows_to_process = [row.to_dict() for _, row in df.iterrows()]
    results = []

    print(f"Analyze {len(rows_to_process)} masks...")
    
    with ProcessPoolExecutor(max_workers=24) as executor:
        futures = {executor.submit(calculate_full_cloud_stats, row, mask_dir): row for row in rows_to_process}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extraction Stats"):
            results.append(future.result())

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv, index=False)
    
    print(f"\nStats saved to : {output_csv}")
    
    success = df_results[df_results['status'] == 'SUCCESS']
    if not success.empty:
        print(f"Average Thick Cloud : {success['thick_cloud_pct'].mean():.1f}%")
        print(f"Average Thin Cloud  : {success['thin_cloud_pct'].mean():.1f}%")