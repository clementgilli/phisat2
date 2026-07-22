import math
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.io import MemoryFile
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import reproject
from contextlib import ExitStack
import pyproj
from shapely.geometry import box
from shapely.ops import transform
from pathlib import Path
import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import argparse

def find_scl_paths(l2a_paths_list):
    scl_paths = []
    for l2a_path in l2a_paths_list:
        granule = Path(l2a_path) / "GRANULE"
        if not granule.exists(): continue
        tiles = [d for d in granule.iterdir() if d.is_dir() and d.name.startswith('L2A_T')]
        if not tiles: continue
        
        r20m_dir = tiles[0] / "IMG_DATA" / "R20m"
        matches = list(r20m_dir.glob("*_SCL_20m.jp2"))
        if matches:
            scl_paths.append(str(matches[0]))
    return scl_paths

def process_single_cloud_mask(row_dict, output_dir):
    
    phisat_id = row_dict['product_id']
    target_tif = row_dict['merged_tif_path']
    l2a_paths_list = str(row_dict['l2a_paths']).split(',')
    
    tif_name = f"{phisat_id}_S2B_cloud_mask.tif"
    out_mask_path = os.path.join(output_dir, tif_name)

    if os.path.exists(out_mask_path):
        row_dict['mask_path'] = out_mask_path
        row_dict['mask_status'] = 'ALREADY_EXISTS'
        return row_dict

    try:
        scl_paths = find_scl_paths(l2a_paths_list)
        if not scl_paths:
            raise Exception("No SCL band found in L2A paths")

        with rasterio.open(target_tif) as src_master:
            master_meta = src_master.profile.copy()
            master_shape = (src_master.height, src_master.width)
            master_transform = src_master.transform
            master_crs = src_master.crs
            master_bounds = src_master.bounds 

        bbox_master = box(master_bounds.left, master_bounds.bottom, master_bounds.right, master_bounds.top)

        with rasterio.Env(GDAL_NUM_THREADS='1', GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR'):
            datasets_for_band = []
            memfiles = []
            
            for fp in scl_paths:
                with rasterio.open(fp) as src:
                    project = pyproj.Transformer.from_crs(master_crs, src.crs, always_xy=True).transform
                    native_bbox = transform(project, bbox_master)
                    
                    try:
                        out_image, out_transform = mask(src, [native_bbox], crop=True)
                    except ValueError: continue
                    
                    mem = MemoryFile()
                    memfiles.append(mem)

                    with mem.open(driver='GTiff', height=out_image.shape[1], width=out_image.shape[2],
                                count=1, dtype=out_image.dtype, crs=src.crs, transform=out_transform) as ds_write:
                        ds_write.write(out_image[0], 1)

                    ds_read = mem.open(mode='r')
                    datasets_for_band.append(ds_read)

            if not datasets_for_band:
                raise Exception("Mask crop empty (no overlap)")

            with ExitStack() as stack:
                vrts = []
                for ds in datasets_for_band:
                    vrt = stack.enter_context(WarpedVRT(ds, crs=master_crs))
                    vrts.append(vrt)
                
                merged_scl, merged_transform = merge(vrts, method='first')
            
            final_mask = np.empty((1, master_shape[0], master_shape[1]), dtype=np.uint8)
            
            reproject(
                source=merged_scl,
                destination=final_mask,
                src_transform=merged_transform,
                src_crs=master_crs,
                dst_transform=master_transform,
                dst_crs=master_crs,
                resampling=Resampling.nearest
            )
            
            for d in datasets_for_band: d.close()
            for m in memfiles: m.close()

        master_meta.update({
            'count': 1,
            'dtype': 'uint8',
            'nodata': 0,
            'compress': 'lzw'
        })
        
        with rasterio.open(out_mask_path, 'w', **master_meta) as dst:
            dst.write(final_mask[0], 1)
            dst.descriptions = tuple(["SCL_CLOUD_MASK"])

        row_dict['mask_path'] = out_mask_path
        row_dict['mask_status'] = 'SUCCESS'
        return row_dict

    except Exception as e:
        row_dict['mask_path'] = None
        row_dict['mask_status'] = f"ERROR: {str(e)}"
        return row_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract S2B Cloud Masks")
    parser.add_argument("--output_csv", required=True, help="Path to final output CSV")
    parser.add_argument("--output_dir", required=True, help="Directory to save TIFs masks")
    parser.add_argument("--workers", type=int, default=16, help="Number of CPU cores to use")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    df_meta_phi = pd.read_csv("/shared/projects/phisat2/data/index/phisat2_metadata.csv")
    df_meta_s2b = pd.read_csv("/shared/projects/phisat2/data/index/phisat2_to_s2b.csv")
    df_meta_s2b.drop(columns=['min_lat', 'max_lat', 'min_lon', 'max_lon'], inplace=True, errors='ignore')
    df_merged = pd.read_csv("/shared/projects/phisat2/data/index/s2b_merged.csv")
    
    df_full = pd.merge(df_meta_phi, df_meta_s2b, on="product_id", how="inner")
    df_full = pd.merge(df_full, df_merged, on="product_id", how="inner")
    
    df_valid = df_full[df_full['merged_tif_path'].notna()].copy()
    
    if os.path.exists(args.output_csv):
        processed_df = pd.read_csv(args.output_csv)
        processed_ids = set(processed_df['product_id'].astype(str))
        df_valid = df_valid[~df_valid['product_id'].astype(str).isin(processed_ids)]
        print(f"Starting processing. {len(df_valid)} masks remaining.")
    else:
        out_cols = ['product_id', 'mask_path', 'mask_status']
        pd.DataFrame(columns=out_cols).to_csv(args.output_csv, index=False)
        print(f"Starting processing for {len(df_valid)} masks.")

    rows_to_process = [row.to_dict() for _, row in df_valid.iterrows()]

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_cloud_mask, row, args.output_dir): row for row in rows_to_process}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting Masks"):
            result_row = future.result()
            res_dict = {
                'product_id': result_row['product_id'],
                'mask_path': result_row['mask_path'],
                'mask_status': result_row['mask_status']
            }
            pd.DataFrame([res_dict]).to_csv(args.output_csv, mode='a', header=False, index=False)