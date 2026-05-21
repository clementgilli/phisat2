import math
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.io import MemoryFile
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from contextlib import ExitStack
import pyproj
from shapely.geometry import box
from shapely.ops import transform
from pathlib import Path
import os
import xml.etree.ElementTree as ET
import json
from datetime import datetime
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import argparse

S2_BANDS_SIM = ["B02", "B03", "B04", "B08", "B05", "B06", "B07"]
S2_BANDS_NAMES = ["BLUE", "GREEN", "RED", "NIR_BROAD", "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3"]
XML_BAND_ID = {1: "B02", 2: "B03", 3: "B04", 7: "B08", 4: "B05", 5: "B06", 6: "B07"}

def get_earth_sun_distance(date_obj):
    day_of_year = date_obj.timetuple().tm_yday
    return 1.0 - 0.01672 * math.cos(math.radians(0.9856 * (day_of_year - 4)))

def extract_simulator_metadata(l1c_path, target_date_str):
    meta = {"earth_sun_dist": 1.0, "solar_irradiances": {}, "sun_zenith_angles": None}
    
    try:
        dt = datetime.strptime(target_date_str[:10], "%Y-%m-%d")
        meta["earth_sun_dist"] = get_earth_sun_distance(dt)
    except: pass

    safe_dir = Path(l1c_path)
    xml_main = safe_dir / "MTD_MSIL1C.xml"
    if xml_main.exists():
        try:
            tree = ET.parse(xml_main)
            root = tree.getroot()
            irrad_list = root.find(".//Solar_Irradiance_List")
            if irrad_list is not None:
                for elem in irrad_list.findall("SOLAR_IRRADIANCE"):
                    b_id = int(elem.get("bandId", -1))
                    if b_id in XML_BAND_ID:
                        meta["solar_irradiances"][XML_BAND_ID[b_id]] = float(elem.text)
        except: pass

    granule_dir = safe_dir / "GRANULE"
    if granule_dir.exists():
        tiles = [d for d in granule_dir.iterdir() if d.is_dir() and d.name.startswith('L1C_T')]
        if tiles:
            xml_tile = tiles[0] / "MTD_TL.xml"
            if xml_tile.exists():
                try:
                    tree = ET.parse(xml_tile)
                    root = tree.getroot()
                    zenith_grid = root.find(".//Sun_Angles_Grid/Zenith/Values_List")
                    if zenith_grid is not None:
                        rows = []
                        for val_elem in zenith_grid.findall("VALUES"):
                            if val_elem.text:
                                rows.append([float(v) for v in val_elem.text.strip().split()])
                        meta["sun_zenith_angles"] = rows
                except: pass
    return meta

def get_bands_paths_sim(l1c_paths_list):
    band_map = {b: [] for b in S2_BANDS_SIM}
    for l1c_path in l1c_paths_list:
        granule = Path(l1c_path) / "GRANULE"
        if not granule.exists(): continue
        tiles = [d for d in granule.iterdir() if d.is_dir() and d.name.startswith('L1C_T')]
        if not tiles: continue
        img_data = tiles[0] / "IMG_DATA"
        for band in S2_BANDS_SIM:
            matches = list(img_data.glob(f"*_{band}.jp2"))
            if matches: band_map[band].append(str(matches[0]))
    return band_map

def process_single_product(row_dict, output_dir, buffer_km=35.0):
    
    phisat_id = row_dict['product_id']
    min_lon, min_lat = row_dict['min_lon'], row_dict['min_lat']
    max_lon, max_lat = row_dict['max_lon'], row_dict['max_lat']
    target_date = row_dict['target_date']
    l1c_paths_list = row_dict['l1c_paths'].split(',')
    
    tif_name = f"{phisat_id}_S2B_7bands.tif"
    json_name = f"{phisat_id}_S2B_metadata.json"
    tif_path = os.path.join(output_dir, tif_name)
    json_path = os.path.join(output_dir, json_name)

    if os.path.exists(tif_path) and os.path.exists(json_path):
        row_dict['merged_tif_path'] = tif_path
        row_dict['merged_json_path'] = json_path
        row_dict['merged_status'] = 'ALREADY_EXISTS'
        return row_dict

    try:
        metadata = extract_simulator_metadata(l1c_paths_list[0], target_date)
        band_map = get_bands_paths_sim(l1c_paths_list)
        
        mean_lat = (min_lat + max_lat) / 2.0
        buf_deg_lat = buffer_km / 111.32
        buf_deg_lon = buffer_km / (111.32 * math.cos(math.radians(mean_lat)))
        bbox_4326 = box(min_lon - buf_deg_lon, min_lat - buf_deg_lat, max_lon + buf_deg_lon, max_lat + buf_deg_lat)

        final_channels = []
        master_transform = None
        master_crs = None
        master_shape = None
        
        with rasterio.Env(GDAL_NUM_THREADS='1', GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR'):
            for band in S2_BANDS_SIM:
                paths = band_map[band]
                if not paths: continue
                
                datasets_for_band = []
                memfiles = []
                
                for fp in paths:
                    with rasterio.open(fp) as src:
                        project = pyproj.Transformer.from_crs("EPSG:4326", src.crs, always_xy=True).transform
                        native_bbox = transform(project, bbox_4326)
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

                if not datasets_for_band: continue

                if master_crs is None:
                    master_crs = datasets_for_band[0].crs

                # On aligne virtuellement toutes les tuiles sur le même CRS avant le merge
                with ExitStack() as stack:
                    vrts = []
                    for ds in datasets_for_band:
                        vrt = stack.enter_context(WarpedVRT(ds, crs=master_crs))
                        vrts.append(vrt)
                    
                    band_merged, band_transform = merge(vrts, method='first')

                if master_transform is None:
                    master_transform = band_transform
                    master_shape = (band_merged.shape[1], band_merged.shape[2])
                
                if band_merged.shape[1:] != master_shape:
                    resampled_band = np.empty(master_shape, dtype=band_merged.dtype)
                    with MemoryFile() as memfile:
                        with memfile.open(driver='GTiff', height=band_merged.shape[1], width=band_merged.shape[2],
                                          count=1, dtype=band_merged.dtype, crs=master_crs, transform=band_transform) as dataset:
                            dataset.write(band_merged[0], 1)
                        with memfile.open() as dataset:
                            resampled_band = dataset.read(1, out_shape=master_shape, resampling=Resampling.bilinear)
                    final_channels.append(resampled_band)
                else:
                    final_channels.append(band_merged[0])
                
                for d in datasets_for_band: d.close()
                for m in memfiles: m.close()

        if not final_channels or len(final_channels) != len(S2_BANDS_SIM):
            raise Exception("Missing bands after merge")

        stack = np.array(final_channels)
        
        profile = {
            'driver': 'GTiff', 'dtype': stack.dtype, 'count': 7,
            'height': stack.shape[1], 'width': stack.shape[2],
            'crs': master_crs, 'transform': master_transform, 'compress': 'lzw'
        }
        with rasterio.open(tif_path, 'w', **profile) as dst:
            dst.write(stack)
            dst.descriptions = tuple(S2_BANDS_NAMES)
            
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        return {
            'product_id': phisat_id,
            'merged_tif_path': tif_path,
            'merged_json_path': json_path,
            'merged_status': 'SUCCESS'
        }

    except Exception as e:
        return {
            'product_id': phisat_id,
            'merged_tif_path': None,
            'merged_json_path': None,
            'merged_status': f"ERROR: {str(e)}"
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop and extract S2B data for Simulator")
    parser.add_argument("--input", required=True, help="Path to input CSV (phisat2_to_s2b.csv)")
    parser.add_argument("--output_csv", required=True, help="Path to final output CSV")
    parser.add_argument("--output_dir", required=True, help="Directory to save TIFs and JSONs")
    parser.add_argument("--workers", type=int, default=16, help="Number of CPU cores to use")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    df = pd.read_csv(args.input)
    df_valid = df[df['status'] == 'SUCCESS'].copy()
    
    start_idx = 0
    if os.path.exists(args.output_csv):
        processed_df = pd.read_csv(args.output_csv)
        processed_ids = set(processed_df['product_id'].astype(str))
        df_valid = df_valid[~df_valid['product_id'].astype(str).isin(processed_ids)]
        print(f"Starting processing. {len(df_valid)} images remaining.")
    else:
        out_cols = ['product_id', 'merged_tif_path', 'merged_json_path', 'merged_status']
        pd.DataFrame(columns=out_cols).to_csv(args.output_csv, index=False)
        print(f"Starting processing for {len(df_valid)} images.")

    rows_to_process = [row.to_dict() for _, row in df_valid.iterrows()]

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_product, row, args.output_dir): row for row in rows_to_process}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Cropping & Merging S2B"):
            result_row = future.result()
            
            pd.DataFrame([result_row]).to_csv(args.output_csv, mode='a', header=False, index=False)