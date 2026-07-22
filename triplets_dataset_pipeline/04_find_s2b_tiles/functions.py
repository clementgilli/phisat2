import requests
import pandas as pd
import numpy as np
import json
import os
from datetime import timedelta
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from shapely.wkt import loads
from shapely.ops import unary_union
from shapely.geometry import box
from tqdm import tqdm
import rasterio
from rasterio.warp import transform_bounds
from rasterio.merge import merge
from rasterio.vrt import WarpedVRT
from contextlib import ExitStack
import matplotlib.pyplot as plt
import math
from pathlib import Path

session = requests.Session()

def fetch_l1c_twin(url, l2a_prod):
    exact_time = l2a_prod['ContentDate']['Start']
    tile_id = l2a_prod['Name'].split('_')[5]
    query = (f"Collection/Name eq 'SENTINEL-2' and ContentDate/Start eq {exact_time} "
             f"and contains(Name, '_{tile_id}_') and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq 'S2MSI1C')")
    res = session.get(url, params={"$filter": query, "$select": "Name,S3Path"}).json().get('value', [])
    if not res: return None
    return {"l1c": res[0].get("S3Path", "").rstrip("/"), "l2a": l2a_prod.get("S3Path", "").rstrip("/")}

def verify_mosaic_coverage(candidate_products, target_box):
    url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
    names_filter = " or ".join([f"Name eq '{p['Name']}'" for p in candidate_products])
    res = session.get(url, params={"$filter": names_filter}).json().get('value', [])
    if not res: return 0.0
        
    polygons = []
    for item in res:
        wkt = item.get('Footprint') or item.get('GeoFootprint') or item.get('OData.CSC.Footprint') or ''
        if wkt:
            wkt_clean = str(wkt)
            if "SRID=4326;" in wkt_clean: wkt_clean = wkt_clean.split("SRID=4326;")[1]
            wkt_clean = wkt_clean.replace("'", "").replace("geography", "").strip()
            try: polygons.append(loads(wkt_clean))
            except: pass
                
    if not polygons: return 0.0
    mosaic_geom = unary_union(polygons)
    return mosaic_geom.intersection(target_box).area / target_box.area

def get_optimal_s2_mosaic_tradeoff(min_lon, min_lat, max_lon, max_lat, target_date_str, buffer_km=35.0, window_days=15, w_time=0.0, w_cloud=1.0):
    mean_lat = (min_lat + max_lat) / 2.0
    buffer_deg_lat = buffer_km / 111.32
    buffer_deg_lon = buffer_km / (111.32 * np.cos(np.radians(mean_lat)))
    
    min_lon_buf, max_lon_buf = min_lon - buffer_deg_lon, max_lon + buffer_deg_lon
    min_lat_buf, max_lat_buf = min_lat - buffer_deg_lat, max_lat + buffer_deg_lat
    target_box = box(min_lon_buf, min_lat_buf, max_lon_buf, max_lat_buf)
    area = f"POLYGON(({min_lon_buf} {min_lat_buf}, {max_lon_buf} {min_lat_buf}, {max_lon_buf} {max_lat_buf}, {min_lon_buf} {max_lat_buf}, {min_lon_buf} {min_lat_buf}))"
    
    target_dt = pd.to_datetime(target_date_str).tz_localize(None)
    start_str = (target_dt - timedelta(days=window_days)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    end_str = (target_dt + timedelta(days=window_days)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    
    url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
    query_l2a = (f"Collection/Name eq 'SENTINEL-2' and OData.CSC.Intersects(area=geography'SRID=4326;{area}') "
                 f"and ContentDate/Start gt {start_str} and ContentDate/Start lt {end_str} "
                 f"and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq 'S2MSI2A')")
    
    res_l2a = session.get(url, params={"$filter": query_l2a, "$expand": "Attributes", "$select": "Name,ContentDate,S3Path,Attributes", "$top": 500}).json().get('value', [])
    if not res_l2a: return [], None

    required_tiles = set(p['Name'].split('_')[5] for p in res_l2a)
    datatakes = defaultdict(list)
    for p in res_l2a:
        day = p['ContentDate']['Start'][:10]
        datatakes[day].append(p)

    valid_mosaics = []
    for day, products in datatakes.items():
        day_tiles = set(p['Name'].split('_')[5] for p in products)
        if day_tiles == required_tiles:
            avg_cloud = sum(next((a['Value'] for a in p.get('Attributes', []) if a['Name'] == 'cloudCover'), 100.0) for p in products) / len(products)
            s2_dt = pd.to_datetime(products[0]['ContentDate']['Start']).tz_localize(None)
            delta_days = abs((s2_dt - target_dt).total_seconds()) / 86400
            valid_mosaics.append({
                "day": day, "cloud_cover": avg_cloud, "delta_days": delta_days, 
                "base_score": (delta_days * w_time) + (avg_cloud * w_cloud), "products": products
            })

    if not valid_mosaics: return [], None

    valid_mosaics.sort(key=lambda x: x['base_score'])
    top_candidates = valid_mosaics[:3]
    
    best_final_mosaic = None
    best_final_score = float('inf')

    for candidate in top_candidates:
        coverage = verify_mosaic_coverage(candidate['products'], target_box)
        if coverage < 0.85: continue
            
        candidate['coverage'] = coverage
        missing_data_pct = (1.0 - coverage) * 100.0
        effective_bad_pixels = missing_data_pct + candidate['cloud_cover']
        final_score = (candidate['delta_days'] * w_time) + (effective_bad_pixels * w_cloud)
        
        if final_score < best_final_score:
            best_final_score = final_score
            best_final_mosaic = candidate

    if not best_final_mosaic: return [], None

    final_pairs = []
    with ThreadPoolExecutor(max_workers=len(best_final_mosaic['products'])) as executor:
        futures = [executor.submit(fetch_l1c_twin, url, prod) for prod in best_final_mosaic['products']]
        for future in futures:
            res = future.result()
            if res: final_pairs.append(res)

    return final_pairs, best_final_mosaic

def process_single_row(row):
    phisat_id = row['product_id']
    min_lon, min_lat = row['min_lon'], row['min_lat']
    max_lon, max_lat = row['max_lon'], row['max_lat']
    target_date = row['start_date'] 
    
    try:
        final_pairs, best_mosaic = get_optimal_s2_mosaic_tradeoff(
            min_lon, min_lat, max_lon, max_lat, target_date
        )
        
        if final_pairs and best_mosaic:
            l1c_paths = ",".join([p['l1c'] for p in final_pairs])
            l2a_paths = ",".join([p['l2a'] for p in final_pairs])
            
            return {
                'product_id': phisat_id,
                'min_lon': min_lon, 'min_lat': min_lat, 'max_lon': max_lon, 'max_lat': max_lat,
                'target_date': target_date,
                's2_day': best_mosaic['day'],
                's2_cloud_cover': round(best_mosaic['cloud_cover'], 2),
                's2_coverage': round(best_mosaic.get('coverage', 0.0) * 100, 2),
                'delta_days': round(best_mosaic['delta_days'], 2),
                'l1c_paths': l1c_paths,
                'l2a_paths': l2a_paths,
                'status': 'SUCCESS'
            }
        else:
            return {
                'product_id': phisat_id, 'min_lon': min_lon, 'min_lat': min_lat, 'max_lon': max_lon, 'max_lat': max_lat,
                'target_date': target_date, 's2_day': None, 's2_cloud_cover': None, 's2_coverage': None, 'delta_days': None,
                'l1c_paths': None, 'l2a_paths': None, 'status': 'NOT_FOUND'
            }
            
    except Exception as e:
        return {
            'product_id': phisat_id, 'min_lon': min_lon, 'min_lat': min_lat, 'max_lon': max_lon, 'max_lat': max_lat,
            'target_date': target_date, 's2_day': None, 's2_cloud_cover': None, 's2_coverage': None, 'delta_days': None,
            'l1c_paths': None, 'l2a_paths': None, 'status': f'ERROR: {str(e)}'
        }

def process_dataset(input_csv_path, output_csv_path, max_workers=10):
    
    df = pd.read_csv(input_csv_path)
    
    start_idx = 0
    if os.path.exists(output_csv_path):
        processed_df = pd.read_csv(output_csv_path)
        # On trouve les IDs déjà traités pour ne pas les refaire
        processed_ids = set(processed_df['product_id'].astype(str))
        df = df[~df['product_id'].astype(str).isin(processed_ids)]
        print(f"Reprise du traitement. {len(df)} images restantes.")
    else:
        columns = [
            'product_id', 'min_lon', 'min_lat', 'max_lon', 'max_lat', 
            'target_date', 's2_day', 's2_cloud_cover', 's2_coverage', 'delta_days', 
            'l1c_paths', 'l2a_paths', 'status'
        ]
        pd.DataFrame(columns=columns).to_csv(output_csv_path, index=False)
        print(f"Démarrage pour {len(df)} images.")

    if df.empty:
        print("Toutes les images ont déjà été traitées !")
        return

    rows_to_process = [row for _, row in df.iterrows()]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_row, row): row for row in rows_to_process}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Sourcing Sentinel-2"):
            result = future.result()
            
            pd.DataFrame([result]).to_csv(output_csv_path, mode='a', header=False, index=False)

def plot_sanity_check_local(min_lon, min_lat, max_lon, max_lat, buffer_km, final_pairs):

    mean_lat = (min_lat + max_lat) / 2.0
    buffer_deg_lat = buffer_km / 111.32
    buffer_deg_lon = buffer_km / (111.32 * math.cos(math.radians(mean_lat)))

    fig, ax = plt.subplots(figsize=(10, 10))

    phisat_bbox = box(min_lon, min_lat, max_lon, max_lat)
    ax.plot(*phisat_bbox.exterior.xy, color='red', linewidth=2.5, label='PhiSat-2 Bounding Box')
    ax.fill(*phisat_bbox.exterior.xy, color='red', alpha=0.15)

    buf_box = box(min_lon - buffer_deg_lon, min_lat - buffer_deg_lat, 
                  max_lon + buffer_deg_lon, max_lat + buffer_deg_lat)
    ax.plot(*buf_box.exterior.xy, color='orange', linestyle='--', linewidth=2, label=f'Buffer (+{buffer_km}km)')

    tiles_plotted = 0

    for i, pair in enumerate(final_pairs):
        l2a_dir = Path(pair['l2a'])
        if not l2a_dir.exists():
            print(f"Warning: L2A directory not found for pair {i}: {l2a_dir}")
            continue
            
        tile_name = l2a_dir.name.split('_')[5] # ex: T46TFK
        
        scl_path = None
        granule_dir = l2a_dir / "GRANULE"
        tiles = [d for d in os.listdir(granule_dir) if d.startswith('L2A_T')]
        
        if tiles:
            r20m_dir = granule_dir / tiles[0] / "IMG_DATA" / "R20m"
            if r20m_dir.exists():
                for f in os.listdir(r20m_dir):
                    if f.endswith("_SCL_20m.jp2") or f.endswith("_SCL_20m.tif"):
                        scl_path = r20m_dir / f
                        break
        
        if scl_path:
            try:
                with rasterio.open(scl_path) as src:
                    bounds_utm = src.bounds
                    crs_utm = src.crs
                    
                    minx, miny, maxx, maxy = transform_bounds(crs_utm, 'EPSG:4326', *bounds_utm)
                    
                    s2_poly = box(minx, miny, maxx, maxy)
                    ax.plot(*s2_poly.exterior.xy, color='blue', linewidth=1.5, alpha=0.8, 
                            label='Tile Sentinel-2' if tiles_plotted == 0 else "")
                    
                    ax.text(s2_poly.centroid.x, s2_poly.centroid.y, tile_name, 
                            color='blue', fontsize=12, ha='center', va='center', weight='bold')
                    
                    tiles_plotted += 1
            except Exception as e:
                print(f"Error processing tile {tile_name}: {e}")

    if tiles_plotted == 0:
        print("No Sentinel-2 tiles were plotted. Please check the final_pairs data and file paths.")
        plt.close()
        return

    ax.set_aspect('equal')
    plt.title(f"Mosaic Area with Sentinel-2 Tiles")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.show()
    
    
    
def get_b04_paths_from_l1c(final_pairs):
    b04_paths = []
    for pair in final_pairs:
        l1c_dir = Path(pair['l1c'])
        granule = l1c_dir / "GRANULE"
        if not granule.exists(): continue
        
        tiles = [d for d in os.listdir(granule) if d.startswith('L1C_T')]
        if tiles:
            img_data = granule / tiles[0] / "IMG_DATA"
            for f in os.listdir(img_data):
                if f.endswith("_B04.jp2"):
                    b04_paths.append(str(img_data / f))
                    break
    return b04_paths

def preview_merged_and_cropped_s2(min_lon, min_lat, max_lon, max_lat, buffer_km, final_pairs):
    jp2_paths = get_b04_paths_from_l1c(final_pairs)
    if not jp2_paths:
        return

    mean_lat = (min_lat + max_lat) / 2.0
    buffer_deg_lat = buffer_km / 111.32
    buffer_deg_lon = buffer_km / (111.32 * math.cos(math.radians(mean_lat)))

    minx = min_lon - buffer_deg_lon
    miny = min_lat - buffer_deg_lat
    maxx = max_lon + buffer_deg_lon
    maxy = max_lat + buffer_deg_lat

    with ExitStack() as stack:
        srcs = [stack.enter_context(rasterio.open(fp)) for fp in jp2_paths]
        
        vrts = [stack.enter_context(WarpedVRT(src, crs='EPSG:4326')) for src in srcs]
        
        mosaic, out_trans = merge(vrts, bounds=(minx, miny, maxx, maxy), method='first')

    img = mosaic[0]
    
    img = np.where(img == 0, np.nan, img)

    plt.figure(figsize=(12, 10))
    
    extent = [minx, maxx, miny, maxy]
    plt.imshow(img, cmap='gray', extent=extent)
    
    phisat_bbox = box(min_lon, min_lat, max_lon, max_lat)
    plt.plot(*phisat_bbox.exterior.xy, color='red', linewidth=3, label='Zone cible PhiSat-2')
    
    plt.title("Mosaic Sentinel-2", fontsize=16)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend(loc='upper right')
    
    plt.clim(np.nanpercentile(img, 2), np.nanpercentile(img, 98)) 
    
    plt.show()