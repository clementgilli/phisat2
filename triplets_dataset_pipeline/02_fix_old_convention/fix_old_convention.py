import os
import re
import rasterio
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

L1_DIR = Path("/shared/projects/phisat2/data/raw/phisat2/L1")
TARGET_FILENAME = "scene_0_BC_multiband.tiff"

def process_folder(base_path):
    if not base_path.is_dir():
        return "IGNORED"
        
    bands_dir = base_path / "bands"
    target_file = bands_dir / TARGET_FILENAME
    
    if target_file.exists():
        return "ALREADY_GOOD"
        
    band_files = {}
    for f in bands_dir.glob("*.tiff"):
        match = re.search(r'_([0-7])\.tiff$', f.name)
        if match:
            band_idx = int(match.group(1))
            band_files[band_idx] = f
            
    if len(band_files) != 8:
        return "ERROR_MISSING_BANDS"
        
    try:
        with rasterio.open(band_files[0]) as src0:
            meta = src0.meta.copy()
            meta.update(count=8)
            
            with rasterio.open(target_file, 'w', **meta) as dst:
                for i in range(8):
                    with rasterio.open(band_files[i]) as src:
                        dst.write(src.read(1), i + 1)
        return "SUCCESS"
    except Exception as e:
        return f"ERROR_{e}"


if __name__ == '__main__':
    folders = [d for d in L1_DIR.iterdir() if d.is_dir()]
    
    num_cores = max(1, os.cpu_count() - 2)
    
    count_success = 0
    count_errors = 0
    count_skipped = 0
    
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = {executor.submit(process_folder, folder): folder for folder in folders}
        
        for future in tqdm(as_completed(futures), total=len(folders), desc="Standardisation"):
            status = future.result()
            
            if status == "SUCCESS":
                count_success += 1
            elif status == "ALREADY_GOOD" or status == "IGNORED":
                count_skipped += 1
            else:
                count_errors += 1

    print("\n" + "="*40)
    print(f"BILAN FINAL :")
    print(f"{count_success} images corrigées")
    print(f"{count_skipped} images déjà au bon format ignorées")
    print(f"{count_errors} erreurs rencontrées")
    print("="*40)