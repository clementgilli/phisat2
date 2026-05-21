import os
import numpy as np
import rasterio as rio
from rasterio.enums import Resampling
import pandas as pd
from functools import partial
from omnicloudmask import predict_from_load_func
from pathlib import Path

def load_phisat2_custom(input_path, scale=2, fixed_vmax=10000.0):
    bands = [4, 3, 8]
    
    with rio.open(input_path) as src:
        out_h = int(src.height / scale)
        out_w = int(src.width / scale)

        data = src.read(
            bands,
            out_shape=(len(bands), out_h, out_w),
            resampling=Resampling.nearest
        )

        profile = src.profile.copy()
        transform = src.transform * src.transform.scale(
            (src.width / out_w),
            (src.height / out_h)
        )
        profile.update({
            "height": out_h,
            "width": out_w,
            "transform": transform,
            "count": 3,
            "dtype": "float32"
        })

    data_norm = data.astype(np.float32) / fixed_vmax
    data_norm = np.clip(data_norm, 0, 1)

    return data_norm, profile

df_phisat2_meta = pd.read_csv("/shared/projects/phisat2/data/index/phisat2_metadata.csv")
output_directory = Path("/shared/projects/phisat2/data/interim/phisat2_cloud_masks")
output_directory.mkdir(parents=True, exist_ok=True)

symlink_dir = Path("/shared/projects/phisat2/data/interim/phisat2_symlinks")
symlink_dir.mkdir(parents=True, exist_ok=True)

scene_paths_virtual = []

print("🔗 Création des liens symboliques pour renommer les inputs...")

for _, row in df_phisat2_meta.iterrows():
    product_id = row['product_id']
    real_path = f"{row['folder_path']}/bands/scene_0_BC_multiband.tiff"
    
    virtual_path = symlink_dir / f"{product_id}_phisat2.tiff"
    
    if not virtual_path.exists():
        os.symlink(real_path, virtual_path)
        
    scene_paths_virtual.append(str(virtual_path))

output_directory = "/shared/projects/phisat2/data/interim/phisat2_cloud_masks"
os.makedirs(output_directory, exist_ok=True)

my_loader = partial(load_phisat2_custom, scale=2, fixed_vmax=10000.0)

print(f"Starting inference on {len(scene_paths_virtual)} images...")

pred_paths = predict_from_load_func(
    scene_paths=scene_paths_virtual,
    load_func=my_loader,
    output_dir=output_directory,
    overwrite=False,
    inference_device="cuda",
    inference_dtype="bf16",
    batch_size=32,
    compile_models=False
)

print(f"Done! Cloud masks saved in {output_directory}")