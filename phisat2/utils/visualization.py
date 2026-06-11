import numpy as np

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
    "burned": {
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