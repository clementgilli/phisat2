import torch

PHISAT2_REAL_BANDS = ["PAN", "BLUE", "GREEN", "RED", "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3", "NIR_BROAD"]
PHISAT2_SIM_BANDS  = ["BLUE", "GREEN", "RED", "PAN", "NIR_BROAD", "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3"]
S2_BANDS           = ["COASTAL_AEROSOL", "BLUE", "GREEN", "RED", "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3", "NIR_BROAD", "NIR_NARROW", "WATER_VAPOR", "CIRRUS", "SWIR_1", "SWIR_2"]

STATS = {
    "phisat2_sim": {
        "BLUE": (1884.56169544, 1899.72067083), "GREEN": (1701.8988641, 1743.80445286), "RED": (1818.49680678, 2020.09785262), "PAN": (1856.58051233, 1873.41863641),
        "NIR_BROAD": (2364.33335501, 1924.71680909), "RED_EDGE_1": (1961.68849886, 2034.2549607), "RED_EDGE_2": (2294.99146283, 1992.56097028), "RED_EDGE_3": (2457.69823862, 1996.09805038)
    },
    "phisat2_real": {
        "PAN": (49.23, 9.09), "BLUE": (49.78, 7.28), "GREEN": (49.02, 6.52), "RED": (48.42, 6.95),
        "RED_EDGE_1": (55.40, 7.95), "RED_EDGE_2": (57.35, 8.31), "RED_EDGE_3": (56.78, 8.36), "NIR_BROAD": (51.16, 8.38)
    },
    "s2": {
        "COASTAL_AEROSOL": (2357.089, 1624.683), "BLUE": (2137.385, 1675.806), "GREEN": (2018.788, 1557.708), "RED": (2082.986, 1833.702),
        "RED_EDGE_1": (2295.651, 1823.738), "RED_EDGE_2": (2854.537, 1733.977), "RED_EDGE_3": (3122.849, 1732.131), "NIR_BROAD": (3040.560, 1679.732),
        "NIR_NARROW": (3306.481, 1727.260), "WATER_VAPOR": (1473.847, 1024.687), "CIRRUS": (506.070, 442.165), "SWIR_1": (2472.825, 1331.411), "SWIR_2": (1838.929, 1160.419)
    }
}

def get_norm_tensors(sensor: str, bands: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns mean and std tensors for the specified sensor and bands, shaped for broadcasting over (C, H, W)."""
    if sensor not in STATS:
        raise ValueError(f"Sensor {sensor} not found in STATS.")
        
    means, stds = [], []
    for band in bands:
        if band not in STATS[sensor]:
            raise ValueError(f"Band {band} not found for sensor {sensor}.")
        mean, std = STATS[sensor][band]
        means.append(mean)
        stds.append(std)
        
    return torch.tensor(means).view(-1, 1, 1), torch.tensor(stds).view(-1, 1, 1)