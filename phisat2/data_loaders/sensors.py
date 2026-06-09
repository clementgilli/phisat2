import torch

PHISAT2_REAL_BANDS = ["PAN", "BLUE", "GREEN", "RED", "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3", "NIR_BROAD"]
PHISAT2_SIM_BANDS  = ["BLUE", "GREEN", "RED", "PAN", "NIR_BROAD", "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3"]
S2_BANDS           = ["COASTAL_AEROSOL", "BLUE", "GREEN", "RED", "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3", "NIR_BROAD", "NIR_NARROW", "WATER_VAPOR", "CIRRUS", "SWIR_1", "SWIR_2"]

STATS = {
    "phisat2_sim": {
        "BLUE": (0.1884, 0.1899), "GREEN": (0.1701, 0.1743), "RED": (0.1818, 0.2020), "PAN": (0.1856, 0.1873),
        "NIR_BROAD": (0.2364, 0.1924), "RED_EDGE_1": (0.1961, 0.2034), "RED_EDGE_2": (0.2294, 0.1992), "RED_EDGE_3": (0.2457, 0.1996)
    },
    "phisat2_real": {
        "PAN": (49.23**2, 9.09**2), "BLUE": (49.78**2, 7.28**2), "GREEN": (49.02**2, 6.52**2), "RED": (48.42**2, 6.95**2),
        "RED_EDGE_1": (55.40**2, 7.95**2), "RED_EDGE_2": (57.35**2, 8.31**2), "RED_EDGE_3": (56.78**2, 8.36**2), "NIR_BROAD": (51.16**2, 8.38**2)
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