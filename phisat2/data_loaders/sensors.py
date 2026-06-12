import torch

PHISAT2_REAL_BANDS = ["PAN", "BLUE", "GREEN", "RED", "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3", "NIR_BROAD"]
PHISAT2_SIM_BANDS  = ["BLUE", "GREEN", "RED", "PAN", "NIR_BROAD", "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3"]
S2_BANDS           = ["COASTAL_AEROSOL", "BLUE", "GREEN", "RED", "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3", "NIR_BROAD", "NIR_NARROW", "WATER_VAPOR", "CIRRUS", "SWIR_1", "SWIR_2"]

STATS = {
    "phisat2_sim": {
        "BLUE": (0.1428, 0.0699), "GREEN": (0.1367, 0.0742), "RED": (0.1461, 0.096), "PAN": (0.1492, 0.0791),
        "NIR_BROAD": (0.2182, 0.0929), "RED_EDGE_1": (0.1632, 0.0917), "RED_EDGE_2": (0.2025, 0.0899), "RED_EDGE_3": (0.2229, 0.0938)
    },
    "phisat2_real": {
        "PAN": (519.3167, 249.5329), "BLUE": (686.6016, 304.6345), "GREEN": (581.7667, 306.9884), "RED": (541.0254, 341.5756),
        "RED_EDGE_1": (574.5768, 344.6603), "RED_EDGE_2": (579.0032, 315.089), "RED_EDGE_3": (582.2367, 293.4251), "NIR_BROAD": (494.48, 245.161)
    },
    "s2": {
        "COASTAL_AEROSOL": (2618.7167, 605.6178), "BLUE": (2428.1283, 703.0092), "GREEN": (2367.1218, 746.6847), "RED": (2460.8811, 965.7502),
        "RED_EDGE_1": (2632.0329, 919.4613), "RED_EDGE_2": (3025.0125, 901.2289), "RED_EDGE_3": (3228.6377, 941.6696), "NIR_BROAD": (3182.298, 938.9032),
        "NIR_NARROW": (3394.7605, 983.7676), "WATER_VAPOR": (1944.458, 501.031), "CIRRUS": (1023.7026, 30.0558), "SWIR_1": (3239.4603, 1101.8826), "SWIR_2": (2664.1579, 986.4062)
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