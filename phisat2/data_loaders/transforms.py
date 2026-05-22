from __future__ import annotations

import torch

PHISAT2_SIM_MEAN = torch.tensor(
    [49.7866, 49.0253, 48.4297, 49.2364, 51.1648, 55.4065, 57.3572, 56.7808],
    dtype=torch.float32,
).view(8, 1, 1)
PHISAT2_SIM_STD = torch.tensor(
    [7.2800, 6.5203, 6.9570, 9.0981, 8.3858, 7.9555, 8.3155, 8.3664],
    dtype=torch.float32,
).view(8, 1, 1)
PHISAT2_REAL_MEAN = torch.tensor(
    [15.0339, 14.4876, 14.3599, 15.4217, 13.8730, 14.4105, 14.8086, 13.1281],
    dtype=torch.float32,
).view(8, 1, 1)
PHISAT2_REAL_STD = torch.tensor(
    [8.2109, 10.5295, 9.3784, 9.0989, 11.2457, 10.9468, 10.6382, 9.6175],
    dtype=torch.float32,
).view(8, 1, 1)


def normalize_sim_image(image: torch.Tensor) -> torch.Tensor:
    image = torch.sqrt(torch.clamp(image.float(), min=0.0))
    image = torch.clamp(image, max=100.0)
    return (image - PHISAT2_SIM_MEAN) / PHISAT2_SIM_STD


def normalize_real_image(image: torch.Tensor) -> torch.Tensor:
    image = torch.sqrt(torch.clamp(image.float(), min=0.0))
    return (image - PHISAT2_REAL_MEAN) / PHISAT2_REAL_STD


def crop_pair(
    image: torch.Tensor,
    target: torch.Tensor,
    crop_size: int,
    *,
    train: bool,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = image.shape[-2:]
    crop_h = min(crop_size, height)
    crop_w = min(crop_size, width)
    if train and height > crop_h:
        top = int(torch.randint(0, height - crop_h + 1, (1,), generator=generator).item())
    else:
        top = max(0, (height - crop_h) // 2)
    if train and width > crop_w:
        left = int(torch.randint(0, width - crop_w + 1, (1,), generator=generator).item())
    else:
        left = max(0, (width - crop_w) // 2)
    image = image[..., top : top + crop_h, left : left + crop_w]
    target = target[..., top : top + crop_h, left : left + crop_w] if target.ndim >= 2 else target
    return image, target
