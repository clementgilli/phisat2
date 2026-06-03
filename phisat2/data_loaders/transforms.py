from __future__ import annotations

import torch

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
