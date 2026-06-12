from __future__ import annotations

import torch

def apply_spatial_transforms(
    tensors: list[torch.Tensor], 
    is_train: bool, 
    crop_size: int = 224
) -> list[torch.Tensor]:
    
    if not tensors or tensors[0].numel() == 0:
        return tensors
        
    _, H, W = tensors[0].shape
    
    if is_train:
        top = int(torch.randint(0, H - crop_size + 1, (1,)).item()) if H > crop_size else 0
        left = int(torch.randint(0, W - crop_size + 1, (1,)).item()) if W > crop_size else 0
        flip_h = torch.rand(1).item() > 0.5
        flip_v = torch.rand(1).item() > 0.5
    else:
        # Center crop
        top = max(0, (H - crop_size) // 2)
        left = max(0, (W - crop_size) // 2)
        flip_h = False
        flip_v = False

    transformed = []
    for t in tensors:
        # Crop
        t = t[..., top : top + crop_size, left : left + crop_size]
        # Flips
        if flip_h:
            t = torch.flip(t, dims=[-1])
        if flip_v:
            t = torch.flip(t, dims=[-2])
        transformed.append(t)
        
    return transformed

def normalize_tensor(tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    if tensor.numel() > 0:
        return (tensor - mean) / std
    return tensor