from __future__ import annotations

import torch
import torch.nn.functional as F
import math
from phisat2.data_loaders.sensors import S2_BANDS, PHISAT2_REAL_BANDS, PAN_WEIGHTS

def upscale_to_phisat2(tensor: torch.Tensor, is_mask: bool = False) -> torch.Tensor:
    
    scale_factor = 10.0 / 4.75
    
    is_2d = tensor.ndim == 2
    t_4d = tensor.unsqueeze(0).unsqueeze(0) if is_2d else tensor.unsqueeze(0)
    
    if is_mask:
        t_interp = F.interpolate(t_4d.float(), scale_factor=scale_factor, mode='nearest').to(tensor.dtype)
    else:
        t_interp = F.interpolate(t_4d, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        
    return t_interp.squeeze(0).squeeze(0) if is_2d else t_interp.squeeze(0)

def extract_phisat2_bands(img_s2: torch.Tensor) -> torch.Tensor:
    out_bands = []
    
    for band_name in PHISAT2_REAL_BANDS:
        if band_name == "PAN":
            pan_band = torch.zeros_like(img_s2[0], dtype=torch.float32)
            for src_band, weight in PAN_WEIGHTS.items():
                if weight > 0.0:
                    idx = S2_BANDS.index(src_band)
                    pan_band += img_s2[idx] * weight
            out_bands.append(pan_band.to(img_s2.dtype))
        else:
            idx = S2_BANDS.index(band_name)
            out_bands.append(img_s2[idx])
            
    return torch.stack(out_bands, dim=0)

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
        k_rot  = int(torch.randint(0, 4, (1,)).item())
        
        crop_h = crop_size
        crop_w = crop_size
    else:
        test_crop_size = (min(H, W) // 32) * 32
        
        top = max(0, (H - test_crop_size) // 2)
        left = max(0, (W - test_crop_size) // 2)
        
        flip_h = False
        flip_v = False
        k_rot  = 0
        
        crop_h = test_crop_size
        crop_w = test_crop_size

    transformed = []
    for t in tensors:
        t = t[..., top : top + crop_h, left : left + crop_w]
        
        if flip_h:
            t = torch.flip(t, dims=[-1])
        if flip_v:
            t = torch.flip(t, dims=[-2])
            
        if k_rot > 0:
            t = torch.rot90(t, k=k_rot, dims=[-2, -1])
            
        transformed.append(t)
        
    return transformed

def apply_kd_transforms(
    tensor: torch.Tensor, 
    is_train: bool, 
    crop_size: int = 224,
    p_jitter: float = 0.5,
    p_noise: float = 0.5
) -> dict[str, torch.Tensor]:
    
    _, H, W = tensor.shape
    
    if is_train:
        top = int(torch.randint(0, H - crop_size + 1, (1,)).item()) if H > crop_size else 0
        left = int(torch.randint(0, W - crop_size + 1, (1,)).item()) if W > crop_size else 0
        flip_h = torch.rand(1).item() > 0.5
        flip_v = torch.rand(1).item() > 0.5
    else:
        top = max(0, (H - crop_size) // 2)
        left = max(0, (W - crop_size) // 2)
        flip_h = False
        flip_v = False

    t_view = tensor[..., top : top + crop_size, left : left + crop_size]
    
    if flip_h: 
        t_view = torch.flip(t_view, dims=[-1])
    if flip_v: 
        t_view = torch.flip(t_view, dims=[-2])

    if not is_train:
        return {"teacher": t_view, "student": t_view}

    s_view = t_view.clone()

    # A. Brightness & Micro Spectral Jitter
    if torch.rand(1).item() < p_jitter:
        brightness = torch.empty(1).uniform_(0.8, 1.2)
        s_view = s_view * brightness
        
        micro_jitter = torch.empty((s_view.shape[0], 1, 1)).uniform_(0.97, 1.03)
        s_view = s_view * micro_jitter

    # B. Additive Noise
    if torch.rand(1).item() < p_noise:
        noise = torch.randn_like(s_view) * 0.05
        s_view = s_view + noise

    return {
        "teacher": t_view,
        "student": s_view
    }

def normalize_tensor(tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    if tensor.numel() > 0:
        return (tensor - mean) / std
    return tensor