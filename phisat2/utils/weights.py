import torch
import torch.nn as nn
import re

def load_encoder_weights(encoder: nn.Module, path: str):
    print(f"[Weight Loader] Extraction de l'ENCODEUR depuis: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    
    has_student = any("student." in k for k in state_dict.keys())
    
    encoder_dict = {}
    for k, v in state_dict.items():
        
        if has_student and "student." not in k:
            continue
            
        match = re.search(r'(encoders|bottleneck)\..+', k)
        if match:
            clean_k = match.group(0)
            encoder_dict[clean_k] = v
            
    if not encoder_dict:
        raise RuntimeError(f"CRITICAL: No encoder weights found in '{path}'.")
        
    encoder.load_state_dict(encoder_dict, strict=True)
    print("SUCCESS: Encoder weights perfectly loaded and verified!")

def load_decoder_weights(decoder: nn.Module, path: str):
    print(f"\n[Weight Loader] Reading checkpoint: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    
    decoder_dict = {}
    for k, v in state_dict.items():
        k = re.sub(r'^(model\.|student\.|teacher\.|_orig_mod\.)+', '', k)
        
        if k.startswith("head."):
            decoder_dict[k.replace("head.", "", 1)] = v
            
    if not decoder_dict:
        raise RuntimeError(f"CRITICAL: No head weights found in '{path}'.")
        
    decoder.load_state_dict(decoder_dict, strict=True)
    print("SUCCESS: Decoder weights perfectly loaded and verified!")