import re
import torch
import torch.nn as nn


def _strip_compile_prefix(state_dict: dict) -> dict:
    """
    model._orig_mod.encoder.encoders.0.weight
    → model.encoder.encoders.0.weight
    """
    if not any("._orig_mod." in k for k in state_dict):
        return state_dict
    return {k.replace("._orig_mod.", "."): v for k, v in state_dict.items()}

def load_encoder_weights(encoder: nn.Module, path: str) -> None:
    print(f"[WeightLoader] encoder  ← {path}")
    ckpt       = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = _strip_compile_prefix(ckpt.get("state_dict", ckpt))

    has_student = any("student." in k for k in state_dict)

    encoder_dict = {}
    for k, v in state_dict.items():
        if has_student and "student." not in k:
            continue
        match = re.search(r'(encoders|bottleneck)\..+', k)
        if match:
            encoder_dict[match.group(0)] = v

    if not encoder_dict:
        raise RuntimeError(f"No encoder weights found in '{path}'.")

    encoder.load_state_dict(encoder_dict, strict=True)
    print(f"[WeightLoader] OK — {len(encoder_dict)} loaded.")

def load_decoder_weights(decoder: nn.Module, path: str) -> None:
    print(f"[WeightLoader] decoder  ← {path}")
    ckpt       = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = _strip_compile_prefix(ckpt.get("state_dict", ckpt))

    decoder_dict = {}
    for k, v in state_dict.items():
        clean = re.sub(r'^(model\.|student\.|teacher\.)+', '', k)
        if clean.startswith("head."):
            decoder_dict[clean[len("head."):]] = v

    if not decoder_dict:
        raise RuntimeError(f"No head weights found in '{path}'.")

    decoder.load_state_dict(decoder_dict, strict=True)
    print(f"[WeightLoader] OK — {len(decoder_dict)} loaded.")