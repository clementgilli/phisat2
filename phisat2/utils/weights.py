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

def load_encoder_weights(encoder: nn.Module, path: str, adapt_13_to_8: bool = False) -> None:
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

    if adapt_13_to_8:
        print("[WeightLoader] 13 bands -> 8 bands")
        s2_idx = {"BLUE": 1, "GREEN": 2, "RED": 3, "RE1": 4, "RE2": 5, "RE3": 6, "NIR": 7}
        
        first_conv_key = None
        for k, v in encoder_dict.items():
            if v.ndim == 4 and v.shape[1] == 13:
                first_conv_key = k
                break
                
        if first_conv_key:
            teacher_w = encoder_dict[first_conv_key]
            
            student_w = torch.zeros(
                (teacher_w.size(0), 8, teacher_w.size(2), teacher_w.size(3)),
                device=teacher_w.device
            )
            
            student_w[:, 0, :, :] = teacher_w[:, [s2_idx["BLUE"], s2_idx["GREEN"], s2_idx["RED"]], :, :].mean(dim=1)
            
            student_w[:, 1, :, :] = teacher_w[:, s2_idx["BLUE"], :, :]
            student_w[:, 2, :, :] = teacher_w[:, s2_idx["GREEN"], :, :]
            student_w[:, 3, :, :] = teacher_w[:, s2_idx["RED"], :, :]
            student_w[:, 4, :, :] = teacher_w[:, s2_idx["RE1"], :, :]
            student_w[:, 5, :, :] = teacher_w[:, s2_idx["RE2"], :, :]
            student_w[:, 6, :, :] = teacher_w[:, s2_idx["RE3"], :, :]
            student_w[:, 7, :, :] = teacher_w[:, s2_idx["NIR"], :, :]
            
            encoder_dict[first_conv_key] = student_w
        else:
            raise RuntimeError("Failed to adapt weights from 13 to 8 bands.")

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