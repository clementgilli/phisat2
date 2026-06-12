import argparse
import os
import torch
from pathlib import Path

def extract_and_clean_encoder(state_dict: dict, role: str) -> dict:
    clean_dict = {}
    
    if role == "student":
        prefixes_to_try = ["student._orig_mod.", "student."]
    else:
        prefixes_to_try = ["model._orig_mod.encoder.", "model.encoder."]
    
    chosen_prefix = None
    for p in prefixes_to_try:
        if any(k.startswith(p) for k in state_dict.keys()):
            chosen_prefix = p
            break
            
    if not chosen_prefix:
        return clean_dict

    for key, value in state_dict.items():
        if key.startswith(chosen_prefix):
            clean_key = key[len(chosen_prefix):]
            clean_dict[clean_key] = value
            
    return clean_dict

def extract_and_clean_decoder(state_dict: dict) -> tuple[dict, str]:
    clean_dict = {}
    possible_patterns = ["model._orig_mod.head.", "model.head.", "model._orig_mod.decoder.", "model.decoder."]
    
    chosen_prefix = None
    for p in possible_patterns:
        if any(k.startswith(p) for k in state_dict.keys()):
            chosen_prefix = p
            break
            
    if not chosen_prefix:
        return clean_dict, "Unknown"

    for key, value in state_dict.items():
        if key.startswith(chosen_prefix):
            clean_key = key[len(chosen_prefix):]
            clean_dict[clean_key] = value
            
    return clean_dict, chosen_prefix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_ckpt", type=str)
    parser.add_argument("--student_ckpt", type=str)
    parser.add_argument("--downstream_ckpts", nargs="+", help="Format: task=/path/ckpt.ckpt")
    parser.add_argument("--out_dir", type=str, default="exported_weights")

    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Exporting in {out_dir}/")

    if args.teacher_ckpt:
        ckpt = torch.load(args.teacher_ckpt, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        weights = extract_and_clean_encoder(state_dict, role="teacher")
        if weights:
            torch.save(weights, out_dir / "phisatnet_encoder_simulated.pt")
            print(f"Teacher exported !")
        else:
            print("Failed on Teacher.")

    if args.student_ckpt:
        ckpt = torch.load(args.student_ckpt, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        weights = extract_and_clean_encoder(state_dict, role="student")
        if weights:
            torch.save(weights, out_dir / "phisatnet_encoder_real.pt")
            print(f"Student exported !")
        else:
            print("Failed on Student.")

    if args.downstream_ckpts:
        for item in args.downstream_ckpts:
            if "=" not in item: continue
            task_name, ckpt_path = item.split("=", 1)
            
            if not os.path.exists(ckpt_path):
                print(f"Can't find {task_name} -> {ckpt_path}")
                continue
                
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("state_dict", ckpt)
            
            weights, prefix_detect = extract_and_clean_decoder(state_dict)
            if weights:
                out_path = out_dir / f"phisatnet_decoder_{task_name}.pt"
                torch.save(weights, out_path)
                print(f"Decoder {task_name.upper()} exported !")
            else:
                print("Failed on Decoder.")

    print("\nDone.")

if __name__ == "__main__":
    main()