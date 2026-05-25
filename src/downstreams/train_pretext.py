import os
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from datetime import datetime

# --- Custom Imports ---
from src.dataset.dataset import get_dataloaders
from src.utils import set_seed, load_yaml_config
from src.models.student import GlobalHeadDecoder, PhisatNetEncoder, PhisatNetDecoder

# ==========================================
# 0. UTILS
# ==========================================
def get_infinite_iterator(dataloader):
    """Yields batches indefinitely to allow pseudo-epoch training."""
    while True:
        for batch in dataloader:
            yield batch

# ==========================================
# 1. GEOLOCATION MATH UTILS
# ==========================================
def encode_geolocation(lat: torch.Tensor, lon: torch.Tensor) -> torch.Tensor:
    u = lat / 90.0
    v = lon / 180.0
    sin_u, cos_u = torch.sin(torch.pi * u), torch.cos(torch.pi * u)
    sin_v, cos_v = torch.sin(torch.pi * v), torch.cos(torch.pi * v)
    return torch.stack([sin_u, cos_u, sin_v, cos_v], dim=1)

# ==========================================
# 2. TRAINING PIPELINE
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train Downstream Task Probes on SIM data")
    
    parser.add_argument("--task", type=str, required=True, choices=["climate", "geoloc", "reconstruction"])
    parser.add_argument("--config", type=str, default="./configs/config.yaml")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--encoder_weights", type=str, default="weights/encoder_sim_base.pth")
    
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps_per_epoch", type=int, default=1000)
    parser.add_argument("--val_steps", type=int, default=200)
    
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Initializing {args.task.upper()} probing on {device}...")
    
    config = load_yaml_config(args.config)
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    run_name = f"probe_{args.task}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    run_dir = os.path.join(args.checkpoint_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    with open(os.path.join(run_dir, "training_args.yaml"), 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)
        
    print(f"[INFO] Loading SIM Dataloaders...")
    train_loader, val_loader, _ = get_dataloaders(
        config=config, baseline_mode=0.0, batch_size=args.batch_size, num_workers=args.num_workers
    )
    train_iter = get_infinite_iterator(train_loader)
    val_iter = get_infinite_iterator(val_loader)
    
    cfg_base_filters = 16
    cfg_depth = 3
    cfg_channel_mults = [1, 2, 4, 8]
    bot_channels = cfg_base_filters * cfg_channel_mults[-1] # 16 * 8 = 128

    print("[INFO] Loading Foundation Encoder...")
    encoder = PhisatNetEncoder(n_channels=8, base_filters=cfg_base_filters, depth=cfg_depth, channel_multipliers=cfg_channel_mults).to(device)
    encoder.load_state_dict(torch.load(args.encoder_weights, map_location=device, weights_only=True))
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    print(f"[INFO] Initializing Decoder for task: {args.task}")
    if args.task == "climate":
        decoder = GlobalHeadDecoder(in_channels=bot_channels, out_features=31).to(device)
        climate_weights = torch.tensor(config["training"]["climate_loss_weights"], dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=climate_weights)
        
    elif args.task == "geoloc":
        decoder = GlobalHeadDecoder(in_channels=bot_channels, out_features=4).to(device)
        criterion = nn.MSELoss()
        
    elif args.task == "reconstruction":
        decoder = PhisatNetDecoder(n_classes=8, base_filters=cfg_base_filters, depth=cfg_depth, channel_multipliers=cfg_channel_mults).to(device)
        criterion = nn.L1Loss()
        
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=1e-4)
    print(f"[INFO] Trainable parameters: {sum(p.numel() for p in decoder.parameters()):,}")

    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.epochs):
        decoder.train()
        train_loss = 0.0
        
        pbar = tqdm(range(args.steps_per_epoch), desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for _ in pbar:
            batch = next(train_iter)
            img_sim = batch['sim'].to(device) 
            
            optimizer.zero_grad()
            
            with torch.no_grad():
                feats = encoder(img_sim)
                
            preds = decoder(feats)
            
            if args.task == "climate":
                targets = batch['climate'].to(device)
                loss = criterion(preds, targets)
            elif args.task == "geoloc":
                targets = encode_geolocation(batch['lat'].to(device), batch['lon'].to(device))
                loss = criterion(preds, targets)
            elif args.task == "reconstruction":
                loss = criterion(preds, img_sim)

            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_train_loss = train_loss / args.steps_per_epoch
        
        decoder.eval()
        val_loss = 0.0
        with torch.no_grad():
            for _ in range(args.val_steps):
                batch = next(val_iter)
                img_sim = batch['sim'].to(device)
                
                feats = encoder(img_sim)
                preds = decoder(feats)
                
                if args.task == "climate":
                    loss = criterion(preds, batch['climate'].to(device))
                elif args.task == "geoloc":
                    loss = criterion(preds, encode_geolocation(batch['lat'].to(device), batch['lon'].to(device)))
                elif args.task == "reconstruction":
                    loss = criterion(preds, img_sim)
                    
                val_loss += loss.item()
                
        avg_val_loss = val_loss / args.val_steps
        print(f"[INFO] Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(decoder.state_dict(), os.path.join(run_dir, f"probe_{args.task}_best.pth"))
            print(f"[INFO] Best model saved! (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"[INFO] Early stopping triggered after {epoch+1} epochs.")
                break

if __name__ == "__main__":
    main()