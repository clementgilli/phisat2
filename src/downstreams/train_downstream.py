from logging import root
import os
import argparse
import yaml
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

# --- Custom Imports ---
from src.dataset.dataset_downstreams import get_dataloaders
from src.utils import set_seed
from src.models.student import PhisatNetEncoder, PhisatNetDecoder, GlobalHeadDecoder

# ==========================================
# 0. TASK CONFIGURATION
# ==========================================
TASK_CONFIG = {
    "lulc":    {"num_classes": 11, "type": "seg"}, 
    "marine":  {"num_classes": 9,  "type": "seg"}, 
    "burned":  {"num_classes": 4,  "type": "seg"},  
    "clouds":  {"num_classes": 2,  "type": "seg"}, 
    "floods":  {"num_classes": 3,  "type": "seg"},
    "roads":   {"num_classes": 1,  "type": "reg"}, 
    "building":{"num_classes": 1,  "type": "reg"}, 
    "fire":    {"num_classes": 3,  "type": "classif"}
}

# ==========================================
# 1. UTILS
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train Downstream Decoders (Seg, Reg, Classif)")
    
    # Task & Paths
    parser.add_argument("--root_dir", type=str, default="/shared/projects/phisat2/data/huggingface", help="Root directory for datasets")
    parser.add_argument("--task", type=str, required=True, choices=list(TASK_CONFIG.keys()), help="Downstream task to train")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save weights")
    parser.add_argument("--encoder_weights", type=str, default="weights/encoder_sim_base.pth", help="Path to frozen encoder weights")
    
    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=50, help="Max pseudo-epochs")
    parser.add_argument("--steps_per_epoch", type=int, default=1000, help="Train steps per epoch")
    parser.add_argument("--val_steps", type=int, default=200, help="Validation steps per epoch")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--use_scheduler", action="store_true", help="Enable Cosine Annealing LR Scheduler")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    
    # Environment
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")

    return parser.parse_args()

def get_infinite_iterator(dataloader):
    while True:
        for batch in dataloader:
            yield batch

def plot_and_save_loss(train_losses, val_losses, save_path, task_name, task_type):
    plt.figure(figsize=(10, 5))
    loss_name = 'MSE Loss' if task_type == 'reg' else 'Cross-Entropy Loss'
    plt.plot(train_losses, color='blue', linewidth=1.5, label=f'Train Loss ({loss_name})')
    plt.plot(val_losses, color='orange', linewidth=1.5, label=f'Val Loss ({loss_name})', linestyle='--')
    plt.title(f"Downstream Training: {task_name.upper()} ({task_type})")
    plt.xlabel("Pseudo-Epoch")
    plt.ylabel(loss_name)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def format_target(batch_label, task_type, device):
    if task_type in ["seg", "classif"]:
        targets = batch_label.to(device).long()
        if targets.dim() == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)
    elif task_type == "reg":
        targets = batch_label.to(device).float()
        if targets.dim() == 3:
            targets = targets.unsqueeze(1) # [B, H, W] -> [B, 1, H, W]
    return targets

# ==========================================
# 2. VALIDATION LOOP
# ==========================================
@torch.no_grad()
def validate(encoder, decoder, val_iter, device, num_steps, task_type, criterion):
    decoder.eval()
    val_loss_total = 0.0
    
    for _ in tqdm(range(num_steps), desc="Validating", leave=False):
        batch = next(val_iter)
        images = batch['img'].to(device)
        targets = format_target(batch['label'], task_type, device)
            
        feats = encoder(images)
        preds = decoder(feats)

        loss = criterion(preds, targets)
        val_loss_total += loss.item()
        
    return val_loss_total / num_steps

# ==========================================
# 3. MAIN PIPELINE
# ==========================================
def main():
    args = parse_args()
    
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Initializing {args.task.upper()} downstream training on {device}...")
    
    # Setup directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    run_name = f"downstream_{args.task}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    run_dir = os.path.join(args.checkpoint_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    plot_path = os.path.join(run_dir, "training_validation_curve.png")
    
    # Save training args
    with open(os.path.join(run_dir, "training_args.yaml"), 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)
        
    task_specs = TASK_CONFIG[args.task]
    task_type = task_specs['type']
    
    # Load Dataloaders
    print(f"[INFO] Initializing DataLoaders for {args.task}...")
    train_loader, val_loader, _ = get_dataloaders(args.root_dir, task_name=args.task, batch_size=args.batch_size, num_workers=args.num_workers)    
    
    train_iter = get_infinite_iterator(train_loader)
    val_iter = get_infinite_iterator(val_loader)
    
    # Architecture Hyperparameters
    cfg_base_filters = 16
    cfg_depth = 3
    cfg_channel_mults = [1, 2, 4, 8]
    
    # --- 1. LOAD ENCODER (FROZEN) ---
    print("[INFO] Loading and Freezing Foundation Encoder...")
    encoder = PhisatNetEncoder(n_channels=8, base_filters=cfg_base_filters, depth=cfg_depth, channel_multipliers=cfg_channel_mults).to(device)
    encoder.load_state_dict(torch.load(args.encoder_weights, map_location=device, weights_only=True))
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    # --- 2. INITIALIZE DECODER (TRAINABLE) ---
    print(f"[INFO] Initializing {task_type.upper()} Decoder for {args.task} ({task_specs['num_classes']} outputs)")
    
    # Routing architecture based on task type
    if task_type in ["seg", "reg"]:
        decoder = PhisatNetDecoder(
            n_classes=task_specs['num_classes'], 
            base_filters=cfg_base_filters, 
            depth=cfg_depth, 
            channel_multipliers=cfg_channel_mults
        ).to(device)
    elif task_type == "classif":
        decoder = GlobalHeadDecoder(
            in_channels=cfg_base_filters * cfg_channel_mults[-1], 
            n_classes=task_specs['num_classes']
        ).to(device)
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    # --- 3. LOSS & OPTIMIZER ---
    # Routing loss based on task type
    if task_type in ["seg", "classif"]:
        criterion = nn.CrossEntropyLoss()
    else: # reg
        criterion = nn.MSELoss()
        
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=1e-4)
    
    if args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None
        
    print(f"[INFO] Starting Training: {args.epochs} epochs, {args.steps_per_epoch} steps/epoch.")
    print(f"[INFO] Trainable parameters: {sum(p.numel() for p in decoder.parameters()):,}")
    
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    
    # --- 4. TRAINING LOOP ---
    for epoch in range(args.epochs):
        decoder.train()
        epoch_train_loss = 0.0
        
        pbar = tqdm(range(args.steps_per_epoch), desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for _ in pbar:
            batch = next(train_iter)
            
            images = batch['img'].to(device)
            targets = format_target(batch['label'], task_type, device)
                            
            optimizer.zero_grad()
            
            # Forward pass
            with torch.no_grad():
                feats = encoder(images)
            preds = decoder(feats)
            
            # Compute Loss
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            
            loss_val = loss.item()
            epoch_train_loss += loss_val
            pbar.set_postfix({'loss': f"{loss_val:.4f}"})
            
        # Compile epoch metrics
        avg_train_loss = epoch_train_loss / args.steps_per_epoch
        train_losses.append(avg_train_loss)
        
        avg_val_loss = validate(encoder, decoder, val_iter, device, args.val_steps, task_type, criterion)
        val_losses.append(avg_val_loss)
        
        print(f"[INFO] Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Step scheduler if enabled
        if scheduler is not None:
            scheduler.step()
            
        # Model checkpointing & Plotting
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            best_ckpt_path = os.path.join(run_dir, f"decoder_{args.task}_best.pth")
            torch.save(decoder.state_dict(), best_ckpt_path)
            
            # Update plot with new best model
            plot_and_save_loss(train_losses, val_losses, plot_path, args.task, task_type)
            print(f"[INFO] New best model saved! (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"[INFO] No improvement. Patience: {patience_counter}/{args.patience}")
            
        if patience_counter >= args.patience:
            print(f"[INFO] Early stopping triggered after {epoch+1} epochs.")
            break
            
    plot_and_save_loss(train_losses, val_losses, plot_path, args.task, task_type)
    print("[INFO] Pipeline executed successfully.")

if __name__ == "__main__":
    main()