import os
import argparse
import yaml
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

from src.dataset.dataset import get_dataloaders
from src.utils import set_seed, load_yaml_config
from src.models.student import PhisatNetEncoder

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Scale Cross-Domain Feature Distillation")
    
    # Configuration and Paths
    parser.add_argument("--config", type=str, default="./configs/config.yaml", help="Path to the YAML config file")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save weights")
    parser.add_argument("--encoder_weights", type=str, default="weights/encoder_sim_base.pth", help="Path to the clean SIM encoder weights")
    
    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=50, help="Maximum number of pseudo-epochs")
    parser.add_argument("--steps_per_epoch", type=int, default=1000, help="Number of training batches per pseudo-epoch")
    parser.add_argument("--val_steps", type=int, default=200, help="Number of validation batches per pseudo-epoch")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--use_scheduler", action="store_true", help="Enable Cosine Annealing Learning Rate Scheduler")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (in epochs)")
    
    # Baseline Strategy
    parser.add_argument("--baseline_mode", type=float, choices=[0.0, 0.5, 1.0], default=0.5, 
                        help="Data normalization baseline mode (0, 0.5, or 1)")
    
    # Environment
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")

    return parser.parse_args()

def get_infinite_iterator(dataloader):
    while True:
        for batch in dataloader:
            yield batch

def plot_and_save_loss(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, color='blue', linewidth=1.5, label='Train Loss (MSE)')
    plt.plot(val_losses, color='orange', linewidth=1.5, label='Val Loss (MSE)', linestyle='--')
    plt.title("Multi-Scale Cross-Domain Distillation")
    plt.xlabel("Pseudo-Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

@torch.no_grad()
def validate(model_real, model_sim, val_iter, device, num_steps, loss_weights, mse_fn):
    model_real.eval()
    val_loss_total = 0.0
    
    for _ in tqdm(range(num_steps), desc="Validating", leave=False):
        batch = next(val_iter)
        img_real = batch['real'].to(device)
        img_sim = batch['sim'].to(device)
        
        feat_sim = model_sim(img_sim)
        feat_real = model_real(img_real)
        
        batch_loss = 0.0
        for layer_name in feat_sim.keys():
            loss = mse_fn(feat_real[layer_name], feat_sim[layer_name])
            batch_loss += loss.item() * loss_weights.get(layer_name, 1.0)
            
        val_loss_total += batch_loss
        
    return val_loss_total / num_steps

def main():
    args = parse_args()
    
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    run_name = f"distill_multiscale_mse_{datetime.now().strftime('%Y%m%d_%H%M')}"
    run_dir = os.path.join(args.checkpoint_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    plot_path = os.path.join(run_dir, "training_validation_curve.png")
    
    args_path = os.path.join(run_dir, "training_args.yaml")
    with open(args_path, 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)
    
    config = load_yaml_config(args.config)
    
    print(f"[INFO] Initializing DataLoaders...")
    train_loader, val_loader, _ = get_dataloaders(
        config=config,
        baseline_mode=args.baseline_mode,
        batch_size=args.batch_size, 
        num_workers=args.num_workers
    )
    
    train_iter = get_infinite_iterator(train_loader)
    val_iter = get_infinite_iterator(val_loader)
    
    print("[INFO] Initializing Native Encoder Architectures...")
    cfg_base_filters = 16
    cfg_depth = 3
    cfg_channel_mults = [1, 2, 4, 8]
    
    # Teacher (SIM) - Fully Frozen
    model_sim = PhisatNetEncoder(n_channels=8, base_filters=cfg_base_filters, depth=cfg_depth, channel_multipliers=cfg_channel_mults).to(device)
    model_sim.load_state_dict(torch.load(args.encoder_weights, map_location=device, weights_only=True))
    model_sim.eval()
    for param in model_sim.parameters():
        param.requires_grad = False
        
    # Student (REAL) - Trainable, initialized with SIM weights
    model_real = PhisatNetEncoder(n_channels=8, base_filters=cfg_base_filters, depth=cfg_depth, channel_multipliers=cfg_channel_mults).to(device)
    model_real.load_state_dict(torch.load(args.encoder_weights, map_location=device, weights_only=True))
    for param in model_real.parameters():
        param.requires_grad = True

    mse_loss_fn = nn.MSELoss()
    
    loss_weights = {
        "enc_0": 1.0,  
        "enc_1": 1.0,  
        "enc_2": 1.0,
        "bottleneck": 1.0  
    }
    print(f"[INFO] Using Loss Weights: {loss_weights}")
    
    trainable_params = [p for p in model_real.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    
    if args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None
        
    print(f"[INFO] Starting Training: {args.epochs} pseudo-epochs, {args.steps_per_epoch} steps/epoch.")
    
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.epochs):
        model_real.train()
        epoch_train_loss = 0.0
        
        pbar = tqdm(range(args.steps_per_epoch), desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for _ in pbar:
            batch = next(train_iter)
            img_real = batch['real'].to(device)
            img_sim = batch['sim'].to(device)
            
            optimizer.zero_grad()
            
            with torch.no_grad():
                feat_sim = model_sim(img_sim)
                
            feat_real = model_real(img_real)
            
            loss = 0.0
            for layer_name in feat_sim.keys():
                layer_loss = mse_loss_fn(feat_real[layer_name], feat_sim[layer_name])
                loss += layer_loss * loss_weights.get(layer_name, 1.0)
                
            loss.backward()
            optimizer.step()
            
            loss_val = loss.item()
            epoch_train_loss += loss_val
            pbar.set_postfix({'loss': f"{loss_val:.4f}"})
            
        avg_train_loss = epoch_train_loss / args.steps_per_epoch
        train_losses.append(avg_train_loss)
        
        avg_val_loss = validate(model_real, model_sim, val_iter, device, args.val_steps, loss_weights, mse_loss_fn)
        val_losses.append(avg_val_loss)
        
        print(f"[INFO] Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if scheduler is not None:
            scheduler.step()
            
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_ckpt_path = os.path.join(run_dir, "student_encoder_only_best.pth")
            torch.save(model_real.state_dict(), best_ckpt_path)
            plot_and_save_loss(train_losses, val_losses, plot_path)
            print(f"[INFO] New best model saved! (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"[INFO] No improvement. Patience: {patience_counter}/{args.patience}")
            
        if patience_counter >= args.patience:
            print(f"[INFO] Early stopping triggered after {epoch+1} epochs.")
            break
    
    plot_and_save_loss(train_losses, val_losses, plot_path)
    print("[INFO] Pipeline executed successfully.")

if __name__ == "__main__":
    main()