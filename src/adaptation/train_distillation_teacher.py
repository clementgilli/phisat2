import os
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

# --- Custom Imports ---
from src.models.geoaware_foundation import phisat2net_geoaware
from src.dataset.dataset import get_dataloaders
from src.utils import set_seed, load_yaml_config
from src.models.teacher_distillation_model import DomainFeatureDistiller, Phisat2FeatureExtractor

def parse_args():
    parser = argparse.ArgumentParser(description="Cross-Domain Feature Distillation Training")
    
    # Configuration and Paths
    parser.add_argument("--config", type=str, default="./configs/config.yaml", help="Path to the YAML config file")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save weights")
    
    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=50, help="Maximum number of pseudo-epochs")
    parser.add_argument("--steps_per_epoch", type=int, default=1000, help="Number of training batches per pseudo-epoch")
    parser.add_argument("--val_steps", type=int, default=200, help="Number of validation batches per pseudo-epoch")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--use_scheduler", action="store_true", help="Enable Cosine Annealing Learning Rate Scheduler")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (in epochs)")
    
    # Ablation and Strategy
    parser.add_argument("--baseline_mode", type=float, choices=[0.0, 0.5, 1.0], default=0.5, 
                        help="Data normalization baseline mode (0, 0.5, or 1)")
    parser.add_argument("--loss_type", type=str, choices=["cosine", "mse"], default="cosine", 
                        help="Distillation loss type")
    parser.add_argument("--disable_predictor", action="store_true", 
                        help="Disable the SimSiam predictor (Direct Encoder alignment)")
    parser.add_argument("--lambda_predictor", type=float, default=1.0, 
                        help="Relative learning rate for the predictor compared to the student encoder")
    
    # Environment
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")

    return parser.parse_args()

def get_infinite_iterator(dataloader):
    """Yields batches indefinitely to allow pseudo-epoch training."""
    while True:
        for batch in dataloader:
            yield batch

def plot_and_save_loss(train_losses, val_losses, save_path, loss_type):
    """Generates and saves the training and validation loss plot."""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, color='blue', linewidth=1.5, label=f'Train Loss ({loss_type.upper()})')
    plt.plot(val_losses, color='orange', linewidth=1.5, label=f'Val Loss ({loss_type.upper()})', linestyle='--')
    plt.title(f"Cross-Domain Distillation: Train vs Val Loss")
    plt.xlabel("Pseudo-Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] Loss plot saved to {save_path}")

@torch.no_grad()
def validate(distiller, val_iter, device, num_steps):
    """Runs evaluation on a subset of the validation set."""
    distiller.eval()
    val_loss_total = 0.0
    
    for _ in range(num_steps):
        batch = next(val_iter)
        img_real = batch['real'].to(device)
        img_sim = batch['sim'].to(device)
        
        loss = distiller(img_real, img_sim)
        val_loss_total += loss.item()
        
    return val_loss_total / num_steps

def main():
    args = parse_args()
    
    # 1. Initialization & Setup
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    run_name = f"distill_{args.loss_type}_pred{'OFF' if args.disable_predictor else 'ON'}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    run_dir = os.path.join(args.checkpoint_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    plot_path = os.path.join(run_dir, "training_validation_curve.png")
    
    config = load_yaml_config(args.config)
    
    # 2. Data Loading
    print(f"[INFO] Initializing DataLoaders...")
    train_loader, val_loader, _ = get_dataloaders(
        config=config,
        baseline_mode=args.baseline_mode,
        batch_size=args.batch_size, 
        num_workers=args.num_workers
    )
    
    train_iter = get_infinite_iterator(train_loader)
    val_iter = get_infinite_iterator(val_loader)
    
    # 3. Model Initialization
    print("[INFO] Loading Foundation Model...")
    base_model = phisat2net_geoaware(**config["model"])
    state_dict = torch.load(config["paths"]["model_weights"], map_location='cpu', weights_only=True)
    base_model.load_state_dict(state_dict)
    base_model.to(device)
    base_model.eval()
    
    # 4. Distillation Architecture Setup
    pure_encoder = Phisat2FeatureExtractor(base_model)
    distiller = DomainFeatureDistiller(
        encoder=pure_encoder, 
        feature_dim=config["model"]["dims"][-1],
        use_predictor=not args.disable_predictor,
        loss_type=args.loss_type
    ).to(device)
    
    # 5. Optimizer & Scheduler
    trainable_params = list(distiller.student_encoder.parameters()) + list(distiller.predictor.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    
    optimizer = torch.optim.AdamW([
        {'params': distiller.student_encoder.parameters(), 'lr': args.lr},
        
        {'params': distiller.predictor.parameters(), 'lr': args.lr * args.lambda_predictor}
    ], weight_decay=1e-4)
    
    if args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None
        
    # 6. Training Loop with Early Stopping
    print(f"[INFO] Starting Training: {args.epochs} pseudo-epochs, {args.steps_per_epoch} steps/epoch.")
    
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.epochs):
        distiller.train() 
        epoch_train_loss = 0.0
        
        # --- Training Phase ---
        pbar = tqdm(range(args.steps_per_epoch), desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for _ in pbar:
            batch = next(train_iter)
            img_real = batch['real'].to(device)
            img_sim = batch['sim'].to(device)
            
            optimizer.zero_grad()
            loss = distiller(img_real, img_sim)
            loss.backward()
            optimizer.step()
            
            loss_val = loss.item()
            epoch_train_loss += loss_val
            pbar.set_postfix({'loss': f"{loss_val:.4f}"})
            
        avg_train_loss = epoch_train_loss / args.steps_per_epoch
        train_losses.append(avg_train_loss)
        
        # --- Validation Phase ---
        avg_val_loss = validate(distiller, val_iter, device, args.val_steps)
        val_losses.append(avg_val_loss)
        
        print(f"[INFO] Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if scheduler is not None:
            scheduler.step()
            
        # --- Early Stopping & Checkpointing ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_ckpt_path = os.path.join(run_dir, "student_encoder_best.pth")
            torch.save(distiller.student_encoder.state_dict(), best_ckpt_path)
            plot_and_save_loss(train_losses, val_losses, plot_path, args.loss_type)
            print(f"[INFO] New best model saved! (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"[INFO] No improvement. Patience: {patience_counter}/{args.patience}")
            
        if patience_counter >= args.patience:
            print(f"[INFO] Early stopping triggered after {epoch+1} epochs.")
            break
        
    # 7. Finalization
    print("[INFO] Pipeline executed successfully.")

if __name__ == "__main__":
    main()