import argparse
import os
import torch
import yaml
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.models.geoaware_foundation import phisat2net_geoaware
from src.dataset.dataset import get_dataloaders
from src.utils import set_seed, load_yaml_config, plot_tsne
from src.eval.evaluator_teacher import MultiTaskEvaluator
from src.models.teacher_distillation_model import Phisat2FeatureExtractor

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GeoAware Model for Domain Adaptation")
    
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to the config file")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for the dataloader")
    
    parser.add_argument("--num_batches", type=int, default=None, 
                        help="Number of batches to process. Leave empty to evaluate the ENTIRE dataset.")
    parser.add_argument("--no_lpips", action="store_true", 
                        help="Disable LPIPS calculation to speed up evaluation")
    parser.add_argument("--out_dir", type=str, default=".", 
                        help="Directory to save the output files")
    
    parser.add_argument("--normalization_mode", type=float, default=1., choices=[0., 0.5, 1.], 
                        help="Normalization mode for the data (0: Normalized REAL and SIM by SIM stats, 0.5: Normalized each domain by its own stats, 1: OT)")
    
    parser.add_argument("--student_encoder_path", type=str, default=None,
                        help="Path to the trained student encoder weights. If None, uses the base model for both domains.")
    
    return parser.parse_args()

def main(args):
    config = load_yaml_config(args.config)
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading baseline model weights (SIM)...")
    model = phisat2net_geoaware(**config["model"])
    state_dict = torch.load(config["paths"]["model_weights"], map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Logique de chargement conditionnel pour TARGET (REAL)
    if args.student_encoder_path:
        print(f"Loading student encoder weights for TARGET (REAL) from {args.student_encoder_path}...")
        model_real = phisat2net_geoaware(**config["model"])
        model_real.load_state_dict(state_dict) # Base weights pour le décodeur et les têtes
        
        student_encoder = Phisat2FeatureExtractor(model_real)
        student_encoder.load_state_dict(torch.load(args.student_encoder_path, map_location='cpu', weights_only=True))
        
        model_real.to(device)
        model_real.eval()
    else:
        print("No student encoder specified. Using baseline model for both domains.")
        model_real = model

    print(f"Initializing dataloaders (Batch Size: {args.batch_size}, Workers: {args.num_workers}, Normalization Mode: {args.normalization_mode})...")
    
    _, _, test_loader = get_dataloaders(
        config=config,
        baseline_mode=args.normalization_mode,
        batch_size=args.batch_size, 
        num_workers=args.num_workers
    )

    total_batches_in_loader = len(test_loader)
    max_batches = args.num_batches if args.num_batches is not None else total_batches_in_loader
    max_batches = min(max_batches, total_batches_in_loader)

    print(f"Starting evaluation on {max_batches} batches...")

    all_sim_features = []
    all_real_features = []
    
    use_lpips = not args.no_lpips 
    evaluator = MultiTaskEvaluator(config, use_lpips=use_lpips, device=device)
    evaluator.reset()

    for i, batch in tqdm(enumerate(test_loader), total=max_batches):
        if i >= max_batches:
            break
            
        feat_sim, feat_real = evaluator.evaluate_paired_batch(model, batch, model_real=model_real, return_features=True)
        
        all_sim_features.append(feat_sim.cpu().numpy())
        all_real_features.append(feat_real.cpu().numpy())

    print("\nExtracting features and generating t-SNE plot...")
    X_sim = np.vstack(all_sim_features)
    X_real = np.vstack(all_real_features)
    
    MAX_TSNE_SAMPLES = 4000
    if len(X_sim) > MAX_TSNE_SAMPLES:
        print(f"Subsampling features for t-SNE (Keeping {MAX_TSNE_SAMPLES} per domain)...")
        idx_sim = np.random.choice(len(X_sim), MAX_TSNE_SAMPLES, replace=False)
        idx_real = np.random.choice(len(X_real), MAX_TSNE_SAMPLES, replace=False)
        
        X_sim_plot = X_sim[idx_sim]
        X_real_plot = X_real[idx_real]
    else:
        X_sim_plot = X_sim
        X_real_plot = X_real
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tsne_filename = f"tsne_eval_{timestamp}.png"
    tsne_filepath = os.path.join(args.out_dir, tsne_filename)
    
    plot_tsne(X_sim_plot, X_real_plot, save_path=tsne_filepath)

    print("Computing PAD Score...")
    evaluator.compute_PAD_score(X_sim, X_real)
    evaluator.print_report()
    evaluator.save_report(os.path.join(args.out_dir, f"evaluation_report_{timestamp}.json"))

if __name__ == "__main__":
    args = parse_args()
    main(args)