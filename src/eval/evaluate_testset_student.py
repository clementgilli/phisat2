import argparse
import os
from src.models.student import GlobalHeadDecoder, PhisatNetEncoder, PhisatNetDecoder
import torch
import yaml
from datetime import datetime
import numpy as np
from tqdm import tqdm

from src.dataset.dataset import get_dataloaders
from src.utils import set_seed, load_yaml_config, plot_tsne
from src.eval.evaluator_student import MultiTaskEvaluator

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GeoAware Model for Domain Adaptation")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_batches", type=int, default=None)
    parser.add_argument("--no_lpips", action="store_true")
    parser.add_argument("--out_dir", type=str, default="./outputs")
    parser.add_argument("--normalization_mode", type=float, default=0.5, choices=[0., 0.5, 1.])
    parser.add_argument("--student_encoder_path", type=str, default=None)
    return parser.parse_args()

def main(args):
    config = load_yaml_config(args.config)
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(f"{args.out_dir}/{timestamp}", exist_ok=True)
    
    cfg = config["model_student"]
    
    bot_channels = cfg["base_filters"] * cfg["channel_multipliers"][-1]

    encoder_sim = PhisatNetEncoder(n_channels=8, base_filters=cfg["base_filters"], depth=cfg["depth"], channel_multipliers=cfg["channel_multipliers"]).to(device)
    encoder_sim.load_state_dict(torch.load("weights_hub/encoder_sim_base.pth", weights_only=True))
    encoder_sim.eval()

    if args.student_encoder_path is not None:
        encoder_real = PhisatNetEncoder(n_channels=8, base_filters=cfg["base_filters"], depth=cfg["depth"], channel_multipliers=cfg["channel_multipliers"]).to(device)
        encoder_real.load_state_dict(torch.load(args.student_encoder_path, weights_only=True))
        encoder_real.eval()
    else:
        encoder_real = encoder_sim
        print("[INFO] Using the same encoder for real data as for sim data since no path was provided.")

    head_recon = PhisatNetDecoder(n_classes=8, base_filters=cfg["base_filters"], depth=cfg["depth"], channel_multipliers=cfg["channel_multipliers"]).to(device)
    head_recon.load_state_dict(torch.load("weights_hub/decoder_reconstruction.pth", weights_only=True))
    head_recon.eval()

    head_climate = GlobalHeadDecoder(in_channels=bot_channels, out_features=31).to(device)
    head_climate.load_state_dict(torch.load("weights_hub/decoder_climate.pth", weights_only=True))
    head_climate.eval()
    
    head_geoloc = GlobalHeadDecoder(in_channels=bot_channels, out_features=4).to(device)
    head_geoloc.load_state_dict(torch.load("weights_hub/decoder_geoloc.pth", weights_only=True))
    head_geoloc.eval()
    
    SEG_TASKS = {
        'lc': 11, 'anomaly_detection': 9, 'burned_area': 4, 
        'clouds': 2, 'fire': 3, 'worldfloods': 3
    }
    
    seg_heads = {}
    for task_name, n_classes in SEG_TASKS.items():
        head = PhisatNetDecoder(n_classes=n_classes, base_filters=cfg["base_filters"], depth=cfg["depth"], channel_multipliers=cfg["channel_multipliers"]).to(device)
        
        weights_path = f"weights_hub/decoder_{task_name}.pth" 
        head.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True), strict=False)
        head.eval()
        seg_heads[task_name] = head
    
    _, _, test_loader = get_dataloaders(config=config, baseline_mode=args.normalization_mode, batch_size=args.batch_size, num_workers=args.num_workers)

    total_batches = len(test_loader)
    max_batches = min(args.num_batches if args.num_batches else total_batches, total_batches)

    use_lpips = not args.no_lpips 
    evaluator = MultiTaskEvaluator(config, use_lpips=use_lpips, device=device)
    evaluator.reset()

    for i, batch in tqdm(enumerate(test_loader), total=max_batches):
        if i >= max_batches:
            break
            
        img_sim = batch['sim'].to(device)
        img_real = batch['real'].to(device)
        
        with torch.no_grad():
            
            feat_sim = encoder_sim(img_sim)
            feat_real = encoder_real(img_real)
            
            preds_sim = {}
            preds_real = {}
            
            preds_sim['reconstruction'] = head_recon(feat_sim)
            preds_sim['climate'] = head_climate(feat_sim)
            preds_sim['coords'] = head_geoloc(feat_sim)
            
            preds_real['reconstruction'] = head_recon(feat_real)
            preds_real['climate'] = head_climate(feat_real)
            preds_real['coords'] = head_geoloc(feat_real)
            
            for task_name, head in seg_heads.items():
                preds_sim[task_name] = head(feat_sim)
                preds_real[task_name] = head(feat_real)
            
        evaluator.evaluate_paired_batch(batch, preds_sim, preds_real, feat_sim, feat_real)

    evaluator.compute_PAD_scores()
    
    evaluator.print_report()
    evaluator.save_report(os.path.join(args.out_dir, timestamp, f"evaluation_report_{timestamp}.json"))

    MAX_TSNE_SAMPLES = 4000
    for layer in evaluator.feature_layers:
        X, y = evaluator.get_tsne_features(layer)
        
        X_sim_layer, X_real_layer = X[y == 0], X[y == 1]
        
        if len(X_sim_layer) > MAX_TSNE_SAMPLES:
            idx_sim = np.random.choice(len(X_sim_layer), MAX_TSNE_SAMPLES, replace=False)
            idx_real = np.random.choice(len(X_real_layer), MAX_TSNE_SAMPLES, replace=False)
            X_sim_plot, X_real_plot = X_sim_layer[idx_sim], X_real_layer[idx_real]
        else:
            X_sim_plot, X_real_plot = X_sim_layer, X_real_layer
            
        tsne_filepath = os.path.join(args.out_dir, timestamp, f"tsne_{layer}_{timestamp}.png")
        plot_tsne(X_sim_plot, X_real_plot, save_path=tsne_filepath)

if __name__ == "__main__":
    args = parse_args()
    main(args)