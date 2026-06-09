#!/bin/bash

set -e

source .venv/bin/activate

echo "-> LULC..."
make submit-eval EXPERIMENT=configs/lulc_full.mk CKPT_PATH=/lustre/home/u10010021/phisat2/runs/segmentation/lulc/phisatnet/full_dataset/seed_42/checkpoints/best.ckpt

echo "-> Roads..."
make submit-eval EXPERIMENT=configs/roads_full.mk CKPT_PATH=/lustre/home/u10010021/phisat2/runs/pixel_regression/roads/phisatnet/full_dataset/seed_42/checkpoints/best.ckpt

echo "-> Buildings..."
make submit-eval EXPERIMENT=configs/building_full.mk CKPT_PATH=/lustre/home/u10010021/phisat2/runs/pixel_regression/building/phisatnet/full_dataset/seed_42/checkpoints/best.ckpt

echo "-> DA..."
make submit-eval EXPERIMENT=configs/domain_adaptation.mk CKPT_PATH=/lustre/home/u10010021/phisat2/runs/domain_adaptation/triplets/phisatnet/full_dataset/seed_42/checkpoints/best-v2.ckpt

wandb sync /lustre/home/u10010021/phisat2/runs/segmentation/lulc/phisatnet/full_dataset/seed_42/wandb/offline-run-20260609_112626-8e5w9rgv
wandb sync /lustre/home/u10010021/phisat2/runs/pixel_regression/roads/phisatnet/full_dataset/seed_42/wandb/offline-run-20260609_125901-y3lon7e0
wandb sync /lustre/home/u10010021/phisat2/runs/pixel_regression/building/phisatnet/full_dataset/seed_42/wandb/offline-run-20260609_125901-qai6w1im
wandb sync /lustre/home/u10010021/phisat2/runs/domain_adaptation/triplets/phisatnet/full_dataset/seed_42/wandb/offline-run-20260609_163428-ivqzkzq5

echo "Done."