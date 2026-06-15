#!/bin/bash

set -e

source .venv/bin/activate

make submit-eval EXPERIMENT=configs/lulc_full.mk CKPT_PATH=/lustre/home/u10010021/phisat2/runs/segmentation/lulc/phisatnet/full_dataset/seed_42/checkpoints/best.ckpt
make submit-eval EXPERIMENT=configs/floods_full.mk CKPT_PATH=/lustre/home/u10010021/phisat2/runs/segmentation/floods/phisatnet/full_dataset/seed_42/checkpoints/best-v1.ckpt
make submit-eval EXPERIMENT=configs/burned_area_full.mk CKPT_PATH=/lustre/home/u10010021/phisat2/runs/segmentation/burned/phisatnet/full_dataset/seed_42/checkpoints/best.ckpt
make submit-eval EXPERIMENT=configs/roads_full.mk CKPT_PATH=/lustre/home/u10010021/phisat2/runs/pixel_regression/roads/phisatnet/full_dataset/seed_42/checkpoints/best.ckpt
make submit-eval EXPERIMENT=configs/building_full.mk CKPT_PATH=/lustre/home/u10010021/phisat2/runs/pixel_regression/building/phisatnet/full_dataset/seed_42/checkpoints/best.ckpt
make submit-eval EXPERIMENT=configs/clouds_full.mk CKPT_PATH=/lustre/home/u10010021/phisat2/runs/segmentation/clouds/phisatnet/full_dataset/seed_42/checkpoints/best.ckpt

#make submit-eval EXPERIMENT=configs/fire_full.mk CKPT_PATH=/lustre/home/u10010021/phisat2/runs/classification/fire/phisatnet/full_dataset/seed_42/checkpoints/best.ckpt

make submit-eval-domain-gap EXPERIMENT=configs/domain_adaptation.mk CKPT_PATH=/lustre/home/u10010021/phisat2/runs/domain_adaptation/triplets/phisatnet/full_dataset/seed_42/checkpoints/best.ckpt

#wandb sync /lustre/home/u10010021/phisat2/runs/segmentation/lulc/phisatnet/full_dataset/seed_42/wandb/latest-run
#wandb sync /lustre/home/u10010021/phisat2/runs/segmentation/floods/phisatnet/full_dataset/seed_42/wandb/latest-run
#wandb sync /lustre/home/u10010021/phisat2/runs/segmentation/burned/phisatnet/full_dataset/seed_42/wandb/latest-run
#wandb sync /lustre/home/u10010021/phisat2/runs/pixel_regression/roads/phisatnet/full_dataset/seed_42/wandb/latest-run
#wandb sync /lustre/home/u10010021/phisat2/runs/pixel_regression/building/phisatnet/full_dataset/seed_42/wandb/latest-run
#wandb sync /lustre/home/u10010021/phisat2/runs/domain_adaptation/triplets/phisatnet/full_dataset/seed_42/wandb/latest-run

echo "Done."