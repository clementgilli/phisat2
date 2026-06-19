#!/bin/bash

set -e

source .venv/bin/activate

#make submit-eval EXPERIMENT=configs/lulc_full.mk CKPT_PATH=/lustre/home/u10010021/phisat2/runs/segmentation/lulc/phisatnet/full_dataset/seed_42/checkpoints/best-v1.ckpt
#make submit-eval EXPERIMENT=configs/floods_full.mk CKPT_PATH=/lustre/home/u10010021/phisat2/runs/segmentation/floods/phisatnet/full_dataset/seed_42/checkpoints/best-v1.ckpt
#make submit-eval EXPERIMENT=configs/burned_area_full.mk CKPT_PATH=/lustre/home/u10010021/phisat2/runs/segmentation/burned/phisatnet/full_dataset/seed_42/checkpoints/best-v1.ckpt
#make submit-eval EXPERIMENT=configs/roads_full.mk CKPT_PATH=/lustre/home/u10010021/phisat2/runs/pixel_regression/roads/phisatnet/full_dataset/seed_42/checkpoints/best-v1.ckpt
#make submit-eval EXPERIMENT=configs/building_full.mk CKPT_PATH=/lustre/home/u10010021/phisat2/runs/pixel_regression/building/phisatnet/full_dataset/seed_42/checkpoints/best-v1.ckpt
#make submit-eval EXPERIMENT=configs/clouds_full.mk CKPT_PATH=/lustre/home/u10010021/phisat2/runs/segmentation/clouds/phisatnet/full_dataset/seed_42/checkpoints/best-v1.ckpt
#make submit-eval EXPERIMENT=configs/router.mk CKPT_PATH=/lustre/home/u10010021/phisat2/runs/classification/router/phisatnet/full_dataset/seed_42/checkpoints/best-v1.ckpt


#make submit-eval-domain-gap EXPERIMENT=configs/domain_adaptation.mk CKPT_PATH=/lustre/home/u10010021/phisat2/runs/domain_adaptation/triplets/phisatnet/full_dataset/seed_42/checkpoints/best.ckpt

#wandb sync /lustre/home/u10010021/phisat2/runs/segmentation/lulc/phisatnet/full_dataset/seed_42/wandb/latest-run
#wandb sync /lustre/home/u10010021/phisat2/runs/segmentation/floods/phisatnet/full_dataset/seed_42/wandb/latest-run
#wandb sync /lustre/home/u10010021/phisat2/runs/segmentation/burned/phisatnet/full_dataset/seed_42/wandb/latest-run
#wandb sync /lustre/home/u10010021/phisat2/runs/pixel_regression/roads/phisatnet/full_dataset/seed_42/wandb/latest-run
#wandb sync /lustre/home/u10010021/phisat2/runs/pixel_regression/building/phisatnet/full_dataset/seed_42/wandb/latest-run
#wandb sync /lustre/home/u10010021/phisat2/runs/domain_adaptation/triplets/phisatnet/full_dataset/seed_42/wandb/latest-run

wandb sync /lustre/home/u10010021/phisat2/runs/segmentation/lulc/phisatnet/lulc_train_50_global/seed_42/wandb/latest-run
wandb sync /lustre/home/u10010021/phisat2/runs/segmentation/lulc/phisatnet/lulc_train_500_global/seed_42/wandb/latest-run

wandb sync /lustre/home/u10010021/phisat2/runs/segmentation/floods/phisatnet/floods_train_50_global/seed_42/wandb/latest-run
wandb sync /lustre/home/u10010021/phisat2/runs/segmentation/floods/phisatnet/floods_train_500_global/seed_42/wandb/latest-run

wandb sync /lustre/home/u10010021/phisat2/runs/segmentation/clouds/phisatnet/clouds_50shot/seed_42/wandb/latest-run
wandb sync /lustre/home/u10010021/phisat2/runs/segmentation/clouds/phisatnet/clouds_500shot/seed_42/wandb/latest-run

wandb sync /lustre/home/u10010021/phisat2/runs/pixel_regression/roads/phisatnet/roads_train_50/seed_42/wandb/latest-run
wandb sync /lustre/home/u10010021/phisat2/runs/pixel_regression/roads/phisatnet/roads_train_500/seed_42/wandb/latest-run

wandb sync /lustre/home/u10010021/phisat2/runs/pixel_regression/building/phisatnet/building_train_50/seed_42/wandb/latest-run
wandb sync /lustre/home/u10010021/phisat2/runs/pixel_regression/building/phisatnet/building_train_500/seed_42/wandb/latest-run

echo "Done."