#!/bin/bash

set -e

source .venv/bin/activate

#make submit-eval EXPERIMENT=configs/lulc/lulc_50_shot.mk CKPT_PATH=/lustre/home/u10010021/phisat2/runs/segmentation/lulc/phisatnet/lulc_train_50_global/seed_42/checkpoints/best.ckpt
#make submit-eval EXPERIMENT=configs/floods/floods_50_shot.mk CKPT_PATH=/lustre/home/u10010021/phisat2/runs/segmentation/floods/phisatnet/floods_train_50_global/seed_42/checkpoints/best.ckpt
#make submit-eval EXPERIMENT=configs/burned_area/burned_area_50_shot.mk CKPT_PATH=/lustre/home/u10010021/phisat2/runs/segmentation/burned/phisatnet/burned_train_50_global/seed_42/checkpoints/best.ckpt
#make submit-eval EXPERIMENT=configs/roads/roads_50_shot.mk CKPT_PATH=/lustre/home/u10010021/phisat2/runs/pixel_regression/roads/phisatnet/roads_train_50/seed_42/checkpoints/best.ckpt
#make submit-eval EXPERIMENT=configs/building/building_50_shot.mk CKPT_PATH=/lustre/home/u10010021/phisat2/runs/pixel_regression/building/phisatnet/building_train_50/seed_42/checkpoints/best.ckpt
#make submit-eval EXPERIMENT=configs/clouds/clouds_50_shot.mk CKPT_PATH=/lustre/home/u10010021/phisat2/runs/segmentation/clouds/phisatnet/clouds_50shot/seed_42/checkpoints/best.ckpt

#make submit-eval EXPERIMENT=configs/lulc/lulc_500_shot.mk CKPT_PATH=/lustre/home/u10010021/phisat2/runs/segmentation/lulc/phisatnet/lulc_train_500_global/seed_42/checkpoints/best.ckpt
#make submit-eval EXPERIMENT=configs/floods/floods_500_shot.mk CKPT_PATH=/lustre/home/u10010021/phisat2/runs/segmentation/floods/phisatnet/floods_train_500_global/seed_42/checkpoints/best.ckpt
#make submit-eval EXPERIMENT=configs/burned_area/burned_area_500_shot.mk CKPT_PATH=/lustre/home/u10010021/phisat2/runs/segmentation/burned/phisatnet/burned_train_500_global/seed_42/checkpoints/best.ckpt
#make submit-eval EXPERIMENT=configs/roads/roads_500_shot.mk CKPT_PATH=/lustre/home/u10010021/phisat2/runs/pixel_regression/roads/phisatnet/roads_train_500/seed_42/checkpoints/best.ckpt
#make submit-eval EXPERIMENT=configs/building/building_500_shot.mk CKPT_PATH=/lustre/home/u10010021/phisat2/runs/pixel_regression/building/phisatnet/building_train_500/seed_42/checkpoints/best.ckpt
#make submit-eval EXPERIMENT=configs/clouds/clouds_500_shot.mk CKPT_PATH=/lustre/home/u10010021/phisat2/runs/segmentation/clouds/phisatnet/clouds_500shot/seed_42/checkpoints/best.ckpt
#make submit-eval EXPERIMENT=configs/router/router.mk CKPT_PATH=/lustre/home/u10010021/phisat2/runs/classification/router/phisatnet/full_dataset/seed_42/checkpoints/best.ckpt


#make submit-eval-domain-gap EXPERIMENT=configs/domain_adaptation.mk CKPT_PATH=/lustre/home/u10010021/phisat2/runs/domain_adaptation/triplets/phisatnet/full_dataset/seed_42/checkpoints/best.ckpt

#wandb sync /lustre/home/u10010021/phisat2/runs/segmentation/lulc/phisatnet/full_dataset/seed_42/wandb/latest-run
#wandb sync /lustre/home/u10010021/phisat2/runs/segmentation/floods/phisatnet/full_dataset/seed_42/wandb/latest-run
#wandb sync /lustre/home/u10010021/phisat2/runs/segmentation/burned/phisatnet/full_dataset/seed_42/wandb/latest-run
#wandb sync /lustre/home/u10010021/phisat2/runs/pixel_regression/roads/phisatnet/full_dataset/seed_42/wandb/latest-run
#wandb sync /lustre/home/u10010021/phisat2/runs/pixel_regression/building/phisatnet/full_dataset/seed_42/wandb/latest-run
#wandb sync /lustre/home/u10010021/phisat2/runs/domain_adaptation/triplets/phisatnet/full_dataset/seed_42/wandb/latest-run

wandb sync /lustre/home/u10010021/phisat2/runs/segmentation/lulc/ssl4eos12_resnet18_sentinel2_all_moco/full_dataset/seed_42/wandb/offline-run-20260630_145440-b1hpmcg5
wandb sync /lustre/home/u10010021/phisat2/runs/segmentation/lulc/ssl4eos12_resnet18_sentinel2_all_moco/full_dataset/seed_42/wandb/offline-run-20260630_150031-5fyzba0f
wandb sync /lustre/home/u10010021/phisat2/runs/segmentation/lulc/ssl4eos12_resnet18_sentinel2_all_moco/full_dataset/seed_42/wandb/offline-run-20260630_151513-x4gkwxtf
wandb sync /lustre/home/u10010021/phisat2/runs/segmentation/lulc/ssl4eos12_resnet18_sentinel2_all_moco/full_dataset/seed_42/wandb/offline-run-20260630_152415-6mhgebdr

echo "Done."