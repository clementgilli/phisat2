#!/bin/bash
TEACHERS=("ssl4eos12_vit_small_patch16_224_sentinel2_all_dino" "ssl4eos12_vit_small_patch16_224_sentinel2_all_moco" "dofa_large_patch16_224" "prithvi_eo_v2_300")
#TEACHERS=("terramind_v1_large")

#TEACHERS=("ssl4eos12_resnet50_sentinel2_all_dino" "satlas_resnet50_sentinel2_si_ms_satlas" "ssl4eos12_resnet50_sentinel2_all_decur" "ssl4eos12_resnet50_sentinel2_all_moco" "ssl4eos12_resnet50_sentinel2_all_softcon" "seco_resnet18_sentinel2_rgb_seco" "seco_resnet50_sentinel2_rgb_seco" "ssl4eos12_resnet18_sentinel2_all_moco")

for TEACHER in "${TEACHERS[@]}"; do
    echo "Running evaluation for ${TEACHER}..."
    #make submit-eval EXPERIMENT=configs/roads/roads_kd_full.mk MODEL=${TEACHER}
    #make submit-eval EXPERIMENT=configs/building/building_kd_full.mk MODEL=${TEACHER}
    #make submit-eval EXPERIMENT=configs/clouds/clouds_kd_full.mk MODEL=${TEACHER}
    make submit-eval EXPERIMENT=configs/floods/floods_kd_full.mk MODEL=${TEACHER}
    #make submit-eval EXPERIMENT=configs/lulc/lulc_kd_full.mk MODEL=${TEACHER}
    #make submit-eval EXPERIMENT=configs/burned_area/burned_area_kd_full.mk MODEL=${TEACHER}
    #make submit-eval EXPERIMENT=configs/router/router_kd.mk MODEL=${TEACHER}
    #make submit-eval EXPERIMENT=configs/knowledge_distillation/kd.mk MODEL=${TEACHER} CKPT_PATH=/lustre/home/u10010021/phisat2/runs/knowledge_distillation/triplets/${TEACHER}/full_dataset/seed_42/checkpoints/best.ckpt
done