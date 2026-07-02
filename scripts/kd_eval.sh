#!/bin/bash
TEACHERS=("ssl4eos12_resnet50_sentinel2_all_dino" "satlas_resnet50_sentinel2_si_ms_satlas" "ssl4eos12_resnet50_sentinel2_all_decur" "ssl4eos12_resnet50_sentinel2_all_moco")

for TEACHER in "${TEACHERS[@]}"; do
    echo "Running evaluation for ${TEACHER}..."
    #make submit-eval EXPERIMENT=configs/roads/roads_kd_full.mk MODEL=${TEACHER}
    #make submit-eval EXPERIMENT=configs/building/building_kd_full.mk MODEL=${TEACHER}
    make submit-eval EXPERIMENT=configs/clouds/clouds_kd_full.mk MODEL=${TEACHER}
    #make submit-eval EXPERIMENT=configs/floods/floods_kd_full.mk MODEL=${TEACHER}
    #make submit-eval EXPERIMENT=configs/lulc/lulc_kd_full.mk MODEL=${TEACHER}
    #make submit-eval EXPERIMENT=configs/burned/burned_kd_full.mk MODEL=${TEACHER}
    #make submit-eval EXPERIMENT=configs/router/router_kd.mk MODEL=${TEACHER}
done