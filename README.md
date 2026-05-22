# Phisat-2

## Quickstart
```
git clone https://github.com/clementgilli/phisat2
cd phisat2
make install
```

## Training

The Makefile is the public training interface. Experiments are configured with
Make variables, not YAML files.

```bash
make train TASK=segmentation DATASET=lulc MODEL=terramind_v1_tiny DATALOADER=zarr_downstream SEEDS="13 42 100" EPOCHS=50
make train TASK=classification DATASET=climate MODEL=prithvi_eo_v1_100 DATALOADER=h5_pairs SEED=42 EPOCHS=30
make train TASK=pixel_regression DATASET=reconstruction MODEL=phisat2_geoaware DATALOADER=h5_pairs SEED=7 BATCH_SIZE=16
make list-models
make list-dataloaders
make smoke
```

Useful variables:

```bash
TASK=segmentation|classification|pixel_regression|global_regression
DATASET=lulc|clouds|burned|floods|marine|climate|geoloc|reconstruction
MODEL=phisat2_geoaware|terramind_v1_tiny|prithvi_eo_v1_100|myriad2_full_unet
DATALOADER=zarr_downstream|h5_pairs|synthetic
SEED=42
SEEDS="13 42 100"
EPOCHS=50
BATCH_SIZE=16
LR=0.0001
NUM_WORKERS=4
ROOT_DIR=/path/to/data
OUTPUT_DIR=runs
PRETRAINED=true
ACCELERATOR=auto
DEVICES=auto
PRECISION=32-true
```

Files under `pretrain/weights` are reference artifacts and are not used to
initialize training. The `myriad2_full_unet` model preserves the inspected
full-U-Net topology as a benchmark exception, but it still initializes normally.
