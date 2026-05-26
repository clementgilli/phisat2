# PhiSat-2

## Quickstart

Requirements:
- Python 3.13
- `uv`
- `make`

```bash
git clone https://github.com/clementgilli/phisat2
cd phisat2
make install
```

Open the static docs at [docs/index.html](docs/index.html), or from a terminal:

```bash
python -m webbrowser docs/index.html
```

## How to run

Check the install and list the available runtime options:

```bash
make check
make list-models
make list-dataloaders
```

Run the CPU smoke test with synthetic data:

```bash
make smoke
```

Run a one-batch fast-dev pass with the configured real dataloader:

```bash
make fast-dev-run DATASET=lc ROOT_DIR=data/PhiSatNet NUM_WORKERS=0 PRETRAINED=false CROP_SIZE=128
```

## Training

The Makefile is the public training interface. Experiments are configured with
Make variables, not YAML files.

```bash
make fast-dev-run DATASET=lc ROOT_DIR=data/PhiSatNet NUM_WORKERS=0 PRETRAINED=false CROP_SIZE=128
make train TASK=segmentation DATASET=lulc MODEL=terramind_v1_tiny DATALOADER=zarr_downstream SEEDS="13 42 100" EPOCHS=50
make train TASK=segmentation DATASET=marine MODEL=terramind_v1_tiny DATALOADER=zarr_downstream SEED=42 EPOCHS=30
make train TASK=segmentation DATASET=clouds MODEL=phisat2_geoaware DATALOADER=zarr_downstream SEED=7 BATCH_SIZE=16
make list-models
make list-dataloaders
make smoke
```

Useful variables:

```bash
TASK=segmentation|classification|pixel_regression|global_regression
DATASET=lc|lulc|marine|burned_area|clouds|worldfloods|fire
MODEL=phisat2_geoaware|terramind_v1_tiny|prithvi_eo_v1_100|myriad2_full_unet
DATALOADER=zarr_downstream|h5_pairs|synthetic
SEED=42
SEEDS="13 42 100"
EPOCHS=50
BATCH_SIZE=16
CROP_SIZE=224
LR=0.0001
NUM_WORKERS=4
ROOT_DIR=/path/to/data
OUTPUT_DIR=runs
PRETRAINED=true
ACCELERATOR=auto
DEVICES=auto
STRATEGY=auto
AUTO_DDP=true
PRECISION=32-true
```

With the default `AUTO_DDP=true`, `make train` resolves automatic hardware to
all visible CUDA GPUs. Multi-GPU CUDA runs use Lightning DDP; single-GPU CUDA
runs use that GPU; CPU-only machines keep Lightning's `auto` hardware behavior.

Files under `pretrain/weights` are reference artifacts and are not used to
initialize training. The `myriad2_full_unet` model preserves the inspected
full-U-Net topology as a benchmark exception, but it still initializes normally.

### Decoder behavior

- Shared encoder families (`phisat2_geoaware`, `terramind_*`, `prithvi_*`) use a
  `SharedDecoderModel` wrapper.
- For spatial tasks (`segmentation`, `pixel_regression`), that wrapper always
  uses the same `SharedUNetDecoder` class.
- Raw encoder outputs are first normalized by a `FeaturePyramidAdapter` into a 4-
  level spatial pyramid, so different encoders can still connect to the shared
  decoder contract.
- For non-spatial tasks (`classification`, `global_regression`) the shared wrapper
  uses a `GlobalPoolingHead` instead.
- `myriad2_full_unet` does not use shared decoding; its decoder is embedded in
  the full U-Net model itself.

## Validation

```bash
make check
make test
```
