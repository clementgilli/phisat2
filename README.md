# PhiSat-2

Train machine learning models for satellite imagery processing with support for multiple training paradigms (supervised, self-supervised learning, knowledge distillation, domain adaptation).

## Quick Start

### Requirements
- Python 3.13
- `uv` (package manager)
- `make`

### Installation
```bash
git clone https://github.com/clementgilli/phisat2
cd phisat2
make install
make check
make smoke
```

View documentation locally:
```bash
python -m webbrowser docs/index.html
```

---

## How to Run

List available options:
```bash
make list-models
make list-dataloaders
```

Run a quick smoke test (synthetic data, CPU):
```bash
make smoke
```

Run a one-batch test with a real dataloader:
```bash
make fast-dev-run TASK=segmentation DATASET=lulc DATALOADER=downstream
```

---

## Training

The Makefile is the primary training interface. Experiments are configured with Make variables:

```bash
# Self-supervised pre-training
make train TASK=pretrain_reconstruction DATALOADER=triplets EPOCHS=100 SEEDS="0 42"

# Knowledge distillation
make train TASK=distillation_kd DATALOADER=triplets EPOCHS=50 SEEDS="0 42"

# Domain adaptation
make train TASK=domain_adaptation DATALOADER=triplets EPOCHS=50 SEEDS="0 42"

# Downstream task (linear probing)
make train TASK=segmentation DATASET=lulc DATALOADER=downstream EPOCHS=10 SEEDS="0 42"
```

### Common Make Variables

```bash
TASK=segmentation|classification|pixel_regression|global_regression|pretrain_reconstruction|distillation_kd|domain_adaptation|eval_domain_gap
DATASET=lulc|lc|burned|floods|roads|building|fire
MODEL=<from make list-models>
DATALOADER=downstream|triplets|synthetic
SEEDS=0 or SEEDS="0 42 100"
EPOCHS=50
BATCH_SIZE=16
CROP_SIZE=224
LR=0.0001
NUM_WORKERS=4
ROOT_DIR=data/PhiSatNet
OUTPUT_DIR=runs
PRETRAINED=true
ACCELERATOR=auto
DEVICES=auto
AUTO_DDP=true
PRECISION=32-true
```

### Hardware Configuration

With default `AUTO_DDP=true`:
- Multiple CUDA GPUs: Lightning DDP
- Single GPU: uses that GPU
- CPU: Lightning auto behavior

---

## Code Structure

```
phisat2/
├── cli/                    # Command-line interface
├── models/                 # Model registry, encoders, decoders
├── data_loaders/           # Dataloader implementations
├── tasks/                  # Task specifications
├── training/               # Training modules (SSL, KD, DA)
├── evaluation/             # Evaluation utilities
└── utils/                  # Seeds, weights, visualization

tests/                      # Unit tests for registry and shapes
docs/                       # Documentation
src/                        # Legacy scripts (Phase 0)
```

---

## Model Architecture

**Shared Decoder Models** (most encoders):
- Encoders: `phisat2_geoaware`, `terramind_*`, `prithvi_*`, `dofa_*`, `seco_*`, `ssl4eos12_*`, `satlas_*`
- All wrap with `SharedDecoderModel` and use `FeaturePyramidAdapter`
- For spatial tasks: `SharedUNetDecoder`
- For global tasks: `GlobalPoolingHead`

**Full-Structure Models**:
- `myriad2_full_unet`: Complete U-Net with embedded decoder (segmentation/pixel_regression only)

---

## Validation

```bash
make check              # Verify install
make test               # Run unit tests
make smoke              # Quick smoke test
```

---

## Documentation

Full details available in `docs/index.html`:
- Quickstart, model architecture, training pipeline
- Data loaders, hyperparameters, CLI commands
- Troubleshooting, reproducibility, inference
