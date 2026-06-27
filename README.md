# PhiSat-2

Train PhiSat-2 satellite-imagery models with the current `phisat2` package,
Makefile workflow, and static docs in `docs/`.

The active package supports downstream supervised training, self-supervised
PhiSatNet pretraining, domain adaptation, evaluation, and registry utilities.
Knowledge distillation metadata is present, but the CLI currently raises
`NotImplementedError` for `knowledge_distillation`.

## Quick Start

### Requirements

- Python 3.13
- `uv`
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

## How to Run

List registered models and dataloaders:

```bash
make list-models
make list-dataloaders
```

Run a synthetic CPU smoke test with no local dataset:

```bash
make smoke
```

Run one synthetic development batch with explicit variables:

```bash
make fast-dev-run \
  TASK=segmentation \
  DATASET=clouds \
  MODEL=phisatnet \
  DATALOADER=synthetic \
  SEEDS=0 \
  ACCELERATOR=cpu \
  DEVICES=1 \
  NUM_WORKERS=0 \
  PRETRAINED=false
```

Run one real downstream batch:

```bash
make fast-dev-run \
  TASK=segmentation \
  DATASET=lulc \
  MODEL=phisatnet \
  DATALOADER=downstream \
  ROOT_DIR=/path/to/data
```

Direct CLI invocation uses the package module behind the Makefile:

```bash
uv run --python 3.13 python -m phisat2.cli.cli list-models
```

## Training

The Makefile is the primary interface. Experiments are configured with Make
variables or by passing `EXPERIMENT=configs/<name>.mk`.

```bash
# Self-supervised pretraining on triplets
make pretrain MODEL=phisatnet ROOT_DIR=/path/to/triplets SEEDS="42"

# Domain adaptation from a pretrained PhiSatNet encoder
make domain-adaptation \
  MODEL=phisatnet \
  ROOT_DIR=/path/to/triplets \
  WEIGHTS=/path/to/pretrained_sim.pth \
  SEEDS="42"

# Downstream segmentation
make train-segmentation \
  DATASET=lulc \
  MODEL=phisatnet \
  DATALOADER=downstream \
  ROOT_DIR=/path/to/data \
  WEIGHTS=/path/to/encoder.pth \
  SEEDS="42 7 6"

# Evaluation with a Lightning checkpoint
make eval \
  TASK=segmentation \
  DATASET=lulc \
  MODEL=phisatnet \
  DATALOADER=downstream \
  ROOT_DIR=/path/to/data \
  CKPT_PATH=/path/to/best.ckpt
```

## Common Make Variables

```bash
TASK=segmentation|classification|pixel_regression|global_regression|pretrain_reconstruction|knowledge_distillation|domain_adaptation|eval_domain_gap|eval_encoder
DATASET=lulc|lc|burned|clouds|floods|roads|building|fire
MODEL=phisatnet|terramind_v1_tiny|terramind_v1_small|terramind_v1_base|terramind_v1_large
DATALOADER=downstream|triplets|synthetic
SEEDS=42 or SEEDS="42 7 6"
EPOCHS=50
BATCH_SIZE=16
CROP_SIZE=224
LR=0.0001
NUM_WORKERS=4
ROOT_DIR=.
OUTPUT_DIR=runs
PRETRAINED=true
WEIGHTS=/path/to/encoder_or_checkpoint.pth
CKPT_PATH=/path/to/lightning.ckpt
SUBSET_CSV=/path/to/n_shot_subset.csv
ACCELERATOR=auto
DEVICES=auto
STRATEGY=auto
AUTO_DDP=true
PRECISION=32-true
```

Task-specific datasets are defined in `phisat2/tasks/specs.py`:

- `segmentation`: `lulc`, `lc`, `burned`, `clouds`, `floods`
- `pixel_regression`: `roads`, `building`
- `classification`: `fire`
- `pretrain_reconstruction`, `domain_adaptation`, `eval_domain_gap`: `triplets`
- `eval_encoder`: `lulc`
- `global_regression`: registered as a task type, but no datasets are registered yet

## Hardware Configuration

With default `AUTO_DDP=true`:

- Multiple CUDA GPUs: Lightning DDP
- Single CUDA GPU: single-GPU training
- CPU: Lightning auto behavior unless explicitly overridden

Override with `ACCELERATOR`, `DEVICES`, and `STRATEGY`.

## Code Structure

```text
phisat2/
  cli/                    # Make-backed CLI entrypoint
  models/                 # Model registry, encoders, decoder, heads
  data_loaders/           # Downstream, triplet, and synthetic dataloaders
  tasks/                  # Task and dataset specifications
  training/               # Lightning modules
  evaluation/             # Evaluation modules and metrics
  utils/                  # Seeds, weights, visualization

tests/                    # Unit tests
docs/                     # Static HTML documentation
configs/                  # Reusable Make experiment configs
src/                      # Legacy research and dataset scripts
```

## Model Architecture

Current registered models:

- `phisatnet`: student PhiSatNet encoder for downstream training, pretraining, and domain adaptation
- `terramind_v1_tiny`, `terramind_v1_small`, `terramind_v1_base`, `terramind_v1_large`: TerraTorch teacher backbones

`build_model()` returns a `ModelBundle`. For downstream and pretraining tasks,
it builds a `ComposedModel` from an encoder and one task head:

- Spatial tasks: `PhiSatNetDecoder`
- Classification and other non-spatial outputs: `MultiScaleClassificationHead`

## Validation

```bash
make check              # Byte-compile the phisat2 package
make test               # Run the pytest suite
make smoke              # Synthetic one-batch CLI run
```

## Documentation

Open `docs/index.html` for:

- Quickstart, CLI commands, and training pipeline
- Model architecture and task mapping
- Dataloader behavior and dataset layout
- Checkpoints, reproducibility, troubleshooting, and evaluation notes
