
UV ?= uv
PYTHON ?= $(UV) run --python 3.13 python

TASK ?= segmentation
DATASET ?= lulc
MODEL ?= phisat2_geoaware
DATALOADER ?= zarr_downstream
SEED ?= 42
SEEDS ?= $(SEED)
EPOCHS ?= 50
BATCH_SIZE ?= 16
LR ?= 0.0001
NUM_WORKERS ?= 4
ROOT_DIR ?= .
OUTPUT_DIR ?= runs
PRETRAINED ?= true
ACCELERATOR ?= auto
DEVICES ?= auto
PRECISION ?= 32-true

ifeq ($(PRETRAINED),true)
PRETRAINED_FLAG := --pretrained
else
PRETRAINED_FLAG := --no-pretrained
endif

.DEFAULT_GOAL := help

.PHONY: help install sync mount check test smoke train train-segmentation train-classification train-regression sweep-seeds list-models list-dataloaders clean

help: ## Show available targets.
	@awk 'BEGIN {FS = ":.*##"; printf "Available targets:\n"} /^[a-zA-Z0-9_.-]+:.*##/ {printf "  %-12s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install the package and dependencies with uv.
	$(UV) sync --python 3.13 --group dev

sync: ## Sync the uv-managed environment from pyproject.toml and uv.lock.
	$(UV) sync --python 3.13 --group dev

mount: ## Mount the PhiSatNet Hugging Face bucket locally.
	$(UV) run scripts/mount_phisatnet_bucket.sh

check: ## Run a lightweight import/bytecode check.
	$(PYTHON) -m compileall phisat2

test: ## Run unit tests.
	$(UV) run --python 3.13 pytest

smoke: ## Run a one-batch synthetic Lightning smoke test.
	$(PYTHON) -m phisat2.cli.train fit \
		--task segmentation \
		--dataset clouds \
		--model phisat2_geoaware \
		--dataloader synthetic \
		--seeds 0 \
		--root-dir . \
		--output-dir runs/smoke \
		--max-epochs 1 \
		--batch-size 2 \
		--lr 0.0001 \
		--num-workers 0 \
		--accelerator cpu \
		--devices 1 \
		--precision 32-true \
		--no-pretrained \
		--fast-dev-run

train: ## Train with Make variables: TASK DATASET MODEL DATALOADER SEEDS EPOCHS etc.
	$(PYTHON) -m phisat2.cli.train fit \
		--task $(TASK) \
		--dataset $(DATASET) \
		--model $(MODEL) \
		--dataloader $(DATALOADER) \
		--seeds $(SEEDS) \
		--root-dir $(ROOT_DIR) \
		--output-dir $(OUTPUT_DIR) \
		--max-epochs $(EPOCHS) \
		--batch-size $(BATCH_SIZE) \
		--lr $(LR) \
		--num-workers $(NUM_WORKERS) \
		--accelerator $(ACCELERATOR) \
		--devices $(DEVICES) \
		--precision $(PRECISION) \
		$(PRETRAINED_FLAG)

train-segmentation: ## Train a segmentation model.
	$(MAKE) train TASK=segmentation

train-classification: ## Train a classification model.
	$(MAKE) train TASK=classification

train-regression: ## Train a pixel regression model.
	$(MAKE) train TASK=pixel_regression

sweep-seeds: ## Alias for train with SEEDS set to multiple values.
	$(MAKE) train

list-models: ## List registered model names.
	$(PYTHON) -m phisat2.cli.train list-models

list-dataloaders: ## List registered dataloader names.
	$(PYTHON) -m phisat2.cli.train list-dataloaders

clean: ## Remove common generated Python build and cache artifacts.
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov
	$(PYTHON) -c "from pathlib import Path; import shutil; [shutil.rmtree(p) for p in Path('.').rglob('__pycache__')]"
