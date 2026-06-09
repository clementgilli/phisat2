EXPERIMENT ?=
ifneq ($(EXPERIMENT),)
    include $(EXPERIMENT)
endif

UV ?= uv
PYTHON ?= $(UV) run --python 3.13 python

TASK ?= segmentation
DATASET ?= lulc
MODEL ?= phisatnet
DATALOADER ?= zarr_downstream
SEED ?= 42
SEEDS ?= $(SEED)
EPOCHS ?= 50
BATCH_SIZE ?= 16
CROP_SIZE ?= 224
LR ?= 0.0001
NUM_WORKERS ?= 4
ROOT_DIR ?= .
OUTPUT_DIR ?= runs
PRETRAINED ?= true
ACCELERATOR ?= auto
DEVICES ?= auto
STRATEGY ?= auto
PRECISION ?= 32-true
AUTO_DDP ?= true
SUBSET_CSV ?=
RESUME ?= false
CKPT_FLAG ?=
WEIGHTS ?=
EXTRA_ARGS ?=

ifeq ($(PRETRAINED),true)
PRETRAINED_FLAG := --pretrained
else
PRETRAINED_FLAG := --no-pretrained
endif

ifeq ($(AUTO_DDP),true)
AUTO_DDP_FLAG := --auto-ddp
else
AUTO_DDP_FLAG :=
endif

ifneq ($(SUBSET_CSV),)
    SUBSET_FLAG = --subset-csv $(SUBSET_CSV)
else
    SUBSET_FLAG =
endif

ifneq ($(RESUME),false)
    RESUME_FLAG = --resume
else
    RESUME_FLAG =
endif

ifneq ($(WEIGHTS),)
    WEIGHTS_FLAG = --weights $(WEIGHTS)
else
    WEIGHTS_FLAG =
endif

ifneq ($(DATASET),)
    DATASET_FLAG = --dataset $(DATASET)
else
    DATASET_FLAG =
endif

ifneq ($(DATALOADER),)
    DATALOADER_FLAG = --dataloader $(DATALOADER)
else
    DATALOADER_FLAG =
endif

ifneq ($(strip $(CKPT_PATH)),)
    CKPT_FLAG = --ckpt-path $(CKPT_PATH)
else
    CKPT_FLAG =
endif

.DEFAULT_GOAL := help

.PHONY: help install sync mount check test smoke fast-dev-run train pretrain distillation train-segmentation train-classification train-regression eval sweep-seeds list-models list-dataloaders clean submit-train submit-eval _submit

help: ## Show available targets.
	@awk 'BEGIN {FS = ":.*##"; printf "Available targets:\n"} /^[a-zA-Z0-9_.-]+:.*##/ {printf "  %-22s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

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
		--model phisatnet \
		--dataloader synthetic \
		--seeds 0 \
		--root-dir . \
		--output-dir runs/smoke \
		--max-epochs 1 \
		--batch-size 2 \
		--crop-size 224 \
		--lr 0.0001 \
		--num-workers 0 \
		--accelerator cpu \
		--devices 1 \
		--precision 32-true \
		--no-pretrained \
		--fast-dev-run

fast-dev-run: ## Run a one-batch Lightning fast-dev run with the configured real dataloader.
	$(PYTHON) -m phisat2.cli.train fit \
		--task $(TASK) \
		$(DATASET_FLAG) \
		--model $(MODEL) \
		$(DATALOADER_FLAG) \
		--seeds $(SEEDS) \
		--root-dir $(ROOT_DIR) \
		--output-dir $(OUTPUT_DIR) \
		--max-epochs $(EPOCHS) \
		--batch-size $(BATCH_SIZE) \
		--crop-size $(CROP_SIZE) \
		--lr $(LR) \
		--num-workers $(NUM_WORKERS) \
		--accelerator $(ACCELERATOR) \
		--devices $(DEVICES) \
		--strategy $(STRATEGY) \
		--precision $(PRECISION) \
		$(AUTO_DDP_FLAG) \
		$(PRETRAINED_FLAG) \
		$(SUBSET_FLAG) \
		$(WEIGHTS_FLAG) \
		--fast-dev-run

train: ## Train with Make variables: TASK DATASET MODEL DATALOADER SEEDS EPOCHS WEIGHTS etc.
	$(PYTHON) -m phisat2.cli.train fit \
		--task $(TASK) \
		$(DATASET_FLAG) \
		--model $(MODEL) \
		$(DATALOADER_FLAG) \
		--seeds $(SEEDS) \
		--root-dir $(ROOT_DIR) \
		--output-dir $(OUTPUT_DIR) \
		--max-epochs $(EPOCHS) \
		--batch-size $(BATCH_SIZE) \
		--crop-size $(CROP_SIZE) \
		--lr $(LR) \
		--num-workers $(NUM_WORKERS) \
		--accelerator $(ACCELERATOR) \
		--devices $(DEVICES) \
		--strategy $(STRATEGY) \
		--precision $(PRECISION) \
		$(AUTO_DDP_FLAG) \
		$(PRETRAINED_FLAG) \
		$(SUBSET_FLAG) \
		$(RESUME_FLAG) \
		$(WEIGHTS_FLAG) \
		$(EXTRA_ARGS)


pretrain: ## Train the SSL Onboard CNN Baseline (no dataset required).
	$(MAKE) train TASK=pretrain_reconstruction MODEL=phisatnet DATASET= DATALOADER=

distillation: ## Train the KD pipeline with a Teacher Model (e.g. MODEL=terramind_v1_tiny).
	$(MAKE) train TASK=distillation_kd DATASET= DATALOADER=

train-segmentation: ## Train a segmentation model.
	$(MAKE) train TASK=segmentation

train-classification: ## Train a classification model.
	$(MAKE) train TASK=classification

train-regression: ## Train a pixel regression model.
	$(MAKE) train TASK=pixel_regression

eval:
	$(PYTHON) -m phisat2.cli.eval test \
		--task $(TASK) \
		$(DATASET_FLAG) \
		--model $(MODEL) \
		$(DATALOADER_FLAG) \
		$(CKPT_FLAG) \
		--root-dir $(ROOT_DIR) \
		--batch-size $(BATCH_SIZE) \
		--num-workers $(NUM_WORKERS) \
		--accelerator $(ACCELERATOR) \
		--devices $(DEVICES) \
		--strategy $(STRATEGY) \
		--precision $(PRECISION) \
		$(AUTO_DDP_FLAG) \
		$(EXTRA_ARGS)

sweep-seeds: ## Alias for train with SEEDS set to multiple values.
	$(MAKE) train

list-models: ## List registered model names.
	$(PYTHON) -m phisat2.cli.train list-models

list-dataloaders: ## List registered dataloader names.
	$(PYTHON) -m phisat2.cli.train list-dataloaders

clean: ## Remove common generated Python build and cache artifacts.
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov
	$(PYTHON) -c "from pathlib import Path; import shutil; [shutil.rmtree(p) for p in Path('.').rglob('__pycache__')]"

submit-train:
	@$(MAKE) _submit TARGET=train

submit-eval: 
	@$(MAKE) _submit TARGET=eval

submit-pretrain:
	@$(MAKE) _submit TARGET=pretrain

submit-distillation:
	@$(MAKE) _submit TARGET=distillation

_submit:
	@if [ -z "$(EXPERIMENT)" ]; then echo "Error: Specify a config with EXPERIMENT=configs/..."; exit 1; fi
	@sed -e "s|__JOB_NAME__|$(JOB_NAME)|g" \
		 -e "s|__QUEUE__|$(QUEUE)|g" \
		 -e "s|__WALLTIME__|$(WALLTIME)|g" \
		 -e "s|__GPUS__|$(GPUS)|g" \
		 -e "s|__CPUS__|$(CPUS)|g" \
		 -e "s|__MEM__|$(MEM)|g" \
		 -e "s|__EXPERIMENT__|$(EXPERIMENT)|g" \
		 -e "s|__MAKE_TARGET__|$(TARGET)|g" \
		 -e "s|__CKPT_PATH__|$(CKPT_PATH)|g" \
		 scripts/runner_template.pbs > .temp_job.pbs
	@qsub .temp_job.pbs
	@rm .temp_job.pbs
	@echo "Job $(TARGET) submitted to the cluster !"