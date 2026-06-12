EXPERIMENT ?=
ifneq ($(EXPERIMENT),)
    include $(EXPERIMENT)
endif

UV     ?= uv
PYTHON ?= $(UV) run --python 3.13 python
CLI    := $(PYTHON) -m phisat2.cli.cli

# ── Task / Model / Data ───────────────────────────────────────────────────────
TASK       ?= segmentation
DATASET    ?= lulc
MODEL      ?= phisatnet
DATALOADER ?= zarr_downstream

# ── Training hyperparameters ──────────────────────────────────────────────────
SEED       ?= 42
SEEDS      ?= $(SEED)
EPOCHS     ?= 50
BATCH_SIZE ?= 16
CROP_SIZE  ?= 224
LR         ?= 0.0001
PATIENCE   ?=

# ── Infrastructure ────────────────────────────────────────────────────────────
NUM_WORKERS ?= 4
ROOT_DIR    ?= .
OUTPUT_DIR  ?= runs
PRETRAINED  ?= true
ACCELERATOR ?= auto
DEVICES     ?= auto
STRATEGY    ?= auto
PRECISION   ?= 32-true
AUTO_DDP    ?= true

# ── Checkpoints and weights ───────────────────────────────────────────────────
WEIGHTS      ?=   # Encoder .pth — initialises backbone (SSL / KD / DA)
CKPT_PATH    ?=   # Full Lightning .ckpt — restores module for evaluation
TEACHER_CKPT ?=   # Teacher encoder .pth  (eval-domain-gap only)
STUDENT_CKPT ?=   # Student encoder .pth  (eval-domain-gap only)
DECODERS     ?=   # "dataset=path ..."    (eval-domain-gap only)

# ── Misc ──────────────────────────────────────────────────────────────────────
SUBSET_CSV ?=
RESUME     ?= false
EXTRA_ARGS ?=

# ─────────────────────────────────────────────────────────────────────────────
# Derived flags  (set once, reused across targets)
# ─────────────────────────────────────────────────────────────────────────────

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

ifneq ($(RESUME),false)
RESUME_FLAG = --resume
else
RESUME_FLAG =
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

ifneq ($(SUBSET_CSV),)
SUBSET_FLAG = --subset-csv $(SUBSET_CSV)
else
SUBSET_FLAG =
endif

ifneq ($(WEIGHTS),)
WEIGHTS_FLAG = --weights $(WEIGHTS)
else
WEIGHTS_FLAG =
endif

# CKPT_PATH (full Lightning .ckpt) also maps to --weights at the CLI level
ifneq ($(strip $(CKPT_PATH)),)
CKPT_FLAG = --weights $(CKPT_PATH)
else
CKPT_FLAG =
endif

ifneq ($(PATIENCE),)
PATIENCE_FLAG = --patience $(PATIENCE)
else
PATIENCE_FLAG =
endif

ifneq ($(TEACHER_CKPT),)
TEACHER_CKPT_FLAG = --teacher-ckpt $(TEACHER_CKPT)
else
TEACHER_CKPT_FLAG =
endif

ifneq ($(STUDENT_CKPT),)
STUDENT_CKPT_FLAG = --student-ckpt $(STUDENT_CKPT)
else
STUDENT_CKPT_FLAG =
endif

ifneq ($(DECODERS),)
DECODERS_FLAG = --decoders $(DECODERS)
else
DECODERS_FLAG =
endif

# ─────────────────────────────────────────────────────────────────────────────
.DEFAULT_GOAL := help
.PHONY: help install sync mount check test smoke fast-dev-run \
        train pretrain distillation domain-adaptation \
        train-segmentation train-classification train-regression \
        eval eval-domain-gap sweep-seeds \
        list-models list-dataloaders clean \
        submit-train submit-eval submit-pretrain \
        submit-distillation submit-domain-adaptation submit-eval-domain-gap \
        _submit
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

help: ## Show available targets.
	@awk 'BEGIN {FS = ":.*##"; printf "Available targets:\n"} \
	      /^[a-zA-Z0-9_.-]+:.*##/ {printf "  %-30s %s\n", $$1, $$2}' \
	      $(MAKEFILE_LIST)

install: ## Install package and dev dependencies with uv.
	$(UV) sync --python 3.13 --group dev

sync: ## Sync the uv environment from pyproject.toml and uv.lock.
	$(UV) sync --python 3.13 --group dev

mount: ## Mount the PhiSat-2 Hugging Face bucket locally.
	$(UV) run scripts/mount_phisatnet_bucket.sh

check: ## Bytecode check on the phisat2 package.
	$(PYTHON) -m compileall phisat2

test: ## Run the unit test suite.
	$(UV) run --python 3.13 pytest

list-models: ## List registered model names and roles.
	$(CLI) list-models

list-dataloaders: ## List registered dataloader names.
	$(CLI) list-dataloaders

clean: ## Remove build and cache artifacts.
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov
	$(PYTHON) -c "from pathlib import Path; import shutil; \
	              [shutil.rmtree(p) for p in Path('.').rglob('__pycache__')]"


# ─────────────────────────────────────────────────────────────────────────────
# Sanity checks
# ─────────────────────────────────────────────────────────────────────────────

smoke: ## Synthetic one-batch smoke test — CPU only, no dataset required.
	$(CLI) fit \
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

fast-dev-run: ## One-batch fast-dev-run using the configured real dataloader.
	$(CLI) fit \
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
		$(WEIGHTS_FLAG) \
		--fast-dev-run


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

train: ## Generic fit. Override with TASK= MODEL= DATASET= DATALOADER= SEEDS= etc.
	$(CLI) fit \
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
		$(PATIENCE_FLAG) \
		$(EXTRA_ARGS)

pretrain: ## Phase 1 — SSL pretraining on simulated PhiSat-2 data.
	$(MAKE) train TASK=pretrain_reconstruction MODEL=phisatnet DATASET= DATALOADER=

distillation: ## Phase 2 — Knowledge distillation (MODEL=terramind_v1_*  WEIGHTS=<ssl.ckpt>).
	$(MAKE) train TASK=distillation_kd DATASET= DATALOADER=

domain-adaptation: ## Phase 4 — Sim-to-real domain adaptation (WEIGHTS=<pretrained_sim.pth>).
	$(MAKE) train TASK=domain_adaptation DATASET= DATALOADER=

train-segmentation: ## Downstream segmentation (set DATASET= DATALOADER=).
	$(MAKE) train TASK=segmentation

train-classification: ## Downstream classification (set DATASET= DATALOADER=).
	$(MAKE) train TASK=classification

train-regression: ## Downstream pixel regression (set DATASET= DATALOADER=).
	$(MAKE) train TASK=pixel_regression

sweep-seeds: ## Run training across multiple seeds (set SEEDS="42 43 44").
	$(MAKE) train


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

eval: ## Generic evaluation. Pass CKPT_PATH=<lightning.ckpt> and TASK= DATASET= etc.
	$(CLI) test \
		--task $(TASK) \
		$(DATASET_FLAG) \
		--model $(MODEL) \
		$(DATALOADER_FLAG) \
		--root-dir $(ROOT_DIR) \
		--output-dir $(OUTPUT_DIR) \
		--batch-size $(BATCH_SIZE) \
		--crop-size $(CROP_SIZE) \
		--num-workers $(NUM_WORKERS) \
		--accelerator $(ACCELERATOR) \
		--devices $(DEVICES) \
		--strategy $(STRATEGY) \
		--precision $(PRECISION) \
		$(AUTO_DDP_FLAG) \
		$(CKPT_FLAG) \
		$(EXTRA_ARGS)

eval-domain-gap: ## Domain gap evaluation (TEACHER_CKPT= STUDENT_CKPT= DECODERS=).
	$(CLI) test \
		--task eval_domain_gap \
		--model phisatnet \
		--root-dir $(ROOT_DIR) \
		--output-dir $(OUTPUT_DIR) \
		--batch-size $(BATCH_SIZE) \
		--num-workers $(NUM_WORKERS) \
		--accelerator $(ACCELERATOR) \
		--devices $(DEVICES) \
		--strategy $(STRATEGY) \
		--precision $(PRECISION) \
		$(AUTO_DDP_FLAG) \
		$(TEACHER_CKPT_FLAG) \
		$(STUDENT_CKPT_FLAG) \
		$(DECODERS_FLAG) \
		$(EXTRA_ARGS)


# ─────────────────────────────────────────────────────────────────────────────
# Cluster submission  (requires EXPERIMENT=configs/my_run.mk)
# ─────────────────────────────────────────────────────────────────────────────

submit-train:
	@$(MAKE) _submit TARGET=train

submit-eval:
	@$(MAKE) _submit TARGET=eval

submit-pretrain:
	@$(MAKE) _submit TARGET=pretrain

submit-distillation:
	@$(MAKE) _submit TARGET=distillation

submit-domain-adaptation:
	@$(MAKE) _submit TARGET=domain-adaptation

submit-eval-domain-gap:
	@$(MAKE) _submit TARGET=eval-domain-gap

_submit:
	@if [ -z "$(EXPERIMENT)" ]; then \
	    echo "Error: specify a config with EXPERIMENT=configs/my_run.mk"; exit 1; \
	fi
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
	@echo "Job '$(TARGET)' submitted!"