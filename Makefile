
PYTHON ?= python
UV ?= uv

.DEFAULT_GOAL := help

.PHONY: help install sync mount check clean

help: ## Show available targets.
	@awk 'BEGIN {FS = ":.*##"; printf "Available targets:\n"} /^[a-zA-Z0-9_.-]+:.*##/ {printf "  %-12s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install the package and dependencies with uv.
	$(UV) sync

sync: ## Sync the uv-managed environment from pyproject.toml and uv.lock.
	$(UV) sync

mount: ## Mount the PhiSatNet Hugging Face bucket locally.
	$(UV) run scripts/mount_phisatnet_bucket.sh

check: ## Run a lightweight import/bytecode check.
	$(PYTHON) -m compileall src

clean: ## Remove common generated Python build and cache artifacts.
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov
	$(PYTHON) -c "from pathlib import Path; import shutil; [shutil.rmtree(p) for p in Path('.').rglob('__pycache__')]"
