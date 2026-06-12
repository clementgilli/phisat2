from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from phisat2.data_loaders import build_datamodule, list_dataloaders
from phisat2.models.registry import build_model, list_models, ModelBundle
from phisat2.tasks import resolve_task_spec
from phisat2.tasks.specs import TASKS

from phisat2.training.pretrain_ssl import SSLPretrainModule
from phisat2.training.domain_adaptation import DomainAdaptationModule
from phisat2.training.downstream import DownstreamModule
from phisat2.evaluation.domain_eval import DomainEvalModule
from phisat2.utils.seed import seed_everything


# ─────────────────────────────────────────────────────────────────────────────
# Tasks that own their dataloader
# ─────────────────────────────────────────────────────────────────────────────

# Maps task name → dataloader name (no need for --dataset / --dataloader)
_PAIRED_TASKS: dict[str, str] = {
    "pretrain_reconstruction": "triplets",  # (sim)
    "distillation_kd":         "triplets",  # (sim, s2)
    "domain_adaptation":       "triplets",     # (real, sim)
    "eval_domain_gap":         "triplets",     # (real, sim)
}


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def _add_shared_args(p: argparse.ArgumentParser) -> None:
    """Arguments common to both `fit` and `test`."""
    p.add_argument("--task",       required=True, choices=list(TASKS))
    p.add_argument("--model",      required=True,
                   help="Model name from the registry (e.g. 'phisatnet', 'terramind_v1_base').")
    p.add_argument("--dataset",    type=str, default=None)
    p.add_argument("--dataloader", type=str, default=None)
    p.add_argument("--subset-csv", type=str, default=None,
                   help="Path to an N-shot CSV to filter the dataset.")
    p.add_argument("--root-dir",    default=".")
    p.add_argument("--output-dir",  default="runs")
    p.add_argument("--batch-size",  type=int,  default=16)
    p.add_argument("--crop-size",   type=int,  default=224)
    p.add_argument("--num-workers", type=int,  default=4)
    p.add_argument("--accelerator", default="auto")
    p.add_argument("--devices",     default="auto")
    p.add_argument("--strategy",    default="auto")
    p.add_argument("--precision",   default="32-true")
    p.add_argument("--auto-ddp",    action="store_true",
                   help="Auto-select DDP when multiple GPUs are available.")
    # Weight loading
    p.add_argument("--weights",      type=str, default=None,
                   help="Encoder .pth (SSL/DA/KD) or full Lightning .ckpt (downstream test).")
    p.add_argument("--teacher-ckpt", type=str, default=None,
                   help="Teacher encoder .pth (eval_domain_gap only).")
    p.add_argument("--student-ckpt", type=str, default=None,
                   help="Student encoder .pth (eval_domain_gap only). "
                        "Falls back to --teacher-ckpt if not provided.")
    p.add_argument("--decoders",     type=str, nargs="+", default=[],
                   help="Decoder checkpoints: 'dataset_name=path/to/ckpt' "
                        "(eval_domain_gap only). Example: lulc=weights/lulc.pth")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PhiSat-2 pipeline")
    sub    = parser.add_subparsers(dest="command", required=True)

    # ── fit ───────────────────────────────────────────────────────────────
    fit = sub.add_parser("fit", help="Train a model.")
    _add_shared_args(fit)
    fit.add_argument("--seeds",      nargs="+", type=int, required=True,
                     help="One or more random seeds. One training run per seed.")
    fit.add_argument("--max-epochs", type=int,   default=50)
    fit.add_argument("--lr",         type=float, default=1e-4)
    fit.add_argument("--patience",   type=int,   default=None,
                     help="Early stopping patience in epochs. Disabled if not provided.")
    fit.add_argument("--fast-dev-run", action="store_true",
                     help="Run a single batch for train + val (sanity check).")
    fit.add_argument("--resume", action="store_true",
                     help="Resume from the last checkpoint if it exists.")
    pret = fit.add_mutually_exclusive_group()
    pret.add_argument("--pretrained",    dest="pretrained", action="store_true",
                      help="Load pretrained backbone weights (teachers).")
    pret.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    fit.set_defaults(pretrained=True)

    # ── test ──────────────────────────────────────────────────────────────
    test = sub.add_parser("test", help="Evaluate a model on the test set.")
    _add_shared_args(test)
    test.add_argument("--seed", type=int, default=42)

    # ── utilities ─────────────────────────────────────────────────────────
    sub.add_parser("list-models",      help="List all registered model names.")
    sub.add_parser("list-dataloaders", help="List all registered dataloader names.")

    return parser


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_hardware(args: argparse.Namespace) -> dict[str, object]:
    accel    = getattr(args, "accelerator", "auto")
    devices  = getattr(args, "devices",     "auto")
    strategy = getattr(args, "strategy",    "auto")

    if (
        getattr(args, "auto_ddp", False)
        and devices  == "auto"
        and strategy == "auto"
        and accel in {"auto", "gpu", "cuda"}
    ):
        n = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if n > 1:
            return {"accelerator": "gpu", "devices": n, "strategy": "ddp"}
        if n == 1:
            return {"accelerator": "gpu", "devices": 1, "strategy": "auto"}

    return {"accelerator": accel, "devices": devices, "strategy": strategy}


def _auto_detect_dataloader(args: argparse.Namespace) -> None:
    """
    For tasks that use paired/triplet data, set dataset and dataloader automatically.
    For downstream tasks, require them explicitly from the user.
    """
    if args.task in _PAIRED_TASKS:
        auto = _PAIRED_TASKS[args.task]
        if args.dataset or args.dataloader:
            print(
                f"[WARN] --dataset/--dataloader ignored for '{args.task}'. "
                f"Using '{auto}' automatically."
            )
        args.dataset    = auto
        args.dataloader = auto
        print(f"[INFO] Task '{args.task}' → dataloader='{auto}'.")
    else:
        if not args.dataset or not args.dataloader:
            raise ValueError(
                f"Task '{args.task}' requires explicit --dataset and --dataloader."
            )


def _build_module(bundle: ModelBundle, spec, lr: float) -> L.LightningModule:
    """
    Instantiate the correct Lightning module from a ModelBundle.

    torch.compile is applied to `module.student` only (not the frozen teacher,
    and not the whole LightningModule) so that Lightning checkpointing remains
    straightforward and attribute access is preserved.
    """
    task = bundle.task

    if task == "pretrain_reconstruction":
        module = SSLPretrainModule(bundle.model, spec, lr=lr)

    elif task == "domain_adaptation":
        module = DomainAdaptationModule(
            student_model=bundle.student,
            teacher_model=bundle.teacher,
            spec=spec,
            lr=lr,
        )

    elif task == "distillation_kd":
        raise NotImplementedError(
            "CrossArchKDModule not wired yet — implement Phase 2 here."
        )

    elif task == "eval_domain_gap":
        module = DomainEvalModule(
            teacher_encoder=bundle.teacher,
            student_encoder=bundle.student,
            decoders=bundle.decoders,
        )

    else:
        # All downstream tasks: segmentation, classification, regression
        module = DownstreamModule(bundle.model, spec, lr=lr)

    # Compile only the trainable student — skip frozen teacher and eval-only modules
    if (
        torch.__version__ >= "2.0"
        and task not in {"eval_domain_gap"}
        and hasattr(module, "student")
    ):
        module.student = torch.compile(module.student)

    return module


# ─────────────────────────────────────────────────────────────────────────────
# fit
# ─────────────────────────────────────────────────────────────────────────────

def run_fit(args: argparse.Namespace) -> None:
    _auto_detect_dataloader(args)

    spec        = resolve_task_spec(args.task, args.dataset)
    output_root = Path(args.output_dir)
    subset_name = Path(args.subset_csv).stem if args.subset_csv else "full_dataset"

    for seed in args.seeds:
        seed_everything(seed)
        L.seed_everything(seed, workers=True)

        seed_dir = (
            output_root
            / spec.task / spec.dataset / args.model / subset_name / f"seed_{seed}"
        )
        seed_dir.mkdir(parents=True, exist_ok=True)

        # ── Data ─────────────────────────────────────────────────────────
        datamodule = build_datamodule(
            args.dataloader,
            root_dir=args.root_dir,
            spec=spec,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=seed,
            crop_size=args.crop_size,
            fast_dev_run=args.fast_dev_run,
            subset_csv=args.subset_csv,
        )

        # ── Model ─────────────────────────────────────────────────────────
        bundle = build_model(
            args.model,
            spec,
            pretrained=args.pretrained,
            input_bands=datamodule.input_bands,
            weights_path=args.weights,
            teacher_ckpt=args.teacher_ckpt,
            student_ckpt=args.student_ckpt,
            decoders=args.decoders,
        )

        # ── Lightning module ──────────────────────────────────────────────
        module = _build_module(bundle, spec, lr=args.lr)

        # ── Callbacks ─────────────────────────────────────────────────────
        callbacks = []
        if not args.fast_dev_run:
            callbacks.append(
                ModelCheckpoint(
                    dirpath=seed_dir / "checkpoints",
                    filename="best",
                    monitor="val_loss",
                    mode="min",
                    save_last=True,
                )
            )
            if args.patience is not None:
                callbacks.append(
                    EarlyStopping(monitor="val_loss", patience=args.patience, mode="min")
                )

        # ── Trainer ───────────────────────────────────────────────────────
        run_name = f"{spec.task}_{spec.dataset}_{args.model}_{subset_name}_s{seed}"
        hardware = _resolve_hardware(args)

        trainer = L.Trainer(
            **hardware,
            precision=args.precision,
            max_epochs=args.max_epochs,
            default_root_dir=seed_dir,
            logger=WandbLogger(
                project="PhiSat2",
                name=run_name,
                save_dir=seed_dir,
                config=vars(args),
            ),
            callbacks=callbacks,
            fast_dev_run=args.fast_dev_run,
            log_every_n_steps=1,
        )

        # ── Resume ────────────────────────────────────────────────────────
        ckpt = None
        if args.resume:
            last = seed_dir / "checkpoints" / "last.ckpt"
            if last.exists():
                ckpt = str(last)
                print(f"[INFO] Resuming from {ckpt}")

        trainer.fit(module, datamodule=datamodule, ckpt_path=ckpt)


# ─────────────────────────────────────────────────────────────────────────────
# test
# ─────────────────────────────────────────────────────────────────────────────

def run_test(args: argparse.Namespace) -> None:
    _auto_detect_dataloader(args)

    spec = resolve_task_spec(args.task, args.dataset)

    seed_everything(args.seed)
    L.seed_everything(args.seed, workers=True)

    subset_name = Path(args.subset_csv).stem if args.subset_csv else "full_dataset"
    eval_dir = (
        Path(args.output_dir)
        / spec.task / spec.dataset / args.model / subset_name / f"eval_seed_{args.seed}"
    )
    eval_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────
    datamodule = build_datamodule(
        args.dataloader,
        root_dir=args.root_dir,
        spec=spec,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        crop_size=args.crop_size,
        fast_dev_run=False,
        subset_csv=args.subset_csv,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    bundle = build_model(
        args.model,
        spec,
        pretrained=False,
        input_bands=datamodule.input_bands,
        weights_path=args.weights,
        teacher_ckpt=args.teacher_ckpt,
        student_ckpt=args.student_ckpt,
        decoders=args.decoders,
    )

    # lr=0.0 — not used at test time but required by module constructors
    module = _build_module(bundle, spec, lr=0.0)

    # ── Trainer ───────────────────────────────────────────────────────────
    run_name = f"test_{spec.task}_{spec.dataset}_{args.model}"
    hardware = _resolve_hardware(args)

    trainer = L.Trainer(
        **hardware,
        precision=args.precision,
        default_root_dir=eval_dir,
        #logger=WandbLogger(
        #    project="PhiSat2",
        #    name=run_name,
        #    save_dir=eval_dir,
        #    config=vars(args),
        #),
    )

    # For eval_domain_gap, encoder weights are already baked in by build_model
    # (loaded from --teacher-ckpt / --student-ckpt). No Lightning ckpt to reload.
    # For all other tasks, --weights is the full Lightning .ckpt to restore.
    ckpt_path = None if args.task == "eval_domain_gap" else args.weights

    results = trainer.test(module, datamodule=datamodule, ckpt_path=ckpt_path)

    if results:
        out = eval_dir / "test_metrics.json"
        with open(out, "w") as f:
            json.dump(results[0], f, indent=4)
        print(f"[INFO] Metrics saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args   = parser.parse_args(argv)

    if args.command == "list-models":
        for e in list_models():
            print(f"{e.name:<25} {e.role:<10} {e.description}")
        return

    if args.command == "list-dataloaders":
        for e in list_dataloaders():
            print(f"{e.name:<25} {e.description}")
        return

    if args.command == "fit":
        run_fit(args)
    elif args.command == "test":
        run_test(args)


if __name__ == "__main__":
    main()