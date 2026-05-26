from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from phisat2.data_loaders import build_datamodule, list_dataloaders
from phisat2.models import build_model, list_models
from phisat2.tasks import resolve_task_spec
from phisat2.training.lightning_module import PhiSat2LightningModule
from phisat2.utils.seed import seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PhiSat-2 Makefile-driven Lightning trainer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit = subparsers.add_parser("fit", help="Run training for one or more seeds.")
    fit.add_argument("--task", required=True, choices=["segmentation", "pixel_regression", "classification", "global_regression"])
    fit.add_argument("--dataset", required=True)
    fit.add_argument("--model", required=True)
    fit.add_argument("--dataloader", required=True)
    fit.add_argument("--seeds", nargs="+", type=int, required=True)
    fit.add_argument("--root-dir", default=".")
    fit.add_argument("--output-dir", default="runs")
    fit.add_argument("--max-epochs", type=int, default=50)
    fit.add_argument("--batch-size", type=int, default=16)
    fit.add_argument("--crop-size", type=int, default=224)
    fit.add_argument("--lr", type=float, default=1e-4)
    fit.add_argument("--num-workers", type=int, default=4)
    fit.add_argument("--accelerator", default="auto")
    fit.add_argument("--devices", default="auto")
    fit.add_argument("--strategy", default="auto")
    fit.add_argument("--precision", default="32-true")
    fit.add_argument(
        "--auto-ddp",
        action="store_true",
        help="Use every visible CUDA GPU and DDP when hardware settings are otherwise auto.",
    )
    fit.add_argument("--fast-dev-run", action="store_true")
    pretrained = fit.add_mutually_exclusive_group()
    pretrained.add_argument("--pretrained", dest="pretrained", action="store_true")
    pretrained.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    fit.set_defaults(pretrained=True)

    subparsers.add_parser("list-models", help="List registered model names.")
    subparsers.add_parser("list-dataloaders", help="List registered dataloader names.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "list-models":
        for entry in list_models():
            marker = "shared-decoder" if entry.shared_decoder else "full-structure"
            print(f"{entry.name}\t{marker}\t{entry.description}")
        return
    if args.command == "list-dataloaders":
        for entry in list_dataloaders():
            print(f"{entry.name}\t{entry.description}")
        return
    run_fit(args)


def resolve_trainer_hardware(args: argparse.Namespace) -> dict[str, object]:
    accelerator = getattr(args, "accelerator", "auto")
    devices = getattr(args, "devices", "auto")
    strategy = getattr(args, "strategy", "auto")
    auto_ddp = getattr(args, "auto_ddp", False)

    if (
        auto_ddp
        and devices == "auto"
        and strategy == "auto"
        and accelerator in {"auto", "gpu", "cuda"}
    ):
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if gpu_count > 1:
            return {"accelerator": "gpu", "devices": gpu_count, "strategy": "ddp"}
        if gpu_count == 1:
            return {"accelerator": "gpu", "devices": 1, "strategy": "auto"}

    return {"accelerator": accelerator, "devices": devices, "strategy": strategy}


def run_fit(args: argparse.Namespace) -> None:
    spec = resolve_task_spec(args.task, args.dataset)
    output_root = Path(args.output_dir)
    for seed in args.seeds:
        seed_everything(seed)
        L.seed_everything(seed, workers=True)
        seed_dir = output_root / spec.task / spec.dataset / args.model / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        datamodule = build_datamodule(
            args.dataloader,
            root_dir=args.root_dir,
            spec=spec,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=seed,
            crop_size=args.crop_size,
            fast_dev_run=args.fast_dev_run,
        )
        model = build_model(args.model, spec, pretrained=args.pretrained)
        module = PhiSat2LightningModule(model, spec, lr=args.lr)
        hardware = resolve_trainer_hardware(args)
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

        trainer = L.Trainer(
            accelerator=hardware["accelerator"],
            devices=hardware["devices"],
            strategy=hardware["strategy"],
            precision=args.precision,
            max_epochs=args.max_epochs,
            default_root_dir=seed_dir,
            logger=CSVLogger(save_dir=seed_dir, name="logs"),
            callbacks=callbacks,
            fast_dev_run=args.fast_dev_run,
            log_every_n_steps=1,
        )
        trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    main()
