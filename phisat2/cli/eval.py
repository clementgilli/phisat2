from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.loggers import WandbLogger
import json

from phisat2.data_loaders import build_datamodule, list_dataloaders
from phisat2.models import build_model, list_models
from phisat2.tasks import resolve_task_spec
from phisat2.training.lightning_module import PhiSat2LightningModule
from phisat2.utils.seed import seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PhiSat-2 Makefile-driven Lightning evaluator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    test_parser = subparsers.add_parser("test", help="Run evaluation on a specific checkpoint.")
    test_parser.add_argument("--task", required=True, choices=["segmentation", "pixel_regression", "classification", "global_regression"])
    test_parser.add_argument("--dataset", required=True)
    test_parser.add_argument("--model", required=True)
    test_parser.add_argument("--dataloader", required=True)
    test_parser.add_argument("--ckpt-path", type=str, required=True, help="Chemin vers le fichier best.ckpt")    
    test_parser.add_argument("--seed", type=int, default=42, help="Seed unique pour l'évaluation")
    test_parser.add_argument("--root-dir", default=".")
    test_parser.add_argument("--output-dir", default="eval_runs", help="Dossier séparé pour les logs de test")
    test_parser.add_argument("--batch-size", type=int, default=16)
    test_parser.add_argument("--crop-size", type=int, default=224)
    test_parser.add_argument("--num-workers", type=int, default=4)
    test_parser.add_argument("--accelerator", default="auto")
    test_parser.add_argument("--devices", default="auto")
    test_parser.add_argument("--strategy", default="auto")
    test_parser.add_argument("--precision", default="32-true")
    test_parser.add_argument(
        "--auto-ddp",
        action="store_true",
        help="Use every visible CUDA GPU and DDP when hardware settings are otherwise auto.",
    )

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
    run_test(args)


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


def run_test(args: argparse.Namespace) -> None:
    spec = resolve_task_spec(args.task, args.dataset)
    output_root = Path(args.output_dir)
    
    seed_everything(args.seed)
    L.seed_everything(args.seed, workers=True)
    
    eval_dir = output_root / spec.task / spec.dataset / args.model / f"eval_seed_{args.seed}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    datamodule = build_datamodule(
        args.dataloader,
        root_dir=args.root_dir,
        spec=spec,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        crop_size=args.crop_size,
        fast_dev_run=False,
        subset_csv=None,
    )
    
    model = build_model(args.model, spec, pretrained=False)
    model = torch.compile(model) if torch.__version__ >= "2.0" else model
    
    module = PhiSat2LightningModule.load_from_checkpoint(
        args.ckpt_path,
        model=model,
        spec=spec,
        lr=0.0
    )

    hardware = resolve_trainer_hardware(args)

    ckpt_parent_name = Path(args.ckpt_path).parent.parent.name
    run_name = f"eval_{args.model}_trained_on_{ckpt_parent_name}"

    trainer = L.Trainer(
        accelerator=hardware["accelerator"],
        devices=hardware["devices"],
        strategy=hardware["strategy"],
        precision=args.precision,
        default_root_dir=eval_dir,
        logger=WandbLogger(project="PhiSat2", name=run_name, save_dir=eval_dir, config=vars(args)),
    )
    
    results = trainer.test(module, datamodule=datamodule)

    if results:
        metrics_file = eval_dir / "test_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(results[0], f, indent=4)

if __name__ == "__main__":
    main()