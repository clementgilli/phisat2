import argparse
import subprocess
import sys

from phisat2.cli import train


def test_cli_parses_multiple_seeds():
    args = train.build_parser().parse_args(
        [
            "fit",
            "--task",
            "segmentation",
            "--dataset",
            "lulc",
            "--model",
            "phisat2_geoaware",
            "--dataloader",
            "synthetic",
            "--seeds",
            "1",
            "2",
        ]
    )
    assert args.seeds == [1, 2]


def test_run_fit_creates_seed_directories(monkeypatch, tmp_path):
    class FakeTrainer:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

        def fit(self, module, datamodule=None):
            self.module = module
            self.datamodule = datamodule

    monkeypatch.setattr(train.L, "Trainer", FakeTrainer)
    args = argparse.Namespace(
        task="segmentation",
        dataset="clouds",
        model="phisat2_geoaware",
        dataloader="synthetic",
        seeds=[3, 4],
        root_dir=".",
        output_dir=str(tmp_path),
        max_epochs=1,
        batch_size=2,
        lr=1e-4,
        num_workers=0,
        accelerator="cpu",
        devices="1",
        precision="32-true",
        fast_dev_run=True,
        pretrained=False,
    )
    train.run_fit(args)
    assert (tmp_path / "segmentation" / "clouds" / "phisat2_geoaware" / "seed_3").is_dir()
    assert (tmp_path / "segmentation" / "clouds" / "phisat2_geoaware" / "seed_4").is_dir()


def test_make_train_dry_run_uses_python_cli():
    result = subprocess.run(
        [
            "make",
            "-n",
            "train",
            "TASK=segmentation",
            "DATASET=clouds",
            "MODEL=phisat2_geoaware",
            "DATALOADER=synthetic",
            "SEEDS=1 2",
            "EPOCHS=1",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "-m phisat2.cli.train fit" in result.stdout
    assert "--seeds 1 2" in result.stdout


def test_module_lists_models():
    result = subprocess.run(
        [sys.executable, "-m", "phisat2.cli.train", "list-models"],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "terramind_v1_tiny" in result.stdout
