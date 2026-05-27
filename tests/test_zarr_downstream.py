import numpy as np
import torch
import zarr

from phisat2.data_loaders.zarr_downstream import ZarrDownstreamDataset
from phisat2.tasks import resolve_task_spec


def test_zarr_downstream_loads_phileo_lc_store(tmp_path):
    store = tmp_path / "phileo-bench_lc.zarr"
    for split, count in (("trainval", 4), ("test", 1)):
        for index in range(count):
            patch = zarr.open_group(store / split / f"{index:07d}", mode="w", zarr_format=3)
            patch.create_array(
                "img",
                data=np.ones((8, 16, 16), dtype=np.float32),
                chunks=(4, 8, 8),
            )
            patch.create_array(
                "label",
                data=np.full((1, 16, 16), index % 2, dtype=np.float32),
                chunks=(1, 8, 16),
            )

    spec = resolve_task_spec("segmentation", "lc")
    train_dataset = ZarrDownstreamDataset(tmp_path, spec, split="train", seed=0, val_ratio=0.25, crop_size=8)
    val_dataset = ZarrDownstreamDataset(store, spec, split="val", seed=0, val_ratio=0.25, crop_size=8)

    assert len(train_dataset) == 3
    assert len(val_dataset) == 1
    sample = train_dataset[0]
    assert sample["image"].shape == (8, 8, 8)
    assert sample["mask"].shape == (8, 8)
    assert sample["image"].dtype == torch.float32
    assert sample["mask"].dtype == torch.int64
