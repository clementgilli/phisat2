from phisat2.data_loaders import list_dataloaders
from phisat2.models import list_models
from phisat2.tasks import resolve_task_spec


def test_dataloader_registry_contains_public_loaders():
    names = {entry.name for entry in list_dataloaders()}
    assert {"zarr_downstream", "h5_pairs", "synthetic"} <= names


def test_model_registry_marks_myriad_exception():
    entries = {entry.name: entry for entry in list_models()}
    assert entries["phisat2_geoaware"].shared_decoder is True
    assert entries["dofa_small_patch16_224"].shared_decoder is True
    assert entries["seco_resnet18_sentinel2_rgb_seco"].shared_decoder is True
    assert entries["satlas_swin_t_sentinel2_si_ms"].shared_decoder is True
    assert entries["myriad2_full_unet"].shared_decoder is False


def test_task_specs_define_expected_outputs():
    assert resolve_task_spec("segmentation", "lulc").num_outputs == 11
    assert resolve_task_spec("classification", "climate").num_outputs == 31
    assert resolve_task_spec("global_regression", "geoloc").num_outputs == 4
    assert resolve_task_spec("pixel_regression", "reconstruction").num_outputs == 8
