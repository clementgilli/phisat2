import torch

from phisat2.models import build_model
from phisat2.tasks import resolve_task_spec
from phisat2.training.lightning_module import PhiSat2LightningModule


def test_lightning_module_freezes_encoder_for_linear_probe_training():
    spec = resolve_task_spec("segmentation", "clouds")
    model = build_model("phisat2_geoaware", spec, pretrained=False)
    module = PhiSat2LightningModule(model, spec, lr=1e-4)

    encoder_params = list(model.encoder.parameters())
    assert encoder_params
    assert all(not param.requires_grad for param in encoder_params)

    module.train()
    assert module.training
    assert not model.encoder.training
    assert model.head.training

    optimizer = module.configure_optimizers()
    optimized_param_ids = {
        id(param)
        for group in optimizer.param_groups
        for param in group["params"]
    }
    encoder_param_ids = {id(param) for param in encoder_params}

    assert optimized_param_ids.isdisjoint(encoder_param_ids)
    assert all(id(param) in optimized_param_ids for param in model.adapter.parameters())
    assert all(id(param) in optimized_param_ids for param in model.head.parameters())


def test_lightning_training_forward_runs_encoder_without_grad():
    spec = resolve_task_spec("segmentation", "clouds")
    model = build_model("phisat2_geoaware", spec, pretrained=False)
    module = PhiSat2LightningModule(model, spec, lr=1e-4)
    grad_modes = []

    def record_encoder_grad_mode(_module, _inputs):
        grad_modes.append(torch.is_grad_enabled())

    hook = model.encoder.register_forward_pre_hook(record_encoder_grad_mode)
    try:
        module.train()
        prediction = module(torch.randn(2, 8, 32, 32))
    finally:
        hook.remove()

    assert grad_modes == [False]
    assert prediction.requires_grad
