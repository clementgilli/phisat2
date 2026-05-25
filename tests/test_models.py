import torch
import pytest

from phisat2.models import build_model
from phisat2.models.decoders.shared_unet import SharedUNetDecoder
from phisat2.models.encoders.myriad2_full import Myriad2FullUNet
from phisat2.tasks import resolve_task_spec


def test_shared_decoder_forward_shape():
    decoder = SharedUNetDecoder((8, 16, 32, 64), output_channels=5)
    features = [
        torch.randn(2, 8, 32, 32),
        torch.randn(2, 16, 16, 16),
        torch.randn(2, 32, 8, 8),
        torch.randn(2, 64, 4, 4),
    ]
    assert decoder(features).shape == (2, 5, 32, 32)


def test_phisat2_geoaware_segmentation_forward():
    spec = resolve_task_spec("segmentation", "clouds")
    model = build_model("phisat2_geoaware", spec, pretrained=False)
    output = model(torch.randn(2, 8, 32, 32))
    assert output.shape == (2, 2, 32, 32)


def test_phisat2_geoaware_classification_forward():
    spec = resolve_task_spec("classification", "climate")
    model = build_model("phisat2_geoaware", spec, pretrained=False)
    output = model(torch.randn(2, 8, 32, 32))
    assert output.shape == (2, 31)


def test_myriad2_full_unet_matches_inspected_topology():
    model = Myriad2FullUNet(output_channels=4)
    assert tuple(model.encoders[0].channel_proj.weight.shape) == (16, 8, 1, 1)
    assert tuple(model.bottleneck.channel_proj.weight.shape) == (48, 32, 1, 1)
    assert tuple(model.final_conv.weight.shape) == (4, 16, 1, 1)
    assert tuple(model.classifier.weight.shape) == (4, 16, 1, 1)
    assert model(torch.randn(2, 8, 32, 32)).shape == (2, 4, 32, 32)


def test_myriad2_full_unet_rejects_non_spatial_tasks():
    spec = resolve_task_spec("classification", "climate")
    with pytest.raises(ValueError, match="only supports spatial tasks"):
        build_model("myriad2_full_unet", spec, pretrained=False)
