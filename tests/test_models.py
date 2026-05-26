import torch
import pytest

from phisat2.models import build_model
from phisat2.models.adapters.feature_pyramid import FeaturePyramidAdapter
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


def test_feature_adapter_drops_cls_token_from_vit_features():
    adapter = FeaturePyramidAdapter((8, 16, 32, 64))
    pyramid = adapter(torch.randn(2, 65, 32), image_size=(128, 128))
    assert [feature.shape for feature in pyramid] == [
        torch.Size([2, 8, 128, 128]),
        torch.Size([2, 16, 64, 64]),
        torch.Size([2, 32, 32, 32]),
        torch.Size([2, 64, 16, 16]),
    ]


def test_feature_adapter_accepts_channels_last_swin_features():
    adapter = FeaturePyramidAdapter((8, 16, 32, 64))
    pyramid = adapter(torch.randn(2, 7, 7, 32), image_size=(224, 224))
    assert [feature.shape for feature in pyramid] == [
        torch.Size([2, 8, 224, 224]),
        torch.Size([2, 16, 112, 112]),
        torch.Size([2, 32, 56, 56]),
        torch.Size([2, 64, 28, 28]),
    ]


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


@pytest.mark.parametrize(
    "model_name",
    [
        "dofa_small_patch16_224",
        "seco_resnet18_sentinel2_rgb_seco",
        "satlas_swin_t_sentinel2_si_ms",
    ],
)
def test_gfm_backbones_forward_with_lulc_channels(model_name):
    spec = resolve_task_spec("segmentation", "lulc")
    model = build_model(model_name, spec, pretrained=False).eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 8, 224, 224))
    assert output.shape == (1, 11, 224, 224)
