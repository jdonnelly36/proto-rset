import torch

from protopnet.embedding import AddonLayers


def test_addon_output_dim():
    add_on_layers = AddonLayers(
        num_prototypes=3 * 2,
        input_channels=512,
        proto_channel_multiplier=2**0,
        num_addon_layers=0,
    )
    out = add_on_layers(torch.randn(1, 512, 7, 7))
    assert out.shape[1] == 512

    add_on_layers = AddonLayers(
        num_prototypes=3 * 2,
        input_channels=512,
        proto_channel_multiplier=2**-1,
        num_addon_layers=2,
    )
    out = add_on_layers(torch.randn(1, 512, 7, 7))
    assert out.shape[1] == 512 // 2
