import torch

from protopnet.backbones import construct_backbone, model_zoo_features

densenet_size_map = {
    121: (1, 1024, 7, 7),
    161: (1, 2208, 7, 7),
    169: (1, 1664, 7, 7),
    201: (1, 1920, 7, 7),
}


def test_backbone_dimensions():
    for name, _ in model_zoo_features.items():
        if name == "spikenet":
            continue

        backbone = construct_backbone(name, pretrained=False)
        input = torch.randn(1, 3, 224, 224)
        output = backbone(input)

        if name.startswith("vgg"):
            assert output.shape == (1, 512, 14, 14), (name, output.shape)
        elif name.startswith("resnet"):
            assert output.shape == (1, 512, 7, 7) or output.shape == (1, 2048, 7, 7), (
                name,
                output.shape,
            )
        elif name.startswith("densenet"):
            assert output.shape == densenet_size_map[int(name[-3:])], (
                name,
                output.shape,
            )

        assert backbone.latent_dimension == output.shape[1:], (
            name,
            backbone.latent_dimension,
            output.shape[1:],
        )
