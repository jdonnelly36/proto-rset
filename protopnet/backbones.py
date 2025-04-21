import logging

import torch
import torchvision

from .embedding import EmbeddedBackbone
from .pretrained.convnext_features import (
    convnext_b_1k_features,
    convnext_b_22k_features,
    convnext_l_1k_features,
    convnext_l_22k_features,
    convnext_s_1k_features,
    convnext_s_22k_features,
    convnext_t_1k_features,
    convnext_t_22k_features,
)
from .pretrained.densenet_features import (
    densenet121_features,
    densenet161_features,
    densenet169_features,
    densenet201_features,
)
from .pretrained.resnet_features import (
    resnet18_features,
    resnet34_features,
    resnet50_features,
    resnet101_features,
    resnet152_features,
)
from .pretrained.spikenet_features import spikenet_features
from .pretrained.vgg_features import (
    vgg11_bn_features,
    vgg11_features,
    vgg13_bn_features,
    vgg13_features,
    vgg16_bn_features,
    vgg16_features,
    vgg19_bn_features,
    vgg19_features,
)

log = logging.getLogger(__name__)

model_zoo_features = {
    "resnet18": resnet18_features,
    "resnet34": resnet34_features,
    "resnet50": resnet50_features,
    "resnet101": resnet101_features,
    "resnet152": resnet152_features,
    "densenet121": densenet121_features,
    "densenet161": densenet161_features,
    "densenet169": densenet169_features,
    "densenet201": densenet201_features,
    "vgg11": vgg11_features,
    "vgg11_bn": vgg11_bn_features,
    "vgg13": vgg13_features,
    "vgg13_bn": vgg13_bn_features,
    "vgg16": vgg16_features,
    "vgg16_bn": vgg16_bn_features,
    "vgg19": vgg19_features,
    "vgg19_bn": vgg19_bn_features,
    "spikenet": spikenet_features,
    "convnext_t_1k": convnext_t_1k_features,
    "convnext_t_22k": convnext_t_22k_features,
    "convnext_s_1k": convnext_s_1k_features,
    "convnext_s_22k": convnext_s_22k_features,
    "convnext_b_1k": convnext_b_1k_features,
    "convnext_b_22k": convnext_b_22k_features,
    "convnext_l_1k": convnext_l_1k_features,
    "convnext_l_22k": convnext_l_22k_features,
}


def construct_backbone(base_architecture, pretrained=True, input_channels: int = (3, 224, 224)):
    if base_architecture in model_zoo_features:
        log.info(f"Using {base_architecture} backbone from model zoo")
        return EmbeddedBackbone(
            model_zoo_features[base_architecture](pretrained=pretrained),
            input_channels=input_channels
        )
    elif base_architecture in torch.hub.list(
        f"pytorch/vision:v{torchvision.__version__.split('+')[0]}"
    ):
        log.info(f"Using {base_architecture} backbone from torchhub")
        return EmbeddedBackbone(
            torch.hub.load(
                f"pytorch/vision:v{torchvision.__version__.split('+')[0]}",
                base_architecture,
                pretrained=pretrained,
            ).features,
            input_channels=input_channels
        )
    else:
        raise ValueError(f"Unknown base architecture {base_architecture}")
