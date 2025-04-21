import os

import torch
from torchvision.models import convnext_base

model_dir = "./pretrained_models"

convnext_url = "https://dl.fbaipublicfiles.com/convnext/"

# only models with (224x224) resolution
# for 1k models, using "exponential moving average" versions
convnext_model_path = {
    "convnext_t_1k": "convnext_tiny_1k_224_ema.pth",
    "convnext_t_22k": "convnext_tiny_22k_224.pth",
    "convnext_s_1k": "convnext_small_1k_224_ema.pth",
    "convnext_s_22k": "convnext_small_22k_224.pth",
    "convnext_b_1k": "convnext_base_1k_224_ema.pth",
    "convnext_b_22k": "convnext_base_22k_224.pth",
    "convnext_l_1k": "convnext_large_1k_224_ema.pth",
    "convnext_l_22k": "convnext_large_22k_224.pth",
}


def create_convnext(model_type, pretrained=False):
    model = convnext_base(weights=None)

    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            os.path.join(convnext_url, convnext_model_path[model_type]),
            map_location="cpu",
            check_hash=True,
        )
        checkpoint_model = checkpoint["model"]

        model._load_from_state_dict(checkpoint_model, "", {}, True, [], [], [])

    return model.features


def convnext_t_1k_features(pretrained=False):
    return create_convnext(model_type="convnext_t_1k", pretrained=pretrained)


def convnext_t_22k_features(pretrained=False):
    return create_convnext(model_type="convnext_t_22k", pretrained=pretrained)


def convnext_s_1k_features(pretrained=False):
    return create_convnext(model_type="convnext_s_1k", pretrained=pretrained)


def convnext_s_22k_features(pretrained=False):
    return create_convnext(model_type="convnext_s_22k", pretrained=pretrained)


def convnext_b_1k_features(pretrained=False):
    return create_convnext(model_type="convnext_b_1k", pretrained=pretrained)


def convnext_b_22k_features(pretrained=False):
    return create_convnext(model_type="convnext_b_22k", pretrained=pretrained)


def convnext_l_1k_features(pretrained=False):
    return create_convnext(model_type="convnext_l_1k", pretrained=pretrained)


def convnext_l_22k_features(pretrained=False):
    return create_convnext(model_type="convnext_l_22k", pretrained=pretrained)


if __name__ == "__main__":
    convnext_t_1k = convnext_t_1k_features(pretrained=True)
    print(convnext_t_1k)

    convnext_t_22k = convnext_t_22k_features(pretrained=True)
    print(convnext_t_22k)

    convnext_s_1k = convnext_s_1k_features(pretrained=True)
    print(convnext_s_1k)

    convnext_s_22k = convnext_s_22k_features(pretrained=True)
    print(convnext_s_22k)

    convnext_b_1k = convnext_b_1k_features(pretrained=True)
    print(convnext_b_1k)

    convnext_b_22k = convnext_b_22k_features(pretrained=True)
    print(convnext_b_22k)

    convnext_l_1k = convnext_l_1k_features(pretrained=True)
    print(convnext_l_1k)

    convnext_l_22k = convnext_l_22k_features(pretrained=True)
    print(convnext_l_22k)
