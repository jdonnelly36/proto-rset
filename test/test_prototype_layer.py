import pytest
import torch

from protopnet.prototype_layers import PrototypeLayer, DeformablePrototypeLayer
from protopnet.activations import CosPrototypeActivation


def test_prototype_density_metrics():
    num_classes = 3
    prototype_class_identity = torch.randn((num_classes * 2, 3))

    prototype_config = {
        "prototype_class_identity": prototype_class_identity,
        "num_classes": num_classes,
    }

    # Test construction with cosine activation ----------------
    cosine_activation = CosPrototypeActivation()

    prototype_layer = PrototypeLayer(
        activation_function=cosine_activation,
        **prototype_config,
        prototype_dimension=(2, 2),
    )

    proto_tensors = torch.ones_like(prototype_layer.prototype_tensors)

    # Running forward to compute latent space size
    prototype_layer(torch.randn(1, proto_tensors.shape[1], 14, 14))

    prototype_layer.set_prototype_tensors(proto_tensors)
    unique_proto_stats = prototype_layer.get_prototype_complexity()

    assert (
        unique_proto_stats["n_unique_proto_parts"] == 1
    ), f'Error: With only 1 unique part, reported {unique_proto_stats["n_unique_proto_parts"]} parts'
    assert (
        unique_proto_stats["n_unique_protos"] == 1
    ), f'Error: With only 1 unique prototypes, reported {unique_proto_stats["n_unique_protos"]} prototypes'

    proto_tensors[0, :, :, :] = torch.randn(*proto_tensors[0, :, :, :].shape)
    prototype_layer.set_prototype_tensors(proto_tensors)
    unique_proto_stats = prototype_layer.get_prototype_complexity()

    assert (
        unique_proto_stats["n_unique_proto_parts"] == 5
    ), f'Error: With 5 unique parts, reported {unique_proto_stats["n_unique_proto_parts"]} parts'
    assert (
        unique_proto_stats["n_unique_protos"] == 2
    ), f'Error: With 2 unique prototypes, reported {unique_proto_stats["n_unique_protos"]} prototypes'

    proto_tensors[1, :, :, :] = proto_tensors[0, :, :, :] + 1e-12
    prototype_layer.set_prototype_tensors(proto_tensors)
    unique_proto_stats = prototype_layer.get_prototype_complexity()

    assert (
        unique_proto_stats["n_unique_proto_parts"] == 5
    ), f'Error: With 5 unique parts and added floating point error, reported {unique_proto_stats["n_unique_proto_parts"]} parts'
    assert (
        unique_proto_stats["n_unique_protos"] == 2
    ), f'Error: With 2 unique prototypes and added floating point error, reported {unique_proto_stats["n_unique_protos"]} prototypes'


def test_prototype_density_metrics_deformable():
    num_classes = 3
    prototype_class_identity = torch.randn((num_classes * 2, 3))

    prototype_config = {
        "prototype_class_identity": prototype_class_identity,
        "num_classes": num_classes,
    }

    # Test construction with cosine activation ----------------
    cosine_activation = CosPrototypeActivation()

    prototype_layer = DeformablePrototypeLayer(
        activation_function=cosine_activation,
        **prototype_config,
        prototype_dimension=(2, 2),
    )

    proto_tensors = torch.ones_like(prototype_layer.prototype_tensors)

    # Running forward to compute latent space size
    prototype_layer(torch.randn(1, proto_tensors.shape[1], 14, 14))

    prototype_layer.set_prototype_tensors(proto_tensors)
    unique_proto_stats = prototype_layer.get_prototype_complexity()

    assert (
        unique_proto_stats["n_unique_proto_parts"] == 1
    ), f'Error: With only 1 unique part, reported {unique_proto_stats["n_unique_proto_parts"]} parts'
    assert (
        unique_proto_stats["n_unique_protos"] == 1
    ), f'Error: With only 1 unique prototypes, reported {unique_proto_stats["n_unique_protos"]} prototypes'

    proto_tensors[0, :, :, :] = torch.randn(*proto_tensors[0, :, :, :].shape)
    prototype_layer.set_prototype_tensors(proto_tensors)
    unique_proto_stats = prototype_layer.get_prototype_complexity()

    assert (
        unique_proto_stats["n_unique_proto_parts"] == 5
    ), f'Error: With 5 unique parts, reported {unique_proto_stats["n_unique_proto_parts"]} parts'
    assert (
        unique_proto_stats["n_unique_protos"] == 2
    ), f'Error: With 2 unique prototypes, reported {unique_proto_stats["n_unique_protos"]} prototypes'

    proto_tensors[1, :, :, :] = proto_tensors[0, :, :, :] + 1e-12
    prototype_layer.set_prototype_tensors(proto_tensors)
    unique_proto_stats = prototype_layer.get_prototype_complexity()

    assert (
        unique_proto_stats["n_unique_proto_parts"] == 5
    ), f'Error: With 5 unique parts and added floating point error, reported {unique_proto_stats["n_unique_proto_parts"]} parts'
    assert (
        unique_proto_stats["n_unique_protos"] == 2
    ), f'Error: With 2 unique prototypes and added floating point error, reported {unique_proto_stats["n_unique_protos"]} prototypes'
