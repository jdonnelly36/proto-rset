import pytest
import torch

from protopnet.prediction_heads import PrototypePredictionHead


def test_topk():
    num_classes = 3
    prototype_class_identity = torch.randn((num_classes, num_classes))

    input = torch.randn(10, num_classes, 224, 224)
    prev_activations = None
    for k in range(1, 4):
        prototype_config = {
            "prototype_class_identity": prototype_class_identity,
            "num_classes": num_classes,
            "k_for_topk": k,
        }

        prediction_head = PrototypePredictionHead(**prototype_config)
        activations = prediction_head.forward(
            input, None, return_similarity_score_to_each_prototype=True
        )["similarity_score_to_each_prototype"]

        if prev_activations is not None:
            assert (prev_activations >= activations).all()

        prev_activations = activations
