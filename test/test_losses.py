import math

import numpy as np
import pytest
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from torch.nn.functional import one_hot

from protopnet.activations import CosPrototypeActivation
from protopnet.backbones import construct_backbone
from protopnet.datasets.util import calculate_class_weights
from protopnet.losses import (
    AverageSeparationCost,
    BalancedCrossEntropyCost,
    ClusterCost,
    CrossEntropyCost,
    FineAnnotationCost,
    L1CostClassConnectionLayer,
    OrthogonalityLoss,
    SeparationCost,
)
from protopnet.prediction_heads import PrototypePredictionHead
from protopnet.prototype_layers import PrototypeLayer
from protopnet.prototypical_part_model import ProtoPNet


def test_cross_entropy_cost():
    # Do not need to test hard as it is a simple wrapper around torch.nn.functional.cross_entropy

    cost = CrossEntropyCost()

    logits = torch.tensor(
        [[1000000.0, 0.0, 0.0], [0.0, 1000000.0, 0.0], [0.0, 0.0, 1000000.0]]
    )
    targets = torch.tensor([0, 1, 2])

    # Cost expects the logits as opposed to probabilities
    loss = cost(logits, targets)
    assert loss == 0, loss


# Expanded parameterized test for class-specific cluster cost calculation
@pytest.mark.parametrize(
    "similarity_score_to_each_prototype, prototypes_of_correct_class, expected_cost",
    [
        # 1D Tensor
        # pytest.param(
        #     torch.tensor([0.1, 0.2, 0.3]),
        #     torch.tensor([0, 1, 0], dtype=torch.float32),
        #     torch.mean(torch.tensor([0.2])),
        #     id="1d_tensor",
        # ),
        # 2D Tensor
        pytest.param(
            torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]),
            torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=torch.float32),
            torch.mean(torch.tensor([0.2, 0.4, 0.9])),
            id="2d_tensor",
        ),
        # 2D Tensor with all zeros
        pytest.param(
            torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=torch.float32),
            torch.mean(torch.tensor([0.0, 0.0, 0.0])),
            id="2d_all_zeros",
        ),
        # 3D Tensor
        pytest.param(
            torch.tensor([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]),
            torch.tensor([[[0, 1], [1, 0]], [[1, 0], [0, 1]]], dtype=torch.float32),
            torch.mean(torch.tensor([0.2, 0.3, 0.5, 0.8])),
            id="3d_tensor",
        ),
        # 4D Tensor with negative values
        pytest.param(
            torch.tensor(
                [[[[0.1], [-0.2]], [[-0.3], [0.4]]], [[[0.5], [-0.6]], [[-0.7], [0.8]]]]
            ),
            torch.tensor(
                [[[[0], [1]], [[1], [0]]], [[[1], [0]], [[0], [1]]]],
                dtype=torch.float32,
            ),
            torch.mean(torch.tensor([0, 0, 0.5, 0.8])),
            id="4d_negative_values",
        ),
    ],
)
def test_cluster_cost_class_specific(
    similarity_score_to_each_prototype, prototypes_of_correct_class, expected_cost
):
    cluster_cost = ClusterCost(class_specific=True)
    loss = cluster_cost(similarity_score_to_each_prototype, prototypes_of_correct_class)
    assert torch.isclose(loss, expected_cost), f"Expected {expected_cost}, got {loss}"


def test_cluster_cost_1d():
    cluster_cost = ClusterCost()
    similarity_score_to_each_prototype = torch.tensor([0.1, 0.2, 0.3])
    with pytest.raises(AssertionError):
        cluster_cost(similarity_score_to_each_prototype)


# Expanded parameterized test for non-class-specific cluster cost calculation
@pytest.mark.parametrize(
    "similarity_score_to_each_prototype, expected_cost",
    [
        # 1D Tensor
        # pytest.param(
        #     torch.tensor([0.1, 0.2, 0.3]),
        #     torch.mean(torch.tensor([0.3])),
        #     id="1d_tensor",
        # ),
        # 2D Tensor
        pytest.param(
            torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]),
            torch.mean(torch.tensor([0.3, 0.6, 0.9])),
            id="2d_tensor",
        ),
        # 2D Tensor with all zeros
        pytest.param(
            torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            torch.mean(torch.tensor([0.0, 0.0, 0.0])),
            id="2d_all_zeros",
        ),
        # 2D Tensor with all zeros
        pytest.param(
            torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            torch.mean(torch.tensor([0.0, 0.0, 0.0])),
            id="2d_all_zeros",
        ),
        # 3D Tensor
        pytest.param(
            torch.tensor([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]),
            torch.mean(torch.tensor([0.3, 0.4, 0.7, 0.8])),
            id="3d_tensor",
        ),
        # 4D Tensor with negative values
        pytest.param(
            torch.tensor(
                [[[[0.1], [-0.2]], [[-0.3], [0.4]]], [[[0.5], [-0.6]], [[-0.7], [0.8]]]]
            ),
            torch.mean(torch.tensor([0.1, 0.4, 0.5, 0.8])),
            id="4d_negative_values",
        ),
    ],
)
def test_cluster_cost_non_class_specific(
    similarity_score_to_each_prototype, expected_cost
):
    cluster_cost = ClusterCost(class_specific=False)
    loss = cluster_cost(similarity_score_to_each_prototype)
    assert torch.isclose(loss, expected_cost), f"Expected {expected_cost}, got {loss}"


@pytest.mark.parametrize(
    "incorrect_class_prototype_activations, expected_cost",
    [
        # 1D Tensor
        pytest.param(
            torch.tensor([0.1, 0.2, 0.3]),
            torch.mean(torch.tensor([0.1, 0.2, 0.3])),
            id="1d_tensor",
        ),
        # 2D Tensor
        pytest.param(
            torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]),
            torch.mean(torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])),
            id="2d_tensor",
        ),
        # 2D Tensor with all zeros
        pytest.param(
            torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            torch.mean(torch.tensor([0.0, 0.0, 0.0])),
            id="2d_all_zeros",
        ),
        # 3D Tensor
        pytest.param(
            torch.tensor([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]),
            torch.mean(torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])),
            id="3d_tensor",
        ),
        # 4D Tensor with negative values
        pytest.param(
            torch.tensor(
                [[[[0.1], [-0.2]], [[-0.3], [0.4]]], [[[0.5], [-0.6]], [[-0.7], [0.8]]]]
            ),
            torch.mean(torch.tensor([0.1, -0.2, -0.3, 0.4, 0.5, -0.6, -0.7, 0.8])),
            id="4d_negative_values",
        ),
    ],
)
def test_separation_cost(incorrect_class_prototype_activations, expected_cost):
    separation_cost = SeparationCost()
    loss = separation_cost(incorrect_class_prototype_activations)
    assert torch.isclose(loss, expected_cost), f"Expected {expected_cost}, got {loss}"


def test_separation_cost_with_none():
    separation_cost = SeparationCost()
    with pytest.raises(ValueError):
        separation_cost(None)


@pytest.mark.parametrize(
    "weights, expected_l1_cost",
    [
        # Single weight
        pytest.param(torch.tensor([[1.0]]), 1.0, id="single_weight"),
        # 2x2 Matrix with positive and negative values
        pytest.param(torch.tensor([[1.0, -2.0], [3.0, -4.0]]), 10.0, id="2x2_matrix"),
        # 2x2 Matrix with all zeros
        pytest.param(torch.zeros((2, 2)), 0.0, id="2x2_zeros"),
        # 3x3 Matrix with mixed values
        pytest.param(
            torch.tensor([[1.0, -1.0, 2.0], [3.0, -3.0, 4.0], [5.0, -5.0, 6.0]]),
            30.0,
            id="3x3_mixed_values",
        ),
        # 4x4 Matrix with random values (expected cost to be calculated)
        pytest.param(torch.randn((4, 4)), "dynamic", id="4x4_random"),
    ],
)
def test_l1_cost_class_connection_layer(weights, expected_l1_cost):
    class MockProtoPNet(nn.Module):
        def __init__(self, weights):
            super(MockProtoPNet, self).__init__()
            self.prototype_prediction_head = nn.Module()
            self.prototype_prediction_head.class_connection_layer = nn.Module()
            self.prototype_prediction_head.class_connection_layer.weight = nn.Parameter(
                weights
            )

    model = MockProtoPNet(weights)
    l1_cost_layer = L1CostClassConnectionLayer()
    l1_cost = l1_cost_layer(model)

    if expected_l1_cost == "dynamic":
        expected_l1_cost = weights.abs().sum().item()

    assert torch.isclose(
        l1_cost, torch.tensor(expected_l1_cost)
    ), f"Expected {expected_l1_cost}, got {l1_cost.item()}"


@pytest.mark.parametrize(
    "incorrect_class_prototype_activations, prototypes_of_wrong_class, expected_cost",
    [
        # Basic test with uniform wrong class prototypes
        pytest.param(
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([[1.0], [1.0], [1.0]]),
            2.0,
            id="uniform_wrong_class_prototypes",
        ),
        # Test with varying wrong class prototypes
        pytest.param(
            torch.tensor([10.0, 20.0, 30.0]),
            torch.tensor([[2.0], [4.0], [6.0]]),
            5.0,
            id="varying_wrong_class_prototypes",
        ),
        # Test with zeros in activations, should handle division by zero
        pytest.param(
            torch.tensor([0.0, 0.0, 0.0]),
            torch.tensor([[1.0], [2.0], [3.0]]),
            0.0,
            id="zero_activations",
        ),
        # Test with all zeros in wrong class prototypes
        pytest.param(
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([[0.0], [0.0], [0.0]]),
            float(
                "inf"
            ),  # Expecting infinity or a very large number due to division by zero
            id="zero_wrong_class_prototypes",
        ),
        # 2D Tensor with uniform wrong class prototypes
        pytest.param(
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.tensor([[1.0], [1.0]]),
            2.5,
            id="2d_uniform_wrong_class_prototypes",
        ),
        # 3D Tensor with varying wrong class prototypes
        pytest.param(
            torch.tensor([[[10.0], [20.0]], [[30.0], [40.0]]]),
            torch.tensor([[2.0], [4.0]]),
            torch.mean(torch.tensor([5, 2.5, 10, 5, 15, 7.5, 20, 10])),
            id="3d_varying_wrong_class_prototypes",
        ),
    ],
)
def test_average_separation_cost(
    incorrect_class_prototype_activations, prototypes_of_wrong_class, expected_cost
):
    average_separation_cost = AverageSeparationCost()
    cost = average_separation_cost(
        incorrect_class_prototype_activations, prototypes_of_wrong_class
    )

    # Handling the infinity case separately as `torch.isclose` does not handle infinities.
    if expected_cost == float("inf"):
        assert cost.item() == float(
            "inf"
        ), f"Expected {expected_cost}, got {cost.item()}"
    else:
        assert torch.isclose(
            cost, torch.tensor(expected_cost)
        ), f"Expected {expected_cost}, got {cost.item()}"


def test_avg_separation_cost_1d():
    average_separation_cost = AverageSeparationCost()
    similarity_score_to_each_prototype = torch.tensor([0.1, 0.2, 0.3])
    prototypes_of_wrong_class = torch.tensor([0.1, 0.2, 0.3])

    with pytest.raises(AssertionError):
        average_separation_cost(
            similarity_score_to_each_prototype, prototypes_of_wrong_class
        )


def test_orthogonality_loss_0_for_ortho():
    """
    Evaluates whether, given orthogonal prototypes,
    our orthogonality loss is 0
    """
    backbone = construct_backbone("resnet18")

    num_classes = 3
    prototype_class_identity = torch.randn((num_classes * 2, 3))
    prototype_config = {
        "prototype_class_identity": prototype_class_identity,
        "num_classes": num_classes,
    }

    prototype_shape = (num_classes * 2, 512, 1, 1)

    proto_activation = CosPrototypeActivation()

    prototype_layer = PrototypeLayer(
        num_classes=num_classes,
        prototype_class_identity=prototype_class_identity,
        activation_function=proto_activation,
        prototype_dimension=(1, 1),
    )

    proto_vals = torch.zeros(prototype_shape)
    for k in range(prototype_shape[0]):
        proto_vals[k, k] = 1

    prototype_layer.prototype_tensors = torch.nn.Parameter(proto_vals)

    prediction_head = PrototypePredictionHead(**prototype_config)

    protopnet = ProtoPNet(
        backbone,
        torch.nn.Identity(),
        proto_activation,
        prototype_layer,
        prediction_head,
        warn_on_errors=True,
    )
    orthogonality_loss = OrthogonalityLoss()

    assert torch.isclose(orthogonality_loss(protopnet), torch.tensor(0.0))


def test_orthogonality_loss_1_for_colin():
    """
    Evaluates whether, given orthogonal prototypes,
    our orthogonality loss is 0
    """
    backbone = construct_backbone("resnet18")

    num_classes = 3
    prototype_class_identity = torch.randn((num_classes * 2, 3))
    prototype_config = {
        "prototype_class_identity": prototype_class_identity,
        "num_classes": num_classes,
    }

    prototype_shape = (num_classes * 2, 512, 3, 3)

    proto_activation = CosPrototypeActivation()

    prototype_layer = PrototypeLayer(
        num_classes=num_classes,
        prototype_class_identity=prototype_class_identity,
        activation_function=proto_activation,
        prototype_dimension=(3, 3),
    )

    proto_vals = torch.zeros(prototype_shape)
    for k in range(prototype_shape[0]):
        proto_vals[k, 0] = 1

    prototype_layer.prototype_tensors = torch.nn.Parameter(proto_vals)

    prediction_head = PrototypePredictionHead(**prototype_config)

    protopnet = ProtoPNet(
        backbone,
        torch.nn.Identity(),
        proto_activation,
        prototype_layer,
        prediction_head,
        warn_on_errors=True,
    )
    orthogonality_loss = OrthogonalityLoss()

    # We will have a n_proto_parts_per_class x n_proto_parts_per_class matrix
    n_proto_parts_per_class = (
        (prototype_shape[0] // num_classes) * prototype_shape[-1] * prototype_shape[-2]
    )

    target_per_class = torch.norm(
        torch.ones((n_proto_parts_per_class, n_proto_parts_per_class))
        - torch.eye(n_proto_parts_per_class)
    )
    target = target_per_class * num_classes

    assert torch.isclose(orthogonality_loss(protopnet), target)


@pytest.mark.parametrize(
    "type, target, upsampled_activation, fine_annotation, prototype_class_identity, expected_cost",
    [
        # Test no fine annotation (fa is all 0's) ie allowed to activate anywhere
        pytest.param(
            "serial",
            [0, 1, 2],  # target class id
            torch.ones([3, 6, 5, 5]),  # upsampled annotation
            torch.zeros([3, 1, 5, 5]),  # fine annotation
            one_hot(torch.tensor([0, 0, 1, 1, 2, 2])),  # prototype_class_identity
            torch.tensor(
                sum([math.sqrt(100), math.sqrt(50) + math.sqrt(50), math.sqrt(100)])
            ),
            id="annotation_on_everything_serial",
        ),
        # Test with white-out fine annotation (fa is all 1's) ie don't want it to active anywhere
        pytest.param(
            "serial",
            [0, 1, 2],  # target class id
            torch.ones([3, 6, 5, 5]),  # upsampled annotation
            torch.ones([3, 1, 5, 5]),  # fine annotation
            one_hot(torch.tensor([0, 0, 1, 1, 2, 2])),  # prototype_class_identity
            torch.tensor(
                sum(
                    [
                        math.sqrt(100) + math.sqrt(50),
                        math.sqrt(50) + math.sqrt(50) + math.sqrt(50),
                        math.sqrt(100) + math.sqrt(50),
                    ]
                )
            ),
            id="annotation_on_nothing_serial",
        ),
        # Test no fine annotation (fa is all 0's) ie allowed to activate anywhere
        pytest.param(
            "l2_norm",
            torch.tensor([0, 1, 2]),  # target class id
            torch.ones([3, 6, 5, 5]),  # upsampled annotation
            torch.zeros([3, 1, 5, 5]),  # fine annotation
            one_hot(torch.tensor([0, 0, 1, 1, 2, 2])),  # prototype_class_identity
            torch.tensor(60.0),
            id="annotation_on_everything_l2",
        ),
        # Test with white-out fine annotation (fa is all 1's) ie don't want it to active anywhere
        pytest.param(
            "l2_norm",
            torch.tensor([0, 1, 2]),  # target class id
            torch.ones([3, 6, 5, 5]),  # upsampled annotation
            torch.ones([3, 1, 5, 5]),  # fine annotation
            one_hot(torch.tensor([0, 0, 1, 1, 2, 2])),  # prototype_class_identity
            torch.tensor(90.0),
            id="annotation_on_nothing_l2",
        ),
        # Test no fine annotation (fa is all 0's) ie allowed to activate anywhere
        pytest.param(
            "square",
            torch.tensor([0, 1, 2]),  # target class id
            torch.ones([3, 6, 5, 5]),  # upsampled annotation
            torch.zeros([3, 1, 5, 5]),  # fine annotation
            one_hot(torch.tensor([0, 0, 1, 1, 2, 2])),  # prototype_class_identity
            torch.tensor(300.0),
            id="annotation_on_everything_square",
        ),
        # Test with white-out fine annotation (fa is all 1's) ie don't want it to active anywhere
        pytest.param(
            "square",
            torch.tensor([0, 1, 2]),  # target class id
            torch.ones([3, 6, 5, 5]),  # upsampled annotation
            torch.ones([3, 1, 5, 5]),  # fine annotation
            one_hot(torch.tensor([0, 0, 1, 1, 2, 2])),  # prototype_class_identity
            torch.tensor(450.0),
            id="annotation_on_nothing_square",
        ),
    ],
)
def test_fine_annotation_cost(
    type,
    target,
    upsampled_activation,
    fine_annotation,
    prototype_class_identity,
    expected_cost,
):
    fa_cost = FineAnnotationCost(fa_loss=type)
    cost = fa_cost(
        target, fine_annotation, upsampled_activation, prototype_class_identity
    )
    print(cost)
    assert torch.isclose(cost, expected_cost), f"Expected {expected_cost}, got {cost}"


class MockDataset:
    def __init__(self, labels):
        self.labels = labels


class MockDataLoader:
    def __init__(self, labels):
        self.dataset = MockDataset(labels)


@pytest.fixture
def dataloader():
    labels = [0, 0, 1, 1, 1, 2, 2, 2, 2]
    return MockDataLoader(labels)


def test_calculate_class_weights(dataloader):
    weights = calculate_class_weights(dataloader)

    # Manually calculated weights for the mock dataset
    # Total samples: 9
    # Class frequencies: [2, 3, 4]
    # Class weights: (total_samples / (num_classes * class_count))
    expected_weights = torch.tensor(
        [9 / (3 * 2), 9 / (3 * 3), 9 / (3 * 4)]  # 1.5  # 1.0  # 0.75
    )

    assert torch.equal(
        weights, expected_weights
    ), "Class weights are not calculated correctly"


@pytest.mark.parametrize("class_weights", [None, torch.tensor([1.0, 2.0, 3.0])])
def test_balanced_cross_entropy(dataloader, class_weights):
    logits = torch.tensor(
        [[0.2, 1.0, 0.1], [0.1, 0.3, 0.6], [0.5, 0.2, 0.3], [0.8, 0.1, 0.1]],
        dtype=torch.float32,
        requires_grad=True,
    )
    target = torch.tensor([0, 1, 2, 0], dtype=torch.long)

    criterion = BalancedCrossEntropyCost(class_weights=class_weights)

    loss = criterion(logits, target)

    assert loss.dim() == 0, "Loss should be a scalar value"

    assert torch.isfinite(loss).item(), "Loss should be a finite value"


def test_balanced_cross_entropy_no_preset_class_weights(dataloader):
    logits = torch.tensor(
        [[0.2, 1.0, 0.1], [0.1, 0.3, 0.6], [0.5, 0.2, 0.3], [0.8, 0.1, 0.1]],
        dtype=torch.float32,
        requires_grad=True,
    )
    target = torch.tensor([0, 1, 2, 0], dtype=torch.long)

    class_weights = calculate_class_weights(dataloader)

    criterion = BalancedCrossEntropyCost(class_weights=class_weights)

    loss = criterion(logits, target)

    assert loss.dim() == 0, "Loss should be a scalar value"
    assert torch.isfinite(loss).item(), "Loss should be a finite value"
    assert loss.item() > 0, "Loss should be a positive value"

    criterion_no_weights = BalancedCrossEntropyCost(class_weights=None)
    loss_no_weights = criterion_no_weights(logits, target)
    assert (
        loss_no_weights.item() != loss.item()
    ), "Loss with and without class weights should differ"
