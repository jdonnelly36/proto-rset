import copy
from typing import Any, Dict, List

import pytest
import torch
import torch.nn as nn

from protopnet.activations import CosPrototypeActivation
from protopnet.backbones import construct_backbone
from protopnet.datasets.torch_extensions import TensorDatasetDict
from protopnet.embedding import AddonLayers, EmbeddedBackbone
from protopnet.models.vanilla_protopnet import VanillaProtoPNet
from protopnet.train.metrics import TrainingMetrics
from protopnet.train.scheduling import ProtoPNetBackpropEpoch, TrainingSchedule
from protopnet.train.trainer import ProtoPNetTrainer


# Define a fixture for shared setup
@pytest.fixture
def setup():
    setup_data = {
        "batch_size": 1,
        "num_classes": 2,
        "num_prototypes_per_class": 2,
        "coefs": {
            "cluster": -0.8,
            "offset_weight_l2": 0.8,
            "separation": 0.8,
            "orthogonality_loss": 0.01,
            "offset_bias_l2": 0.8,
            "l1": 0.01,
            "cross_entropy": 2,
        },
        "device": "cpu",
    }

    cosine_activation_function = CosPrototypeActivation()

    data = torch.randn(2, 3, 224, 224)
    labels = torch.tensor([1, 0], dtype=torch.long)
    dataset = TensorDatasetDict(data, labels)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=setup_data["batch_size"], shuffle=False, num_workers=2
    )
    train_loader_batch_2 = torch.utils.data.DataLoader(
        dataset, batch_size=2 * setup_data["batch_size"], shuffle=False, num_workers=2
    )

    return {
        "setup_data": setup_data,
        "cosine_activation_function": cosine_activation_function,
        "train_loader": train_loader,
        "train_loader_batch_2": train_loader_batch_2,
    }


@pytest.mark.parametrize(
    "phase,schedule,layers_that_should_change",
    [
        pytest.param(
            "project",
            TrainingSchedule(
                num_warm_epochs=0,
                num_last_only_epochs=0,
                num_warm_pre_offset_epochs=0,
                num_joint_epochs=0,
                last_layer_fixed=True,
                project_epochs=[0],
                num_last_only_epochs_after_each_project=0,
            ),
            ["prototype_tensors"],
            id="project",
        ),
        pytest.param(
            "warm",
            TrainingSchedule(
                num_warm_epochs=1,
                num_last_only_epochs=0,
                num_warm_pre_offset_epochs=0,
                num_joint_epochs=0,
                last_layer_fixed=False,
                num_last_only_epochs_after_each_project=0,
            ),
            ["add_on_layers", "prototype_tensors", "head"],
            id="warm",
        ),
        pytest.param(
            "last_only",
            TrainingSchedule(
                num_warm_epochs=0,
                num_last_only_epochs=1,
                num_warm_pre_offset_epochs=0,
                num_joint_epochs=0,
                last_layer_fixed=False,
                num_last_only_epochs_after_each_project=0,
            ),
            ["head"],
            id="last_only",
        ),
        pytest.param(
            "warm_pre_offset",
            TrainingSchedule(
                num_warm_epochs=0,
                num_last_only_epochs=0,
                num_warm_pre_offset_epochs=1,
                num_joint_epochs=0,
                last_layer_fixed=False,
                num_last_only_epochs_after_each_project=0,
            ),
            ["backbone", "add_on_layers", "prototype_tensors", "head"],
            id="warm_pre_offset",
        ),
        pytest.param(
            "warm_no_last",
            TrainingSchedule(
                num_warm_epochs=1,
                num_last_only_epochs=0,
                num_warm_pre_offset_epochs=0,
                num_joint_epochs=0,
                last_layer_fixed=True,
                num_last_only_epochs_after_each_project=0,
            ),
            ["add_on_layers", "prototype_tensors"],
            id="warm_no_last",
        ),
        pytest.param(
            "warm_pre_offset_no_last",
            TrainingSchedule(
                num_warm_epochs=0,
                num_last_only_epochs=0,
                num_warm_pre_offset_epochs=1,
                num_joint_epochs=0,
                last_layer_fixed=True,
                num_last_only_epochs_after_each_project=0,
            ),
            ["backbone", "add_on_layers", "prototype_tensors"],
            id="warm_pre_offset_no_last",
        ),
        pytest.param(
            "joint",
            TrainingSchedule(
                num_warm_epochs=0,
                num_last_only_epochs=0,
                num_warm_pre_offset_epochs=0,
                num_joint_epochs=1,
                last_layer_fixed=False,
                num_last_only_epochs_after_each_project=0,
            ),
            ["backbone", "add_on_layers", "prototype_tensors", "conv_offset"],
            id="joint",
        ),
    ],
)
def test_vanilla_protopnet_training_phase(
    setup: Dict[str, Any],
    phase: str,
    schedule: TrainingSchedule,
    layers_that_should_change: List[str],
):
    setup_data = setup["setup_data"]
    train_loader = setup["train_loader"]

    cosine_activation_function = CosPrototypeActivation()

    vppn = VanillaProtoPNet(
        backbone=construct_backbone("resnet18"),
        add_on_layers=AddonLayers(
            setup_data["num_classes"] * setup_data["num_prototypes_per_class"]
        ),
        activation=cosine_activation_function,
        num_classes=setup_data["num_classes"],
        num_prototypes_per_class=setup_data["num_prototypes_per_class"],
    )

    optimizers_with_schedulers = {
        "warm": (
            torch.optim.Adam(vppn.parameters(), lr=0.00001, weight_decay=1e-3),
            None,
        ),
        "last_only": (
            torch.optim.Adam(vppn.parameters(), lr=0.00001, weight_decay=1e-3),
            None,
        ),
        "joint": (
            torch.optim.Adam(vppn.parameters(), lr=0.00001, weight_decay=1e-3),
            None,
        ),
        "warm_pre_offset": (
            torch.optim.Adam(vppn.parameters(), lr=0.00001, weight_decay=1e-3),
            None,
        ),
    }

    ppn_trainer = ProtoPNetTrainer(
        model=vppn,
        dataloader=train_loader,
        activation_function=setup["cosine_activation_function"],
        optimizers_with_schedulers=optimizers_with_schedulers,
        device=setup_data["device"],
        coefs=setup_data["coefs"],
        project_dataloader=train_loader,
        val_dataloader=train_loader,
        class_specific=True,
        training_metrics=TrainingMetrics(),
    )

    initial_state = copy.deepcopy(ppn_trainer.model.state_dict())

    ppn_trainer.train(schedule, val_each_epoch=False, save_model=False)

    # Compare the initial and final state of the model for the specific layers of interest
    for name, param in vppn.named_parameters():
        initial_weight = initial_state[name]
        final_weight = param.data
        should_update = any(
            layer_name in name for layer_name in layers_that_should_change
        )

        message_not_updated = (
            f"{name} should have been updated during {phase} phase but was not."
        )
        message_updated = (
            f"{name} should not have been updated during {phase} phase but was."
        )

        if should_update:
            assert not torch.equal(initial_weight, final_weight), message_not_updated
        else:
            assert torch.equal(initial_weight, final_weight), message_updated


@pytest.mark.parametrize(
    "phase,schedule,layers_that_should_change",
    [
        pytest.param(
            "project",
            TrainingSchedule(
                num_warm_epochs=0,
                num_last_only_epochs=0,
                num_warm_pre_offset_epochs=0,
                num_joint_epochs=0,
                last_layer_fixed=True,
                project_epochs=[0],
                num_last_only_epochs_after_each_project=0,
            ),
            ["prototype_tensors"],
            id="project",
        ),
        pytest.param(
            "warm",
            TrainingSchedule(
                num_warm_epochs=1,
                num_last_only_epochs=0,
                num_warm_pre_offset_epochs=0,
                num_joint_epochs=0,
                last_layer_fixed=False,
                num_last_only_epochs_after_each_project=0,
            ),
            ["add_on_layers", "prototype_tensors", "head"],
            id="warm",
        ),
        pytest.param(
            "last_only",
            TrainingSchedule(
                num_warm_epochs=0,
                num_last_only_epochs=1,
                num_warm_pre_offset_epochs=0,
                num_joint_epochs=0,
                last_layer_fixed=False,
                num_last_only_epochs_after_each_project=0,
            ),
            ["head"],
            id="last_only",
        ),
        pytest.param(
            "warm_pre_offset",
            TrainingSchedule(
                num_warm_epochs=0,
                num_last_only_epochs=0,
                num_warm_pre_offset_epochs=1,
                num_joint_epochs=0,
                last_layer_fixed=False,
                num_last_only_epochs_after_each_project=0,
            ),
            ["backbone", "add_on_layers", "prototype_tensors", "head"],
            id="warm_pre_offset",
        ),
        pytest.param(
            "warm_no_last",
            TrainingSchedule(
                num_warm_epochs=1,
                num_last_only_epochs=0,
                num_warm_pre_offset_epochs=0,
                num_joint_epochs=0,
                last_layer_fixed=True,
                num_last_only_epochs_after_each_project=0,
            ),
            ["add_on_layers", "prototype_tensors"],
            id="warm_no_last",
        ),
        pytest.param(
            "warm_pre_offset_no_last",
            TrainingSchedule(
                num_warm_epochs=0,
                num_last_only_epochs=0,
                num_warm_pre_offset_epochs=1,
                num_joint_epochs=0,
                last_layer_fixed=True,
                num_last_only_epochs_after_each_project=0,
            ),
            ["backbone", "add_on_layers", "prototype_tensors"],
            id="warm_pre_offset_no_last",
        ),
        pytest.param(
            "joint",
            TrainingSchedule(
                num_warm_epochs=0,
                num_last_only_epochs=0,
                num_warm_pre_offset_epochs=0,
                num_joint_epochs=1,
                last_layer_fixed=False,
                num_last_only_epochs_after_each_project=0,
            ),
            ["backbone", "add_on_layers", "prototype_tensors", "conv_offset"],
            id="joint",
        ),
    ],
)
def test_vanilla_protopnet_training_phase_batch_acc(
    setup: Dict[str, Any],
    phase: str,
    schedule: TrainingSchedule,
    layers_that_should_change: List[str],
):
    setup_data = setup["setup_data"]
    train_loader = setup["train_loader"]
    train_loader_batch_2 = setup["train_loader_batch_2"]

    # Using a dummy backbone because real backbones have things like
    # batch norm, which messes up this equivalence
    backbone = EmbeddedBackbone(
        torch.nn.Conv2d(in_channels=3, out_channels=512, kernel_size=(3, 3))
    )

    cosine_activation_function = CosPrototypeActivation()

    vppn = VanillaProtoPNet(
        backbone=backbone,
        add_on_layers=AddonLayers(
            setup_data["num_classes"] * setup_data["num_prototypes_per_class"]
        ),
        activation=cosine_activation_function,
        num_classes=setup_data["num_classes"],
        num_prototypes_per_class=setup_data["num_prototypes_per_class"],
    )

    initial_state = copy.deepcopy(vppn.state_dict())

    vppn_no_acc = copy.deepcopy(vppn)

    vppn_no_acc_optimizers_with_schedulers = {
        "warm": (
            torch.optim.Adam(vppn_no_acc.parameters(), lr=0.00001, weight_decay=1e-3),
            None,
        ),
        "last_only": (
            torch.optim.Adam(vppn_no_acc.parameters(), lr=0.00001, weight_decay=1e-3),
            None,
        ),
        "joint": (
            torch.optim.Adam(vppn_no_acc.parameters(), lr=0.00001, weight_decay=1e-3),
            None,
        ),
        "warm_pre_offset": (
            torch.optim.Adam(vppn_no_acc.parameters(), lr=0.00001, weight_decay=1e-3),
            None,
        ),
    }

    ppn_trainer_no_acc = ProtoPNetTrainer(
        model=vppn_no_acc,
        dataloader=train_loader_batch_2,
        activation_function=setup["cosine_activation_function"],
        optimizers_with_schedulers=vppn_no_acc_optimizers_with_schedulers,
        device=setup_data["device"],
        coefs=setup_data["coefs"],
        project_dataloader=train_loader_batch_2,
        val_dataloader=train_loader_batch_2,
        class_specific=True,
        num_accumulation_batches=1,
    )

    # Train without batch acc, record result
    assert torch.equal(
        ppn_trainer_no_acc.model.prototype_layer.prototype_tensors,
        vppn.prototype_layer.prototype_tensors,
    )
    ppn_trainer_no_acc.train(schedule, val_each_epoch=False, save_model=False)

    no_acc_state = copy.deepcopy(ppn_trainer_no_acc.model.state_dict())

    # Reinitialize with half the batch size, twice the acc
    vppn_acc = copy.deepcopy(vppn)
    vppn_acc_optimizers_with_schedulers = {
        "warm": (
            torch.optim.Adam(vppn_acc.parameters(), lr=0.00001, weight_decay=1e-3),
            None,
        ),
        "last_only": (
            torch.optim.Adam(vppn_acc.parameters(), lr=0.00001, weight_decay=1e-3),
            None,
        ),
        "joint": (
            torch.optim.Adam(vppn_acc.parameters(), lr=0.00001, weight_decay=1e-3),
            None,
        ),
        "warm_pre_offset": (
            torch.optim.Adam(vppn_acc.parameters(), lr=0.00001, weight_decay=1e-3),
            None,
        ),
    }
    ppn_trainer_acc = ProtoPNetTrainer(
        model=vppn_acc,
        dataloader=train_loader,
        activation_function=setup["cosine_activation_function"],
        optimizers_with_schedulers=vppn_acc_optimizers_with_schedulers,
        device=setup_data["device"],
        coefs=setup_data["coefs"],
        project_dataloader=train_loader,
        val_dataloader=train_loader,
        class_specific=True,
        num_accumulation_batches=2,
        training_metrics=TrainingMetrics(),
    )

    # Train with batch acc, record result
    assert torch.equal(
        ppn_trainer_acc.model.prototype_layer.prototype_tensors,
        vppn.prototype_layer.prototype_tensors,
    )
    ppn_trainer_acc.train(schedule, val_each_epoch=False, save_model=False)
    acc_state = copy.deepcopy(ppn_trainer_acc.model.state_dict())

    # Compare the initial and final state of the model for the specific layers of interest
    for name, param in vppn.named_parameters():
        initial_weight = initial_state[name]
        no_acc_weight = no_acc_state[name]
        acc_weight = acc_state[name]

        should_update = any(
            layer_name in name for layer_name in layers_that_should_change
        )

        message_not_updated = (
            f"{name} should have been updated during {phase} phase but was not."
        )
        message_not_updated_acc = f"{name} should have been updated during {phase} phase but was not with batch accumulation."
        message_updated = (
            f"{name} should not have been updated during {phase} phase but was."
        )
        message_unequal = (
            f"{name} is not the same after training with and without batch accumulation"
        )

        if should_update:
            # Check that weights updated as expected in both cases
            assert not torch.equal(initial_weight, no_acc_weight), message_not_updated
            assert not torch.equal(initial_weight, acc_weight), message_not_updated_acc
        else:
            assert torch.equal(initial_weight, no_acc_weight), message_updated
            assert torch.equal(initial_weight, acc_weight), message_updated

        # Check that both training strategies gave same res
        assert torch.allclose(acc_weight, no_acc_weight, atol=1e-6), message_unequal


def test_test_epochs(setup: Dict[str, Any]):
    setup_data = setup["setup_data"]
    train_loader = setup["train_loader"]

    vppn = VanillaProtoPNet(
        backbone=construct_backbone("resnet18"),
        add_on_layers=AddonLayers(
            setup_data["num_classes"] * setup_data["num_prototypes_per_class"]
        ),
        activation=setup["cosine_activation_function"],
        num_classes=setup_data["num_classes"],
        num_prototypes_per_class=setup_data["num_prototypes_per_class"],
    )

    optimizers_with_schedulers = {
        "warm": (
            torch.optim.Adam(vppn.parameters(), lr=0.00001, weight_decay=1e-3),
            None,
        ),
        "last_only": (
            torch.optim.Adam(vppn.parameters(), lr=0.00001, weight_decay=1e-3),
            None,
        ),
        "joint": (
            torch.optim.Adam(vppn.parameters(), lr=0.00001, weight_decay=1e-3),
            None,
        ),
        "warm_pre_offset": (
            torch.optim.Adam(vppn.parameters(), lr=0.00001, weight_decay=1e-3),
            None,
        ),
    }
    ppn_trainer = ProtoPNetTrainer(
        model=vppn,
        dataloader=train_loader,
        activation_function=setup["cosine_activation_function"],
        optimizers_with_schedulers=optimizers_with_schedulers,
        device=setup_data["device"],
        coefs=setup_data["coefs"],
        project_dataloader=train_loader,
        val_dataloader=train_loader,
        class_specific=True,
    )

    # Save copy of the model
    initial_state = copy.deepcopy(ppn_trainer.model.state_dict())

    # Run eval epoch
    ppn_trainer.eval_epoch(
        0,
        ProtoPNetBackpropEpoch(
            phase="last_only",
            train_backbone=False,
            train_add_on_layers=False,
            train_prototype_layer=False,
            train_conv_offset=False,
            train_prototype_prediction_head=False,
        ),
    )

    # Check that the model is still the same
    for name, param in vppn.named_parameters():
        initial_weight = initial_state[name]
        final_weight = param.data

        assert torch.equal(
            initial_weight, final_weight
        ), "Model should not have changed during evaluation epoch."


@pytest.mark.parametrize(
    "patience,reported_acc,completed_epochs",
    [
        (0, 0.5, 9),
        (1, 0.5, 14),
        (2, 0.5, 19),
        (3, 0.5, 25),
        (None, 0.005, 4),
        (100, 0.005, 4),
    ],
)
def test_early_stopping(
    setup: Dict[str, Any], patience: int, reported_acc: float, completed_epochs: int
):
    setup_data = setup["setup_data"]
    train_loader = setup["train_loader"]

    class OnesModel(nn.Module):
        def __init__(self, output_size):
            super(OnesModel, self).__init__()
            self.backbone = nn.Parameter(torch.ones(1))
            self.output_size = output_size

        def forward(self, x):
            # x is the input tensor. You can use its shape to determine the batch size if needed.
            batch_size = x.size(0)
            return torch.ones(batch_size, *self.output_size, requires_grad=False)

        def project(self, dataloader, class_specific):
            pass

    # Example usage:
    # Create a model instance with the desired output size. For example, (3, 224, 224) for a 3x224x224 tensor.
    model = OnesModel(output_size=(3, 224, 224))

    optimizers_with_schedulers = {
        "warm": (
            torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-3),
            None,
        ),
        "last_only": (
            torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-3),
            None,
        ),
        "joint": (
            torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-3),
            None,
        ),
        "warm_pre_offset": (
            torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-3),
            None,
        ),
    }

    ppn_trainer = ProtoPNetTrainer(
        model=model,
        dataloader=train_loader,
        activation_function=setup["cosine_activation_function"],
        optimizers_with_schedulers=optimizers_with_schedulers,
        device=setup_data["device"],
        coefs=setup_data["coefs"],
        project_dataloader=train_loader,
        val_dataloader=train_loader,
        class_specific=False,
        early_stopping_patience=patience,
        min_post_project_target_metric=0.01,
        target_metric_name="accu",
    )

    # noop
    def mock_epoch(*args, epoch_index=None, **kwargs):
        assert epoch_index is not None
        return reported_acc

    ppn_trainer.run_epoch = mock_epoch

    train_schedule = TrainingSchedule(
        num_warm_epochs=2,
        num_last_only_epochs_after_each_project=1,
        num_joint_epochs=20,
        project_epochs=[4, 7, 10, 13, 17, 20],  # Really 4, 9, 14, 19, 25, 30
    )
    assert len(train_schedule.get_epochs()) == 34, f"{train_schedule}"

    assert (
        ppn_trainer.train(train_schedule, val_each_epoch=True, save_model=False)
        == completed_epochs
    ), f"{train_schedule}"
