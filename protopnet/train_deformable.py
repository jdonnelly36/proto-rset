import logging
import warnings

import torch

import wandb

from . import datasets
from .activations import CosPrototypeActivation
from .backbones import construct_backbone
from .embedding import AddonLayers
from .prediction_heads import PrototypePredictionHead
from .prototype_layers import DeformablePrototypeLayer
from .prototypical_part_model import ProtoPNet
from .train.logging.weights_and_biases import WeightsAndBiasesTrainLogger
from .train.metrics import InterpretableTrainingMetrics
from .train.scheduling import TrainingSchedule
from .train.trainer import ProtoPNetTrainer

logger = logging.getLogger(__name__)


def run(
    *,
    backbone="vgg16",
    pre_project_phase_len=5,
    num_warm_pre_offset_epochs=5,
    phase_multiplier=30,  # for online augmentation
    latent_dim_multiplier_exp=-2,
    joint_lr_step_size=5,
    post_project_phases=10,
    joint_epochs_per_phase=10,
    last_only_epochs_per_phase=20,
    cluster_coef=-0.8,
    separation_coef=0.08,
    l1_coef=0.01,
    num_addon_layers=0,
    fa_type=None,
    fa_coef=0.001,
    num_prototypes_per_class=10,
    offset_weight_l2=0.8,
    orthogonality_loss=0.01,
    offset_bias_l2=0.8,
    cross_entropy=1,
    k_for_topk=1,
    joint_add_on_lr_multiplier=1,
    warm_lr_multiplier=1,
    lr_multiplier=1,
    prototype_dimension=(3, 3),
    class_specific=False,
    dry_run=False,
    verify=False,
    interpretable_metrics=False,
    dataset="CUB200",
):
    """
    Train a Vanilla ProtoPNet.

    Args:
    - backbone: str - See backbones.py
    - pre_project_phase_len: int - number of epochs in each pre-project phase (warm-up, joint). Total preproject epochs is 2*pre_project_phase_len*phase_multiplier.
    - phase_multiplier: int - for each phase, multiply the number of epochs in that phase by this number
    - latent_dim_exp: int - expotential of 2 for the latent dimension of the prototype layer
    - joint_lr_step_size: int - number of epochs between each step in the joint learning rate scheduler. Multiplied by phase_multiplier.
    - last_only_epochs_per_phase: int - coefficient for clustering loss
    - post_project_phases: int - number of times to iterate between last-only, joint, project after the initial pre-project phases
    - cluster_coef: float - coefficient for clustering loss term
    - separation_coef: float - coefficient for separation loss term
    - l1_coef: float - coefficient for clustering loss
    - fa_type: str - one of "serial", "l2", or "square" to indicate which type of fine annotation loss to use. if None, fine annotation is deactivated.
    - fa_coef: float - coefficient for fine annotation loss term
    - num_prototypes_per_class: int - number of prototypes to create for each class
    - lr_multiplier: float - multiplier for learning rates. The base values are from protopnet's training.
    - class_specific: boolean - whether to bind prototypes to individual classes or allow them to be distributed unevenly based on training
    - dry_run: bool - Configure the training run, but do not execute it
    - preflight: bool - Configure a training run for a single epoch of all phases
    - verify: bool - Configure a training run for a single epoch of all phases
    - interpretable_metrics: bool - Whether to calculate interpretable metrics
    """

    # TODO: this should be controlled elsewhere
    if wandb.run is None:
        wandb.init(project="test")

    if verify:
        logger.info("Setting preflight configuration to all 1s")
        pre_project_phase_len = 1
        post_project_phases = 1
        joint_epochs_per_phase = 1
        last_only_epochs_per_phase = 1
        phase_multiplier = 1
        num_prototypes_per_class = 16

        prototype_dimension = (1, 1)

    if fa_type is not None and fa_coef == 0:
        warnings.warn("Run set up to use Fine Annotations, but fa_coef set to 0.")
    elif fa_type is None and fa_coef != 0:
        warnings.warn(
            f"Run set up to not use Fine Annotations, but fa_coef set to {fa_coef}."
        )

    setup = {
        "coefs": {
            "cluster": cluster_coef,
            "offset_weight_l2": offset_weight_l2,
            "separation": separation_coef,
            "orthogonality_loss": orthogonality_loss,
            "offset_bias_l2": offset_bias_l2,
            "l1": l1_coef,
            "fa": fa_coef,
            "cross_entropy": cross_entropy,
        },
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "train_log_filename": "train_log.txt",
    }
    if "dense" in backbone:
        num_accumulation_batches = 5
        batch_sizes = {"train": 95 // 5, "project": 75 // 5, "val": 100 // 5}
    elif "convnext" in backbone:
        num_accumulation_batches = 3
        batch_sizes = {"train": 95 // 3, "project": 75 // 3, "val": 100 // 3}
    else:
        num_accumulation_batches = 1
        batch_sizes = {"train": 95, "project": 75, "val": 100}

    logger.debug("initializing datasets")
    split_dataloaders = datasets.training_dataloaders(dataset, batch_sizes=batch_sizes)
    logger.debug("datasets initialized")

    if type(prototype_dimension) is int:
        prototype_dimension = (prototype_dimension, prototype_dimension)

    # TODO: Share this logic with ProtoPNet
    num_warm_epochs = pre_project_phase_len * phase_multiplier
    # accounting for warm and joint
    total_pre_project = 2 * (pre_project_phase_len) * phase_multiplier

    true_last_only_epochs_per_phase = last_only_epochs_per_phase * phase_multiplier
    num_post_project_backprop_epochs = (
        post_project_phases
        * (joint_epochs_per_phase + last_only_epochs_per_phase)
        * phase_multiplier
    )
    num_joint_epochs = joint_epochs_per_phase * post_project_phases * phase_multiplier
    # NOTE: the last-only epochs are added by the training schedule, so this schedule is just the joint between projects
    joint_between_project = joint_epochs_per_phase * phase_multiplier
    project_epochs = [
        e
        for e in range(
            total_pre_project,
            num_post_project_backprop_epochs + total_pre_project - 1,
            joint_between_project,
        )
    ]

    schedule = TrainingSchedule(
        num_warm_epochs=num_warm_epochs,
        num_last_only_epochs=0,
        num_warm_pre_offset_epochs=num_warm_pre_offset_epochs,
        num_joint_epochs=num_joint_epochs,
        max_epochs=num_post_project_backprop_epochs + total_pre_project,
        last_layer_fixed=False,
        project_epochs=project_epochs,
        num_last_only_epochs_after_each_project=true_last_only_epochs_per_phase,
    )

    cosine_activation_function = CosPrototypeActivation()
    num_prototypes = split_dataloaders.num_classes * num_prototypes_per_class
    prototype_class_identity = torch.zeros(
        num_prototypes, split_dataloaders.num_classes
    )

    for j in range(num_prototypes):
        prototype_class_identity[j, j // num_prototypes_per_class] = 1

    backbone = construct_backbone(backbone)

    prototype_config = {
        "k_for_topk": k_for_topk,
        "num_classes": split_dataloaders.num_classes,
        "prototype_class_identity": prototype_class_identity,
    }
    prediction_head = PrototypePredictionHead(**prototype_config)

    add_on_layers = AddonLayers(
        num_prototypes=num_prototypes_per_class * split_dataloaders.num_classes,
        input_channels=backbone.latent_dimension[0],
        proto_channel_multiplier=2**latent_dim_multiplier_exp,
        num_addon_layers=num_addon_layers,
    )

    deformable_prototypes = DeformablePrototypeLayer(
        num_classes=split_dataloaders.num_classes,
        prototype_class_identity=prototype_class_identity,
        prototype_dimension=prototype_dimension,
        latent_channels=add_on_layers.proto_channels,
    )

    vppn = ProtoPNet(
        backbone=backbone,
        add_on_layers=add_on_layers,
        activation=cosine_activation_function,
        prototype_layer=deformable_prototypes,
        prototype_prediction_head=prediction_head,
    )

    vppn = vppn.to(setup["device"])

    warm_optimizer_lrs = {
        "prototype_tensors": 0.003 * warm_lr_multiplier * lr_multiplier,
        "add_on_layers": 0.00 * warm_lr_multiplier * lr_multiplier,
    }

    warm_pre_offset_optimizer_lrs = {
        "joint_last_layer_lr": 0.0001 * joint_add_on_lr_multiplier * lr_multiplier,
        "prototype_tensors": 0.003 * joint_add_on_lr_multiplier * lr_multiplier,
        "features": 0.0001 * joint_add_on_lr_multiplier * lr_multiplier,
        "add_on_layers": 0.003 * joint_add_on_lr_multiplier * lr_multiplier,
    }

    joint_optimizer_lrs = {
        "joint_last_layer_lr": 0.0001 * joint_add_on_lr_multiplier * lr_multiplier,
        "prototype_tensors": 0.003 * joint_add_on_lr_multiplier * lr_multiplier,
        "conv_offset": 0.0001 * joint_add_on_lr_multiplier * lr_multiplier,
        "features": 0.0001 * joint_add_on_lr_multiplier * lr_multiplier,
        "add_on_layers": 0.003 * joint_add_on_lr_multiplier * lr_multiplier,
    }

    warm_optimizer_specs = [
        {
            "params": vppn.prototype_layer.prototype_tensors,
            "lr": warm_optimizer_lrs["prototype_tensors"],
        },
    ]
    warm_pre_offset_optimizer_specs = [
        {
            "params": vppn.backbone.parameters(),
            "lr": warm_pre_offset_optimizer_lrs["features"],
            "weight_decay": 1e-3,
        },  # bias are now also being regularized
        {
            "params": vppn.prototype_layer.prototype_tensors,
            "lr": warm_pre_offset_optimizer_lrs["prototype_tensors"],
        },
        {
            "params": vppn.prototype_prediction_head.class_connection_layer.parameters(),
            "lr": warm_pre_offset_optimizer_lrs["joint_last_layer_lr"],
        },
    ]
    joint_optimizer_specs = [
        {
            "params": vppn.backbone.parameters(),
            "lr": joint_optimizer_lrs["features"],
            "weight_decay": 1e-3,
        },  # bias are now also being regularized
        {
            "params": vppn.prototype_layer.offset_predictor.parameters(),
            "lr": joint_optimizer_lrs["prototype_tensors"],
        },
        {
            "params": vppn.prototype_layer.prototype_tensors,
            "lr": joint_optimizer_lrs["prototype_tensors"],
        },
        {
            "params": vppn.prototype_prediction_head.class_connection_layer.parameters(),
            "lr": joint_optimizer_lrs["joint_last_layer_lr"],
        },
    ]

    last_layer_optimizer_specs = [
        {
            "params": vppn.prototype_prediction_head.class_connection_layer.parameters(),
            "lr": 1e-4 * lr_multiplier,
        }
    ]

    warm_optimizer = torch.optim.Adam(warm_optimizer_specs)
    warm_pre_offset_optimizer = torch.optim.Adam(warm_pre_offset_optimizer_specs)
    joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
    last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

    # TODO: Make this step for each epoch
    # joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(
    #     joint_optimizer, step_size=joint_lr_step_size, gamma=0.2
    # )

    optimizers_with_schedulers = {
        "warm": (warm_optimizer, None),  # No scheduler for warm-up phase
        "joint": (
            joint_optimizer,
            torch.optim.lr_scheduler.StepLR(
                joint_optimizer,
                step_size=joint_lr_step_size * phase_multiplier,
                gamma=0.2,
            ),
        ),
        # Add the joint LR scheduler
        "last_only": (
            last_layer_optimizer,
            None,
        ),
        "warm_pre_offset": (
            warm_pre_offset_optimizer,
            None,
        ),
    }

    if interpretable_metrics:
        training_metrics = InterpretableTrainingMetrics(
            num_classes=split_dataloaders.num_classes,
            proto_per_class=num_prototypes_per_class,
            # FIXME: these shouldn't be hardcoded
            # instead, the train_dataloaders should return an object
            part_num=15,
            img_size=224,
            half_size=36,
            protopnet=vppn,
            device=setup["device"],
        )
        metric_args = {
            "training_metrics": training_metrics,
            "target_metric_name": "acc_proto_score",
            "min_post_project_target_metric": 0.0,
            "min_save_threshold": 0.10,
        }
        train_logger = WeightsAndBiasesTrainLogger(
            device=setup["device"], calculate_best_for=training_metrics.metric_names()
        )
    else:
        training_metrics = InterpretableTrainingMetrics(
            num_classes=split_dataloaders.num_classes,
            proto_per_class=num_prototypes_per_class,
            # FIXME: these shouldn't be hardcoded
            # instead, the train_dataloaders should return an object
            part_num=15,
            img_size=224,
            half_size=36,
            protopnet=vppn,
            device=setup["device"],
            acc_only=True,
        )
        metric_args = {
            "training_metrics": training_metrics,
            "target_metric_name": "accuracy",
            "min_post_project_target_metric": 0.01,
            "min_save_threshold": 0.66,
        }
        train_logger = WeightsAndBiasesTrainLogger(
            device=setup["device"], calculate_best_for=training_metrics.metric_names()
        )

    ppn_trainer = ProtoPNetTrainer(
        model=vppn,
        dataloader=split_dataloaders.train_loader,
        activation_function=cosine_activation_function,
        optimizers_with_schedulers=optimizers_with_schedulers,
        device=setup["device"],
        coefs=setup["coefs"],
        class_specific=class_specific,
        with_fa=fa_type is not None,
        fa_type=fa_type,
        project_dataloader=split_dataloaders.project_loader,
        val_dataloader=split_dataloaders.val_loader,
        early_stopping_patience=2,
        logger=train_logger,
        num_accumulation_batches=num_accumulation_batches,
        **metric_args,
    )

    if dry_run:
        logger.info("Skipping training due to dry run: %s", schedule)
    else:
        ppn_trainer.train(schedule)
