from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from .prototype_layers import ProtoPNet

from dataclasses import dataclass
from typing import Callable, Union


@dataclass
class LossTerm:
    loss: nn.Module
    coefficient: Union[Callable, float]


class IncorrectClassPrototypeActivations(nn.Module):
    def __init__(self):
        super(IncorrectClassPrototypeActivations, self).__init__()

    def forward(self, *, similarity_score_to_each_prototype, prototypes_of_wrong_class):
        incorrect_class_prototype_activations, _ = torch.max(
            similarity_score_to_each_prototype * prototypes_of_wrong_class, dim=1
        )

        return incorrect_class_prototype_activations


class CrossEntropyCost(nn.Module):
    def __init__(self):
        super(CrossEntropyCost, self).__init__()
        self.name = "cross_entropy"

        # TODO: Should these be functions or lists?
        self.required_forward_results = {"logits", "target"}

    def forward(self, logits: torch.Tensor, target: torch.Tensor, **kwargs):
        cross_entropy = torch.nn.functional.cross_entropy(logits, target)
        return cross_entropy


class BalancedCrossEntropyCost(nn.Module):
    def __init__(self, class_weights: torch.Tensor = None):
        super(BalancedCrossEntropyCost, self).__init__()
        self.name = "balanced_cross_entropy"
        self.class_weights = class_weights
        self.required_forward_results = {"logits", "target"}

    def forward(self, logits: torch.Tensor, target: torch.Tensor, **kwargs):
        cross_entropy = torch.nn.functional.cross_entropy(
            logits, target, weight=self.class_weights
        )
        return cross_entropy


class L1CostClassConnectionLayer(nn.Module):
    def __init__(self):
        super(L1CostClassConnectionLayer, self).__init__()
        self.name = "l1"

    def forward(self, model: "ProtoPNet"):
        l1 = model.prototype_prediction_head.class_connection_layer.weight.norm(p=1)
        return l1


class ClusterCost(nn.Module):
    def __init__(self, class_specific: bool = True):
        super(ClusterCost, self).__init__()
        self.class_specific = class_specific
        self.name = "cluster"

        self.required_forward_results = {
            "similarity_score_to_each_prototype",
            "prototypes_of_correct_class",
        }

    def forward(
        self, similarity_score_to_each_prototype, prototypes_of_correct_class=None
    ):
        # Raise Assertion if similarity_score_to_each_prototype, prototypes_of_correct_class is 1D
        assert similarity_score_to_each_prototype.dim() > 1 and (
            prototypes_of_correct_class is None or prototypes_of_correct_class.dim() > 1
        ), "Max activations or prototypes of correct class is 1D."

        if self.class_specific:
            assert (
                prototypes_of_correct_class is not None
            ), "Prototypes of correct class must be provided to calculate cluster cost."

            # correct_class_prototype_activations
            closest_sample_activations, _ = torch.max(
                similarity_score_to_each_prototype * prototypes_of_correct_class, dim=1
            )
        else:
            closest_sample_activations, _ = torch.max(
                similarity_score_to_each_prototype, dim=1
            )

        cluster_cost = torch.mean(closest_sample_activations)

        return cluster_cost


class SeparationCost(nn.Module):
    def __init__(self):
        super(SeparationCost, self).__init__()
        self.name = "separation"

        self.required_forward_results = {"incorrect_class_prototype_activations"}

    def forward(self, incorrect_class_prototype_activations):
        if incorrect_class_prototype_activations is None:
            raise ValueError(
                "Incorrect class prototype activations must be provided to calculate separation cost"
            )

        separation_cost = torch.mean(incorrect_class_prototype_activations)

        return separation_cost


class AverageSeparationCost(nn.Module):
    def __init__(self):
        super(AverageSeparationCost, self).__init__()
        self.name = "average_separation"

        self.required_forward_results = {
            "incorrect_class_prototype_activations",
            "prototypes_of_wrong_class",
        }

    def forward(
        self,
        incorrect_class_prototype_activations,
        prototypes_of_wrong_class=None,
    ):
        # Raise Assertion if prototypes_of_wrong_class is 1D
        assert prototypes_of_wrong_class.dim() > 1, "Prototypes of wrong class is 1D."

        if not (
            incorrect_class_prototype_activations is not None
            and prototypes_of_wrong_class is not None
        ):
            return None

        avg_separation_cost = incorrect_class_prototype_activations / torch.sum(
            prototypes_of_wrong_class, dim=1
        )

        avg_separation_cost = torch.mean(avg_separation_cost)

        return avg_separation_cost


class OffsetL2Cost(nn.Module):
    def __init__(self):
        super(OffsetL2Cost, self).__init__()

        self.name = "offset_l2"

    def forward(self, input_normalized: torch.Tensor, model: "ProtoPNet"):
        # Need to pass in input_normalized

        # TODO: Need conv_offset Sequential in model for this to work
        # offsets = model.module.conv_offset(input_normalized)
        offsets = torch.ones_like(input_normalized)

        offset_l2 = offsets.norm()
        return offset_l2


class OrthogonalityLoss(nn.Module):
    def __init__(self):
        super(OrthogonalityLoss, self).__init__()

        self.name = "orthogonality_loss"

    def forward(self, model: "ProtoPNet"):
        # Grab our prototype tensors, of shape (num_protos, channel, proto_h, proto_w)
        prototype_tensors = model.prototype_layer.prototype_tensors

        # Seperate prototypes out by class
        prototype_tensors = prototype_tensors.reshape(
            model.prototype_layer.num_prototypes_per_class,
            model.prototype_layer.num_classes,
            *prototype_tensors.shape[-3:],
        )

        # Permute and reshape these to (num_classes, protos_per_class*parts_per_proto, channel)
        prototype_tensors = prototype_tensors.permute(1, 0, 3, 4, 2).reshape(
            model.prototype_layer.num_classes, -1, prototype_tensors.shape[-3]
        )

        # Normalize each part to unit length
        prototype_tensors = F.normalize(prototype_tensors, p=2, dim=-1)

        # Get our (num_classes, protos_per_class*parts_per_proto, protos_per_class*parts_per_proto)
        # orthogonality matrix
        orthogonalities = torch.bmm(
            prototype_tensors, prototype_tensors.transpose(-2, -1)
        )

        # Subtract out the identity matrix
        orthogonalities = orthogonalities - torch.eye(
            orthogonalities.shape[-1], device=orthogonalities.device
        ).unsqueeze(0)

        # And compute our loss
        ortho_loss = torch.sum(torch.norm(orthogonalities, dim=(1, 2)))

        return ortho_loss


class SerialFineAnnotationCost(nn.Module):
    def __init__(self):
        super(SerialFineAnnotationCost, self).__init__()

    def forward(
        self,
        target: torch.Tensor,
        fine_annotation: torch.Tensor,
        upsampled_activation: torch.Tensor,
        prototype_class_identity: torch.Tensor,
        white_coef=None,
    ):
        prototype_targets = prototype_class_identity.argmax(dim=1)
        v, i = prototype_targets.sort()
        if (v != prototype_targets).all():
            raise NotImplementedError(
                "Do not use Serial Fine Annotation cost when prototypes are not grouped together."
            )
        _, class_counts = prototype_targets.unique(return_counts=True)
        unique_counts = class_counts.unique()
        if len(unique_counts) != 1:
            raise NotImplementedError(
                "Do not use Serial Fine Annotation cost when prototype classes are imbalanced."
            )

        proto_num_per_class = list(set(class_counts))[0]
        device = upsampled_activation.device

        all_white_mask = torch.ones(
            upsampled_activation.shape[2], upsampled_activation.shape[3]
        ).to(device)

        fine_annotation_cost = 0

        for index in range(target.shape[0]):
            weight1 = 1 * all_white_mask
            weight2 = 1 * fine_annotation[index]

            if white_coef is not None:
                weight1 *= white_coef

            fine_annotation_cost += (
                torch.norm(
                    upsampled_activation[index, : target[index] * proto_num_per_class]
                    * (weight1)
                )
                + torch.norm(
                    upsampled_activation[
                        index,
                        target[index]
                        * proto_num_per_class : (target[index] + 1)
                        * proto_num_per_class,
                    ]
                    * (weight2)
                )
                + torch.norm(
                    upsampled_activation[
                        index,
                        (target[index] + 1) * proto_num_per_class :,
                    ]
                    * (weight1)
                )
            )

        return fine_annotation_cost


class GenericFineAnnotationCost(nn.Module):
    def __init__(self, scoring_function):
        """
        Parameters:
        ----------
        scoring_function (function): Function for aggregating the loss costs. Will receive the masked activations.
        """
        super(GenericFineAnnotationCost, self).__init__()
        self.scoring_function = scoring_function

    def forward(
        self,
        target: torch.Tensor,
        fine_annotation: torch.Tensor,
        upsampled_activation: torch.Tensor,
        prototype_class_identity: torch.Tensor,
    ):
        """
        Calculates the fine-annotation loss for a given set of inputs.

        Parameters:
        ----------
            target (torch.Tensor): Tensor of targets. Size(Batch)
            upsampled_activation (torch.Tensor): Size(batch, n_prototypes, height, width)
            fine_annotation (torch.Tensor): Fine annotation tensor Size(batch, 1, height, width)
            prototype_class_identity (torch.Tensor): Class identity tensor for prototypes size(num_prototypes, num_classes)

        Returns:
        --------
            fine_annotation_loss (torch.Tensor): Fine annotation loss tensor

         Notes:
        -----
            This function assumes that the input tensors are properly aligned such that the prototype at index i
            in the `upsampled_activation` tensor corresponds to the class at index i in the `prototype_class_identity`
            tensor.

        Called in following files:
            - train_and_eval.py: l2_fine_annotation_loss(), square_fine_annotation_loss()

        """
        target_set = target.unique()
        class_fa_losses = torch.zeros(target_set.shape[0])

        # Assigned but never used in IAIA-BL
        # total_proto = upsampled_activation.shape[1]

        # unhot the one-hot encoding
        prototype_targets = prototype_class_identity.argmax(
            dim=1
        )  # shape: (n_prototype)

        # This shifts our iteration from O(n) to O(#targets)
        for target_val in list(target_set):
            # We have different calculations depending on whether or not the prototype
            # is in class or not, so we will find each group
            in_class_targets = target == target_val  # shape: (batch)
            in_class_prototypes = (
                prototype_targets == target_val
            )  # shape: (n_prototypes)

            # In Class case Size(D', p=y, 244, 244)
            prototype_activation_in_class = upsampled_activation[in_class_targets][
                :, in_class_prototypes, :, :
            ]
            # broadcast fine_annotation to prototypes in dim 1
            prototypes_activation_in_class_masked = (
                prototype_activation_in_class * fine_annotation[in_class_targets]
            )

            # Out of class case Size(batch, p!=y, 244, 244)
            prototype_activation_out_of_class = upsampled_activation[in_class_targets][
                :, ~in_class_prototypes, :, :
            ]

            # regroup after masking to parallelize, Size(batch, p, 244, 244)
            class_activations = torch.cat(
                (
                    prototypes_activation_in_class_masked,
                    prototype_activation_out_of_class,
                ),
                1,
            )

            # Size(D', p) - norms for all prototypes
            class_fa_for_all_prototypes = self.scoring_function(class_activations)

            class_fa_losses[target_val] = torch.sum(class_fa_for_all_prototypes)

        fine_annotation_loss = class_fa_losses.sum()
        return fine_annotation_loss


class FineAnnotationCost(nn.Module):
    def __init__(self, fa_loss: str = "serial"):
        super(FineAnnotationCost, self).__init__()

        self.fa_loss = fa_loss
        self.name = "fine_annotation"
        self.required_forward_results = {
            "target",
            "fine_annotation",
            "upsampled_activation",
            "prototype_class_identity",
        }

        # TODO: Could choose just one cost function here
        # And then determine necessary parameters as kwdict
        # Make it more generic
        self.serial_cost = SerialFineAnnotationCost()
        self.l2_fine_annotation_loss = GenericFineAnnotationCost(self.l2_scoring)
        self.square_fine_annotation_loss = GenericFineAnnotationCost(
            self.square_scoring
        )

        assert self.fa_loss in ["serial", "l2_norm", "square"]

    def forward(
        self,
        target: torch.Tensor,
        fine_annotation: torch.Tensor,
        upsampled_activation: torch.Tensor,
        prototype_class_identity: torch.Tensor,
    ):
        target = torch.tensor(target).int()
        if fine_annotation is None:
            fa_shape = upsampled_activation.shape
            fa_shape[1] = 1
            fine_annotation = torch.zero(fa_shape)
        if self.fa_loss == "serial":
            fine_annotation_cost = self.serial_cost(
                target, fine_annotation, upsampled_activation, prototype_class_identity
            )
        elif self.fa_loss == "l2_norm":
            fine_annotation_cost = self.l2_fine_annotation_loss(
                target,
                fine_annotation,
                upsampled_activation,
                prototype_class_identity,
            )
        elif self.fa_loss == "square":
            fine_annotation_cost = self.square_fine_annotation_loss(
                target,
                fine_annotation,
                upsampled_activation,
                prototype_class_identity,
            )

        return fine_annotation_cost

    def l2_scoring(self, activations):
        return activations.norm(p=2, dim=(2, 3))

    def square_scoring(self, activations):
        return activations.square().sum(dim=(2, 3))


# ProtoDebug Losses
class ProtoForbidLoss(nn.Module):
    def __init__(self, debug_forbid_loader, device):
        super(ProtoForbidLoss, self).__init__()
        self.name = "debug_forbid"
        self.debug_forbid_loader = debug_forbid_loader
        self.device = device

    def forward(self, model: "ProtoPNet"):
        overall_loss = 0
        num_protos = model.prototype_layer.prototype_class_identity.shape[0]
        for data, _ in self.debug_forbid_loader:
            data = data.to(self.device)
            out = model(data, return_prototype_layer_output_dict=True)
            sims = out["prototype_activations"] # bsz, num_protos, latent_w, latent_h
            max_sim = F.max_pool2d(sims,
                             kernel_size=(sims.size()[2],
                                          sims.size()[3])).view(-1, num_protos)
            overall_loss = overall_loss + torch.sum(max_sim)
            del out, data, sims, max_sim

        return overall_loss / len(self.debug_forbid_loader.dataset)
    

class ProtoRememberLoss(nn.Module):
    def __init__(self, debug_remember_loader, device):
        super(ProtoRememberLoss, self).__init__()
        self.name = "debug_remember"
        self.debug_remember_loader = debug_remember_loader
        self.device = device

    def forward(self, model: "ProtoPNet"):
        similarities = []
        num_protos = model.prototype_layer.prototype_class_identity.shape[0]
        for data, _ in self.debug_remember_loader:
            data = data.to(self.device)
            out = model(data, return_prototype_layer_output_dict=True)
            sims = out["prototype_activations"] # bsz, num_protos, latent_w, latent_h
            max_sim = F.max_pool2d(sims,
                             kernel_size=(sims.size()[2],
                                          sims.size()[3])).view(-1, num_protos)
            similarities.append(max_sim)
        similarities = torch.cat(similarities, dim=0)
        return -torch.mean(torch.sum(similarities, dim=1))

