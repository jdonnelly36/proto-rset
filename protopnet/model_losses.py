import torch
import torch.nn as nn

from protopnet.losses import IncorrectClassPrototypeActivations
from protopnet.prototypical_part_model import ProtoPNet
from protopnet.utilities.trainer_utilities import init_or_update


class LinearBatchLoss(nn.Module):
    def __init__(self, batch_losses: list = [], device="cpu"):
        super(LinearBatchLoss, self).__init__()
        self.batch_losses = batch_losses
        self.device = device

    def required_forward_results(self):
        return {
            req
            for loss_component in self.batch_losses
            for req in loss_component.loss.required_forward_results
        }

    def forward(self, **kwargs):
        # Metrics dict comes from kwargs
        metrics_dict = kwargs.get("metrics_dict", {})

        # TODO: Set device to be same as model based variables
        total_loss = torch.tensor(0.0, device=self.device)

        for loss_component in self.batch_losses:
            # Get args for loss from just the loss_component.required_forward_results from kwargs
            current_loss_args = {
                req: kwargs[req] for req in loss_component.loss.required_forward_results
            }

            # assert loss_component is a float
            current_loss = (
                loss_component.loss(**current_loss_args) * loss_component.coefficient
            )

            init_or_update(metrics_dict, loss_component.loss.name, current_loss.item())
            total_loss += current_loss

        return total_loss


class LinearModelRegularization(nn.Module):
    def __init__(self, model_losses: list = [], device="cpu"):
        super(LinearModelRegularization, self).__init__()
        self.model_losses = model_losses
        self.device = device

    def forward(self, model: ProtoPNet, **kwargs):
        metrics_dict = kwargs.get("metrics_dict", {})

        # TODO: Set device to be same as model based variables
        total_loss = torch.tensor(0.0, device=self.device)  # Adjust device as needed

        for loss_component in self.model_losses:
            current_loss = loss_component.loss(model) * loss_component.coefficient

            metrics_dict[loss_component.loss.name] = current_loss.item()
            total_loss += current_loss

        return total_loss


class ProtoPNetLoss(nn.Module):
    def __init__(self, batch_losses, model_losses, debug_round, debug_loss, device="cpu"):
        super(ProtoPNetLoss, self).__init__()

        self.batch_loss = LinearBatchLoss(batch_losses, device)
        self.model_regularization = LinearModelRegularization(model_losses, device)

        self.batch_loss_required_forward_results = (
            self.batch_loss.required_forward_results()
        )

        self.incorrect_class_prototype_activations_fn = (
            IncorrectClassPrototypeActivations()
        )

        self.debug_round = debug_round

        if self.debug_round:
            self.debug_loss = LinearModelRegularization(debug_loss, device)

    def forward(
        self,
        target: torch.Tensor,
        fine_annotation: torch.Tensor,
        model: ProtoPNet,
        metrics_dict: dict,
        # logits: torch.Tensor,
        # similarity_score_to_each_prototype: torch.Tensor,
        # upsampled_activation: torch.Tensor,
        **kwargs,
    ):
        # TODO: Make sure grad is being calculated here if grad_req

        # TODO: Is there a better way to do this syntax?
        if (
            "prototypes_of_correct_class" in self.batch_loss_required_forward_results
            or "prototypes_of_wrong_class" in self.batch_loss_required_forward_results
        ):
            prototypes_of_correct_class = torch.t(
                model.get_prototype_class_identity(target)
            )
            prototypes_of_wrong_class = 1 - prototypes_of_correct_class
        else:
            prototypes_of_correct_class = None
            prototypes_of_wrong_class = None

        if (
            "incorrect_class_prototype_activations"
            in self.batch_loss_required_forward_results
        ):
            # Fail fast- should not occur
            if (
                "similarity_score_to_each_prototype"
                not in self.batch_loss_required_forward_results
            ):
                raise ValueError(
                    "similarity_score_to_each_prototype is required for incorrect_class_prototype_activations"
                )
            else:
                similarity_score_to_each_prototype = kwargs[
                    "similarity_score_to_each_prototype"
                ]

            incorrect_class_prototype_activations = self.incorrect_class_prototype_activations_fn(
                similarity_score_to_each_prototype=similarity_score_to_each_prototype,
                prototypes_of_wrong_class=prototypes_of_wrong_class,
            )

        if "prototype_class_identity" in self.batch_loss_required_forward_results:
            prototype_class_identity = model.prototype_class_identity
        else:
            prototype_class_identity = None

        # Pass in all arguments to batch_loss
        batch_loss = self.batch_loss(
            # pred=logits,
            target=target,
            fine_annotation=fine_annotation,
            # similarity_score_to_each_prototype=similarity_score_to_each_prototype,
            # upsampled_activation=upsampled_activation,
            prototype_class_identity=prototype_class_identity,
            prototypes_of_correct_class=prototypes_of_correct_class,
            prototypes_of_wrong_class=prototypes_of_wrong_class,
            incorrect_class_prototype_activations=incorrect_class_prototype_activations,
            metrics_dict=metrics_dict,
            **kwargs,
        )

        # batch_loss = self.batch_loss(pred, target, similarity_score_to_each_prototype, upsampled_activation, prototypes_of_correct_class, prototypes_of_wrong_class, metrics_dict)
        model_regularization = self.model_regularization(
            model, metrics_dict=metrics_dict
        )

        if self.debug_round:
            debug_losses = self.debug_loss(
                model,
                metrics_dict=metrics_dict
            )
            return batch_loss + model_regularization + debug_losses
        
        else:
            return batch_loss + model_regularization
