import torch
import torch.nn as nn


class PrototypePredictionHead(nn.Module):
    def __init__(
        self,
        num_classes: int,
        prototype_class_identity: torch.Tensor,
        incorrect_class_connection: float = 0,
        k_for_topk: int = 1,
    ):
        super(PrototypePredictionHead, self).__init__()

        self.num_classes = num_classes
        self.incorrect_class_connection = incorrect_class_connection
        self.k_for_topk = k_for_topk
        self.prototype_class_identity = prototype_class_identity

        self.num_prototypes = prototype_class_identity.shape[0]
        self.class_connection_layer = nn.Linear(
            self.num_prototypes,
            self.num_classes,
            bias=False,
        )

        self.__set_last_layer_incorrect_connection()

    def __set_last_layer_incorrect_connection(self):
        """
        the incorrect strength will be actual strength if -0.5 then input -0.5
        """

        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = self.incorrect_class_connection
        self.class_connection_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations
        )

    def forward(
        self,
        prototype_activations: torch.Tensor,
        upsampled_activation: torch.Tensor = None,
        **kwargs,
    ):
        # TODO: Update prototype_activations to be

        _activations = prototype_activations.view(
            prototype_activations.shape[0], prototype_activations.shape[1], -1
        )

        # When k=1, this reduces to the maximum
        k_for_topk = min(self.k_for_topk, _activations.shape[-1])
        topk_activations, _ = torch.topk(_activations, k_for_topk, dim=-1)
        similarity_score_to_each_prototype = torch.mean(topk_activations, dim=-1)

        logits = self.class_connection_layer(similarity_score_to_each_prototype)

        # output_dict = {
        #     "logits": logits,
        #     "similarity_score_to_each_prototype": similarity_score_to_each_prototype,
        #     "upsampled_activation": upsampled_activation,
        # }

        output_dict = {"logits": logits}

        if (
            "return_similarity_score_to_each_prototype" in kwargs
            and kwargs["return_similarity_score_to_each_prototype"]
        ) or (
            "return_incorrect_class_prototype_activations" in kwargs
            and kwargs["return_incorrect_class_prototype_activations"]
        ):
            output_dict[
                "similarity_score_to_each_prototype"
            ] = similarity_score_to_each_prototype

        if (
            "return_upsampled_activation" in kwargs
            and kwargs["return_upsampled_activation"]
        ):
            output_dict["upsampled_activation"] = upsampled_activation

        return output_dict
