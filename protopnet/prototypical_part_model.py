import logging
import time

import torch
from torch import nn
from tqdm.auto import tqdm

from protopnet.prediction_heads import PrototypePredictionHead

logger = logging.getLogger(__name__)


class ProtoPNet(nn.Module):
    def __init__(
        self,
        backbone,
        add_on_layers,
        activation,
        prototype_layer,
        prototype_prediction_head,
        warn_on_errors: bool = False,
        k_for_topk: int = 1,
    ):
        super(ProtoPNet, self).__init__()

        self.backbone = backbone
        self.add_on_layers = add_on_layers
        self.activation = activation
        self.prototype_layer = prototype_layer
        self.prototype_prediction_head = prototype_prediction_head
        self.k_for_topk = k_for_topk

        self.__validate_model(warn_on_errors)

    def __validate_model(self, warn_on_errors: bool = False):
        """
        Validate the integretity of the model - namely, that the three layers are compatible with each other.
        """

        errors = []

        prototype_layer_latent_channels = self.prototype_layer.latent_channels

        if hasattr(self.add_on_layers, "proto_channels"):
            addon_latent_channels = self.add_on_layers.proto_channels
            if addon_latent_channels != prototype_layer_latent_channels:
                errors.append(
                    f"Backbone latent dimension {addon_latent_channels} does not match prototype layer latent dimension {prototype_layer_latent_channels}"
                )

        if getattr(self.prototype_layer, "update_prototypes_on_batch", None) is None:
            errors.append(
                "Prototype layer does not have a push method. This is required for."
            )

        if len(errors) == 0:
            logger.debug("Model validation passed.")
        elif warn_on_errors:
            for error in errors:
                logger.warning(error)
        else:
            for error in errors:
                logger.error(error)
            raise ValueError(
                f"Model validation failed with {len(errors)}. See log for details."
            )

    def get_prototype_complexity(self, decimal_precision=8):
        """
        Computes and returns metrics about how many unique prototypes,
        unique parts, etc the model has
        Args:
            decimal_precision: The number of decimal places up to which we consider for
                equality. I.e., if decimal_precision = 8, 1e-9 equals 2e-9, but 1e-7 != 2e-7
        """
        return self.prototype_layer.get_prototype_complexity(
            decimal_precision=decimal_precision
        )

    def forward(
        self,
        x: torch.Tensor,
        return_prototype_layer_output_dict: bool = False,
        **kwargs,
    ):
        latent_vectors = self.backbone(x)
        latent_vectors = self.add_on_layers(latent_vectors)

        prototype_layer_output_dict = self.prototype_layer(latent_vectors)

        prototype_similarities = prototype_layer_output_dict["prototype_activations"]
        upsampled_activation = prototype_layer_output_dict["upsampled_activation"]

        prediction_logits = self.prototype_prediction_head(
            prototype_similarities, upsampled_activation, **kwargs
        )

        if return_prototype_layer_output_dict:
            output_dict = prediction_logits.copy()
            output_dict.update(prototype_layer_output_dict.copy())
            return output_dict
        else:
            return prediction_logits

    def prune_duplicate_prototypes(self, decimal_precision=8) -> None:
        assert (
            type(self.prototype_prediction_head) is PrototypePredictionHead
        ), "Error: Pruning only supports linear last layer at the moment"

        visited_unique_prototypes = None
        visited_prototype_class_identities = None
        visited_prototype_last_layer_weight = None

        update_proto_dict = len(self.prototype_layer.prototype_info_dict) > 0
        updated_prototype_info_dict = {}

        new_ind_for_proto = 0
        for proto_ind in range(self.prototype_tensors().shape[0]):
            cur_proto = self.prototype_tensors()[proto_ind].unsqueeze(0)
            if visited_unique_prototypes is None:
                visited_unique_prototypes = cur_proto
                visited_prototype_class_identities = (
                    self.prototype_layer.prototype_class_identity[proto_ind].unsqueeze(
                        0
                    )
                )
                visited_prototype_last_layer_weight = (
                    self.prototype_prediction_head.class_connection_layer.weight.data[
                        :, proto_ind
                    ].unsqueeze(1)
                )

                if update_proto_dict:
                    updated_prototype_info_dict[new_ind_for_proto] = (
                        self.prototype_layer.prototype_info_dict[proto_ind]
                    )
                    new_ind_for_proto += 1
            else:
                equiv_protos = (
                    torch.isclose(visited_unique_prototypes, cur_proto)
                    .all(axis=1)
                    .all(axis=1)
                    .all(axis=1)
                )
                if equiv_protos.any():
                    target_equiv_proto = torch.argmax(equiv_protos * 1)
                    visited_prototype_last_layer_weight[
                        :, target_equiv_proto
                    ] += self.prototype_prediction_head.class_connection_layer.weight.data[
                        :, proto_ind
                    ]
                else:
                    visited_unique_prototypes = torch.cat(
                        [visited_unique_prototypes, cur_proto], dim=0
                    )
                    visited_prototype_class_identities = torch.cat(
                        [
                            visited_prototype_class_identities,
                            self.prototype_layer.prototype_class_identity[
                                proto_ind
                            ].unsqueeze(0),
                        ],
                        dim=0,
                    )
                    visited_prototype_last_layer_weight = torch.cat(
                        [
                            visited_prototype_last_layer_weight,
                            self.prototype_prediction_head.class_connection_layer.weight.data[
                                :, proto_ind
                            ].unsqueeze(
                                1
                            ),
                        ],
                        dim=1,
                    )

                    if update_proto_dict:
                        updated_prototype_info_dict[new_ind_for_proto] = (
                            self.prototype_layer.prototype_info_dict[proto_ind]
                        )
                        new_ind_for_proto += 1

        logger.info(
            f"Pruning from {self.prototype_tensors().shape[0]} prototypes to {visited_unique_prototypes.shape[0]}"
        )
        self.prototype_layer.prototype_tensors = torch.nn.Parameter(
            visited_unique_prototypes
        )
        self.prototype_layer.prototype_class_identity = (
            visited_prototype_class_identities
        )
        new_last_layer = torch.nn.Linear(
            visited_unique_prototypes.shape[0],
            self.prototype_layer.num_classes,
            bias=False,
        ).to(self.prototype_layer.prototype_tensors.device)
        new_last_layer.weight.data.copy_(visited_prototype_last_layer_weight)
        self.prototype_prediction_head.class_connection_layer = new_last_layer
        self.prototype_prediction_head.prototype_class_identity = (
            visited_prototype_class_identities
        )
        self.prototype_layer.num_prototypes = visited_unique_prototypes.shape[0]

        if update_proto_dict:
            self.prototype_layer.prototype_info_dict = updated_prototype_info_dict

    def prune_prototype(self, target : int) -> None:
        assert (
            type(self.prototype_prediction_head) is PrototypePredictionHead
        ), "Error: Pruning only supports linear last layer at the moment"

        update_proto_dict = len(self.prototype_layer.prototype_info_dict) > 0
        updated_prototype_info_dict = {}

        if target < self.prototype_tensors().shape[0] - 1:
            visited_unique_prototypes = torch.cat(
                [self.prototype_tensors()[:target], self.prototype_tensors()[target + 1:]],
                dim=0
            )
            visited_prototype_class_identities = torch.cat(
                [
                    self.prototype_layer.prototype_class_identity[:target],
                    self.prototype_layer.prototype_class_identity[target + 1:],
                ],
                dim=0,
            )
            visited_prototype_last_layer_weight = torch.cat(
                [
                    self.prototype_prediction_head.class_connection_layer.weight.data[
                        :, :target
                    ],
                    self.prototype_prediction_head.class_connection_layer.weight.data[
                        :, target + 1:
                    ],
                ],
                dim=1,
            )
            if update_proto_dict:
                for k in list(self.prototype_layer.prototype_info_dict.keys())[:target]:
                    updated_prototype_info_dict[k] = self.prototype_layer.prototype_info_dict[k]
                
                for k in list(self.prototype_layer.prototype_info_dict.keys())[target + 1:]:
                    updated_prototype_info_dict[k - 1] = self.prototype_layer.prototype_info_dict[k]
        else:
            visited_unique_prototypes = self.prototype_tensors()[:target]
            visited_prototype_class_identities = self.prototype_layer.prototype_class_identity[:target]
            visited_prototype_last_layer_weight = self.prototype_prediction_head.class_connection_layer.weight.data[
                :, :target
            ]

            if update_proto_dict:
                for k in list(self.prototype_layer.prototype_info_dict.keys())[:target]:
                    updated_prototype_info_dict[k] = self.prototype_layer.prototype_info_dict[k]


        logger.info(
            f"Pruning from {self.prototype_tensors().shape[0]} prototypes to {visited_unique_prototypes.shape[0]}"
        )
        self.prototype_layer.prototype_tensors = torch.nn.Parameter(
            visited_unique_prototypes
        )
        self.prototype_layer.prototype_class_identity = (
            visited_prototype_class_identities
        )
        new_last_layer = torch.nn.Linear(
            visited_unique_prototypes.shape[0],
            self.prototype_layer.num_classes,
            bias=False,
        ).to(self.prototype_layer.prototype_tensors.device)
        new_last_layer.weight.data.copy_(visited_prototype_last_layer_weight)
        self.prototype_prediction_head.class_connection_layer = new_last_layer
        self.prototype_prediction_head.prototype_class_identity = (
            visited_prototype_class_identities
        )
        self.prototype_layer.num_prototypes = visited_unique_prototypes.shape[0]

        if update_proto_dict:
            self.prototype_layer.prototype_info_dict = updated_prototype_info_dict

    def project(
        self, dataloader: torch.utils.data.DataLoader, class_specific=False
    ) -> None:
        logger.info("projecting prototypes onto %s", dataloader)
        state_before_push = self.training
        self.eval()
        start = time.time()

        # TODO: RENAME THIS
        n_prototypes = self.prototype_layer.num_prototypes

        global_max_proto_act = torch.full((n_prototypes,), -float("inf"))
        global_max_fmap_patches = torch.zeros_like(
            self.prototype_layer.prototype_tensors
        )

        search_batch_size = dataloader.batch_size

        logger.debug("initiating project batches")

        for push_iter, batch_data_dict in enumerate(tqdm(dataloader, desc="Prototype projection")):
            # TODO: ADD TQDM OPTIONALITY TO THIS LOOP
            logger.debug("starting project batch")
            search_batch_input = batch_data_dict["img"]
            search_y = batch_data_dict["target"]
            try:
                sample_ids = batch_data_dict["sample_id"]
            except KeyError:
                sample_ids = None

            start_index_of_search_batch = push_iter * search_batch_size

            search_batch_input = search_batch_input.to(
                self.prototype_layer.prototype_tensors.device
            )

            logger.debug("updating current best prototypes")
            self.prototype_layer.update_prototypes_on_batch(
                self.add_on_layers(self.backbone(search_batch_input)),
                start_index_of_search_batch,
                global_max_proto_act,
                global_max_fmap_patches,
                sample_ids,
                search_y,
                class_specific,
            )
            logger.debug("project batch complete")

        self.prototype_layer.set_prototype_tensors(global_max_fmap_patches)

        end = time.time()
        logger.info("\tpush time: \t{0}".format(end - start))
        self.train(state_before_push)

    def prototype_tensors(self) -> torch.Tensor:
        return self.prototype_layer.prototype_tensors.data

    def get_prototype_class_identity(self, label) -> torch.Tensor:
        return self.prototype_layer.prototype_class_identity[:, label]

    def add_additional_prototype(
        self, proto_vectors: torch.Tensor, proto_cc_vectors: torch.Tensor
    ) -> None:
        """
        Args:
            proto_vectors (torch.Tensor): Tensor containing new prototype vectors to be added to the prototype layer.
                                        Shape: (num_new_prototypes, feature_dim)
            proto_cc_vectors (torch.Tensor): Tensor containing new class connection vectors that will be appended to the
                                            final layer's weight matrix. These represent connections between the new prototypes
                                            and the target classes. Shape: (num_new_prototypes, num_classes)

        Returns:
            None
        """

        # add new proto vectors
        self.prototype_layer.prototype_tensors = nn.Parameter(
            data=torch.cat(
                [self.prototype_layer.prototype_tensors, proto_vectors], dim=0
            ),
            requires_grad=True,
        )

        self.prototype_layer.num_prototypes += proto_vectors.shape[0]

        # modify last layer weight
        assert isinstance(
            self.prototype_prediction_head.class_connection_layer, nn.Linear
        )
        old_last_layer = (
            self.prototype_prediction_head.class_connection_layer.weight.data
        )  # num_classes by num_protos
        new_final_layer = nn.Linear(
            old_last_layer.shape[1] + proto_cc_vectors.shape[0],
            old_last_layer.shape[0],
            bias=False,
        ).to(self.prototype_layer.prototype_class_identity.device)
        # catting torch.Size([200, 2000]) and torch.Size([200, 1]) on dim 1
        new_final_layer.weight.data = torch.cat(
            [old_last_layer, proto_cc_vectors.T], dim=-1
        )
        self.prototype_prediction_head.class_connection_layer = new_final_layer

        # add new class identity
        # NOTE: are we assuming class specific here? (whats the behavior of identity when class agnostic)
        #       and we are using argmax of class connection as class assignment, should we just pass in the labels?

        new_class_identity = torch.zeros(
            (proto_cc_vectors.shape[0], proto_cc_vectors.shape[1]),
            device=self.prototype_layer.prototype_class_identity.device,
        )
        for row_idx, row in enumerate(proto_cc_vectors):
            class_argmax = torch.argmax(row).item()
            new_class_identity[row_idx, class_argmax] = 1

        self.prototype_prediction_head.prototype_class_identity = torch.cat(
            [
                self.prototype_prediction_head.prototype_class_identity.to(self.prototype_layer.prototype_class_identity.device),
                new_class_identity,
            ],
            dim=0,
        )
        self.prototype_layer.prototype_class_identity = torch.cat(
            [
                self.prototype_layer.prototype_class_identity,
                new_class_identity,
            ],
            dim=0,
        )

    def input_channels(self) -> torch.Tensor:
        """
        Returns: The number of input channels to the model
        """
        return self.backbone.input_channels

    def describe_prototypes(self):
        # Resturn string describing the prototypes
        ret_str = ""
        for proto_index, proto_info in self.prototype_layer.prototype_info_dict.items():
            class_connection_vector = (
                self.prototype_prediction_head.class_connection_layer.weight[
                    :, proto_index
                ]
            )
            closest_class = torch.argmax(class_connection_vector)
            ret_str += f"\nPrototype {proto_index} comes from sample {proto_info.sample_id}.\n\tIt has highest class connection to class {closest_class} with a class connection vector of:\n\t\t{class_connection_vector}"
        return ret_str
