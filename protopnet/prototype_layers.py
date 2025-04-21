from collections import namedtuple
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from protopnet.activations import ConvolutionalSharedOffsetPred, CosPrototypeActivation
from protopnet.utilities.project_utilities import custom_unravel_index, hash_func

# used to track where the prototypical part came from
prototype_meta = namedtuple("prototype_meta", ["sample_id", "sample_hash"])


class PrototypeLayer(nn.Module):
    def __init__(
        self,
        num_classes: int,
        activation_function: Callable,
        prototype_class_identity: torch.Tensor,
        latent_channels: int = 512,
        prototype_dimension: tuple = (1, 1),
        k_for_topk: int = 1,
    ):
        super(PrototypeLayer, self).__init__()
        self.num_classes = num_classes
        self.activation_function = activation_function
        self.latent_channels = latent_channels
        self.latent_spatial_size = None

        # TODO: REVIEW THAT THIS CONTAINS DESIRED METADATA
        self.prototype_info_dict = dict()  # add comment of this expected format

        self.with_fa = False

        self.num_prototypes = prototype_class_identity.shape[0]

        self.num_prototypes_per_class = self.num_prototypes // self.num_classes

        # TODO: Determine if this is the correct procedure for avoiding device problems (prototype_class_identity on cpu)
        # Could also just explicitly set it to the device when we set the model
        self.register_buffer("prototype_class_identity", prototype_class_identity)

        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // self.num_prototypes_per_class] = 1

        self.prototype_tensors = nn.Parameter(
            torch.rand(
                self.num_prototypes,
                latent_channels,
                *prototype_dimension,
                requires_grad=True,
            )
        )

        self.k_for_topk = k_for_topk

    def get_prototype_complexity(self, decimal_precision=8):
        """
        Computes and returns metrics about how many unique prototypes,
        unique parts, etc the model has
        Args:
            decimal_precision: The number of decimal places up to which we consider for
                equality. I.e., if decimal_precision = 8, 1e-9 equals 2e-9, but 1e-7 != 2e-7
        """
        # Reorganize so that we have a collection of prototype part vectors
        part_vectors = self.prototype_tensors.permute(0, 2, 3, 1).reshape(
            -1, self.prototype_tensors.shape[1]
        )
        n_unique_proto_parts = (
            torch.round(part_vectors, decimals=decimal_precision).unique(dim=0).shape[0]
        )

        # Repeat to get the number of unique prototype tensors
        stacked_proto_vectors = self.prototype_tensors.reshape(
            self.prototype_tensors.shape[0], -1
        )
        n_unique_protos = (
            torch.round(stacked_proto_vectors, decimals=decimal_precision)
            .unique(dim=0)
            .shape[0]
        )

        min_sparsity = self.num_classes * (
            1 + 1 / (self.latent_spatial_size[0] * self.latent_spatial_size[1])
        )
        prototype_sparsity = n_unique_protos + n_unique_proto_parts / (
            self.latent_spatial_size[0] * self.latent_spatial_size[1]
        )

        prototype_sparsity = min_sparsity / prototype_sparsity

        return {
            "n_unique_proto_parts": n_unique_proto_parts,
            "n_unique_protos": n_unique_protos,
            "prototype_sparsity": prototype_sparsity,
        }

    def forward(self, x: torch.Tensor):
        """
        Expects input size (batch_size, channel_dim, latent_height, latent_width)
        Provides a prototype similarity for each image at each location. This results in a tensor of shape
        (batch_size, num_prototypes, latent_height, latent_width)
        """
        prototype_activations = self.activation_function(x, self.prototype_tensors)

        if not hasattr(self, "latent_spatial_size") or self.latent_spatial_size is None:
            self.latent_spatial_size = (
                prototype_activations.shape[-2],
                prototype_activations.shape[-1],
            )

        # TODO: Add upsampled activation
        if self.with_fa:
            upsampled_activation = torch.nn.Upsample(
                size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False
            )(prototype_activations)
        else:
            upsampled_activation = None

        output_dict = {
            "prototype_activations": prototype_activations,
            "upsampled_activation": upsampled_activation,
        }

        return output_dict

    def set_prototype_tensors(self, new_prototype_tensors):
        prototype_update = torch.reshape(
            new_prototype_tensors,
            tuple(self.prototype_tensors.shape),
        )

        self.prototype_tensors.data.copy_(
            torch.tensor(prototype_update, dtype=torch.float32).to(
                self.prototype_tensors.device
            )
        )

    def update_prototypes_on_batch(
        self,
        protoL_input_torch,
        start_index_of_search_batch,
        global_max_proto_act,
        global_max_fmap_patches,
        sample_ids,
        search_y,
        class_specific,
    ):
        prototype_layer_stride = 1

        # Assuming data is on correct device; setup belongs in the trainer
        # TODO: ALL ON CUDA OR NOT
        proto_act_torch = self.forward(
            protoL_input_torch.to(self.prototype_tensors.device)
        )["prototype_activations"]

        # protoL_input_ = torch.clone(protoL_input_torch.detach().cpu())
        # proto_act_ = torch.clone(proto_act_torch.detach().cpu())

        # del protoL_input_torch, proto_act_torch

        if class_specific:
            # Index class_to_img_index dict with class number, return list of images
            class_to_img_index_dict = {key: [] for key in range(self.num_classes)}
            # img_y is the image's integer label
            for img_index, img_y in enumerate(search_y):
                img_label = img_y.item()
                class_to_img_index_dict[img_label].append(img_index)

        prototype_shape = self.prototype_tensors.shape

        for j in range(self.num_prototypes):
            class_index = j

            if class_specific:
                # target_class is the class of the class_specific prototype
                target_class = torch.argmax(
                    self.prototype_class_identity[class_index]
                ).item()
                # if there is not images of the target_class from this batch
                # we go on to the next prototype
                if len(class_to_img_index_dict[target_class]) == 0:
                    continue
                proto_act_j = proto_act_torch[class_to_img_index_dict[target_class]][
                    :, j, :, :
                ]
            else:
                # if it is not class specific, then we will search through
                # every example
                proto_act_j = proto_act_torch[:, j, :, :]
            batch_max_proto_act_j = torch.amax(proto_act_j)

            if batch_max_proto_act_j > global_max_proto_act[j]:
                batch_argmax_proto_act_j = list(
                    custom_unravel_index(
                        torch.argmax(proto_act_j, axis=None), proto_act_j.shape
                    )
                )
                if class_specific:
                    """
                    change the argmin index from the index among
                    images of the target class to the index in the entire search
                    batch
                    """
                    batch_argmax_proto_act_j[0] = class_to_img_index_dict[target_class][
                        batch_argmax_proto_act_j[0]
                    ]

                # retrieve the corresponding feature map patch
                img_index_in_batch = batch_argmax_proto_act_j[0]
                fmap_height_start_index = (
                    batch_argmax_proto_act_j[1] * prototype_layer_stride
                )
                fmap_width_start_index = (
                    batch_argmax_proto_act_j[2] * prototype_layer_stride
                )

                # TODO: REVISIT SHAPE INDEXING
                fmap_height_end_index = fmap_height_start_index + prototype_shape[-2]
                fmap_width_end_index = fmap_width_start_index + prototype_shape[-1]

                batch_max_fmap_patch_j = protoL_input_torch[
                    img_index_in_batch,
                    :,
                    fmap_height_start_index:fmap_height_end_index,
                    fmap_width_start_index:fmap_width_end_index,
                ]

                # TODO: CONSTRUCT DICTIONARY OUTSIDE THE LOOP ONCE
                # FIXME: We should enforce sample_id is not None
                if sample_ids is not None:
                    self.prototype_info_dict[j] = prototype_meta(
                        sample_ids[img_index_in_batch].numpy().tobytes(),
                        hash_func(protoL_input_torch[img_index_in_batch]),
                    )

                global_max_proto_act[j] = batch_max_proto_act_j
                global_max_fmap_patches[j] = batch_max_fmap_patch_j


class DeformablePrototypeLayer(PrototypeLayer):
    """
    Computes the activation for a deformable prototype as in
        https://arxiv.org/pdf/1801.07698.pdf, but with
        renormalization after deformation instead of
        norm-preserving interpolation.
    """

    def __init__(
        self,
        num_classes: int,
        prototype_class_identity: torch.Tensor,
        offset_predictor: nn.Module = None,
        latent_channels: int = 512,
        prototype_dimension: tuple = (1, 1),
        epsilon_val=1e-5,
        activation_function=CosPrototypeActivation(),
        prototype_dilation: int = 1,
    ):
        """
        Args:
            num_classes: The number of classes for this task
            prototype_class_identity: A onehot tensor indicating which prototypes
                correspond to which class
            offset_predictor: A function that takes as input the latent
                tensor x of shape (batch, channel, height, width) and produces
                a tensor of offsets of shape (batch, 2, proto_h, proto_w, height, width)
                (2 is for a x, y offset for each part at each location)
            latent_channels: The number of output channels from the backbone
            prototype_dimension: A (proto_h, proto_w) tuple indicating the size of
                the prototypes
            episilon_val: A small value to prevent division by zero.
            activation_function: The activation function to use
        """
        super(DeformablePrototypeLayer, self).__init__(
            num_classes=num_classes,
            activation_function=activation_function,
            prototype_class_identity=prototype_class_identity,
            latent_channels=latent_channels,
            prototype_dimension=prototype_dimension,
        )
        self.epsilon_val = epsilon_val
        if offset_predictor is None:
            self.offset_predictor = ConvolutionalSharedOffsetPred(
                prototype_shape=self.prototype_tensors.shape,
                input_feature_dim=latent_channels,
                prototype_dilation=prototype_dilation,
            )
        else:
            self.offset_predictor = offset_predictor
        self.activation_function = activation_function

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: The input tensor of shape (batch_size, feature_dim, latent_height, latent_width)

        Returns: activations (torch.Tensor): Tensor of the activations. This is of shape (batch_size, num_prototypes, activation_height, activation_width).
        """
        # offsets comes out as (batch, proto_h, proto_w, 2, height, width)
        offsets = self.offset_predictor(x)

        # View offsets as (batch, proto_h, proto_w, 2, height, width)
        offsets_reshaped = offsets.view(
            x.shape[0],
            self.prototype_tensors.shape[-2],
            self.prototype_tensors.shape[-1],
            2,
            x.shape[-2],
            x.shape[-1],
        )
        # Move our x,y offset dim to end
        offsets_reshaped = offsets_reshaped.permute(0, 1, 2, 4, 5, 3)

        # Figure out which locations are being sampled in normalized (-1, 1) space
        # Comes out as (batch, proto_h, proto_w, H, W, 2)
        sample_locs = self._offsets_to_sample_locs(offsets_reshaped)

        stacked_interp_x = []
        stacked_proto = []
        for proto_h in range(self.prototype_tensors.shape[-2]):
            for proto_w in range(self.prototype_tensors.shape[-1]):
                stacked_proto.append(self.prototype_tensors[:, :, proto_h, proto_w])
                stacked_interp_x.append(
                    F.grid_sample(
                        x, sample_locs[:, proto_h, proto_w], align_corners=True
                    )
                )

        stacked_interp_x = torch.cat(stacked_interp_x, dim=1)
        stacked_proto = torch.cat(stacked_proto, dim=1).unsqueeze(-1).unsqueeze(-1)

        activations = self.activation_function(stacked_interp_x, stacked_proto)

        if not hasattr(self, "latent_spatial_size") or self.latent_spatial_size is None:
            self.latent_spatial_size = (activations.shape[-2], activations.shape[-1])

        if self.with_fa:
            upsampled_activation = torch.nn.Upsample(
                size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False
            )(activations)
        else:
            upsampled_activation = None

        output_dict = {
            "prototype_activations": activations,
            "prototype_sample_location_map": sample_locs,
            "upsampled_activation": upsampled_activation,
        }
        return output_dict

    def _offsets_to_sample_locs(self, offsets):
        """
        Convert offsets relative to a center location to absolute coordinates,
        normalized between -1 and 1
        Args:
            offsets: A (batch_size, proto_h, proto_w, height, width, 2) tensor
                of offsets relative to the center location at each (height, width)

        Returns: sample_locs, a (batch, proto_h, proto_w, height, width, 2) tensor
            describing the location to compare each prototypical part to at each
            center location and image in the batch
        """
        # Assumes offsets are in unnormalized space, e.g. an offset of
        # 1 moves us by 1 latent cell
        second_last_dim_inits = (
            torch.arange(offsets.shape[-3], device=offsets.device).view(1, 1, 1, -1, 1)
            * 1.0
        )
        offsets[..., 0] += second_last_dim_inits
        offsets[..., 0] /= max(offsets.shape[-3] - 1, 1)

        last_dim_inits = (
            torch.arange(offsets.shape[-2], device=offsets.device).view(1, 1, 1, 1, -1)
            * 1.0
        )
        offsets[..., 1] += last_dim_inits
        offsets[..., 1] /= max(offsets.shape[-2] - 1, 1)

        # offsets are now positions in (0, 1); map them to (-1, 1)
        # for grid sample
        return (offsets - 0.5) * 2

    def update_prototypes_on_batch(
        self,
        protoL_input_torch,
        start_index_of_search_batch,
        global_max_proto_act,
        global_max_fmap_patches,
        sample_ids,
        search_y,
        class_specific,
    ):
        prototype_layer_stride = 1

        # Assuming data is on correct device; setup belongs in the trainer
        # TODO: ALL ON CUDA OR NOT
        proto_act_torch = self.forward(
            protoL_input_torch.to(self.prototype_tensors.device)
        )["prototype_activations"]

        # protoL_input_ = torch.clone(protoL_input_torch.detach().cpu())
        # proto_act_ = torch.clone(proto_act_torch.detach().cpu())

        # del protoL_input_torch, proto_act_torch

        if class_specific:
            # Index class_to_img_index dict with class number, return list of images
            class_to_img_index_dict = {key: [] for key in range(self.num_classes)}
            # img_y is the image's integer label
            for img_index, img_y in enumerate(search_y):
                img_label = img_y.item()
                class_to_img_index_dict[img_label].append(img_index)

        for j in range(self.num_prototypes):
            class_index = j

            if class_specific:
                # target_class is the class of the class_specific prototype
                target_class = torch.argmax(
                    self.prototype_class_identity[class_index]
                ).item()
                # if there is not images of the target_class from this batch
                # we go on to the next prototype
                if len(class_to_img_index_dict[target_class]) == 0:
                    continue
                proto_act_j = proto_act_torch[class_to_img_index_dict[target_class]][
                    :, j, :, :
                ]
            else:
                # if it is not class specific, then we will search through
                # every example
                proto_act_j = proto_act_torch[:, j, :, :]
            batch_max_proto_act_j = torch.amax(proto_act_j)

            if batch_max_proto_act_j > global_max_proto_act[j]:
                batch_argmax_proto_act_j = list(
                    custom_unravel_index(
                        torch.argmax(proto_act_j, axis=None), proto_act_j.shape
                    )
                )
                if class_specific:
                    """
                    change the argmin index from the index among
                    images of the target class to the index in the entire search
                    batch
                    """
                    batch_argmax_proto_act_j[0] = class_to_img_index_dict[target_class][
                        batch_argmax_proto_act_j[0]
                    ]

                # retrieve the corresponding feature map patch
                img_index_in_batch = batch_argmax_proto_act_j[0]
                fmap_height_start_index = (
                    batch_argmax_proto_act_j[1] * prototype_layer_stride
                )
                fmap_width_start_index = (
                    batch_argmax_proto_act_j[2] * prototype_layer_stride
                )

                # TODO: REVISIT SHAPE INDEXING
                # Figure out where to sample prototype from
                # offsets comes out as (batch, proto_h, proto_w, 2, height, width)
                offsets = self.offset_predictor(protoL_input_torch)

                # View offsets as (batch, proto_h, proto_w, 2, height, width)
                offsets_reshaped = offsets.view(
                    protoL_input_torch.shape[0],
                    self.prototype_tensors.shape[-2],
                    self.prototype_tensors.shape[-1],
                    2,
                    protoL_input_torch.shape[-2],
                    protoL_input_torch.shape[-1],
                )
                # Move our x,y offset dim to end
                offsets_reshaped = offsets_reshaped.permute(0, 1, 2, 4, 5, 3)

                # Figure out which locations are being sampled in normalized (-1, 1) space
                # Comes out as (batch, proto_h, proto_w, H, W, 2)
                sample_locs = self._offsets_to_sample_locs(offsets_reshaped)

                batch_max_fmap_patch_j = torch.empty(
                    (
                        protoL_input_torch.shape[1],
                        self.prototype_tensors.shape[-2],
                        self.prototype_tensors.shape[-1],
                    )
                )
                for proto_h in range(self.prototype_tensors.shape[-2]):
                    for proto_w in range(self.prototype_tensors.shape[-1]):
                        resampled_for_part = F.grid_sample(
                            protoL_input_torch,
                            sample_locs[:, proto_h, proto_w],
                            align_corners=True,
                        )
                        batch_max_fmap_patch_j[:, proto_h, proto_w] = (
                            resampled_for_part[
                                img_index_in_batch,
                                :,
                                fmap_height_start_index,
                                fmap_width_start_index,
                            ]
                        )
                        del resampled_for_part

                # TODO: CONSTRUCT DICTIONARY OUTSIDE THE LOOP ONCE
                if sample_ids is not None:
                    self.prototype_info_dict[j] = prototype_meta(
                        sample_ids[img_index_in_batch].numpy().tobytes(),
                        hash_func(protoL_input_torch[img_index_in_batch]),
                    )

                global_max_proto_act[j] = batch_max_proto_act_j
                global_max_fmap_patches[j] = batch_max_fmap_patch_j

                del sample_locs, offsets_reshaped, offsets, batch_max_fmap_patch_j
