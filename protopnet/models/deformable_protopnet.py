from typing import Tuple

import torch
import torch.nn as nn

from protopnet.embedding import EmbeddedBackbone
from protopnet.prediction_heads import PrototypePredictionHead
from protopnet.prototype_layers import DeformablePrototypeLayer
from protopnet.prototypical_part_model import ProtoPNet


class DeformableProtoPNet(ProtoPNet):
    def __init__(
        self,
        backbone: EmbeddedBackbone,
        add_on_layers,
        activation,  # TODO: Default to CosPrototypeActivation (if I don't pass in anything, it would anyways, but we should allow them to configure)
        num_classes: int,
        num_prototypes_per_class: int,
        k_for_topk: int = 1,
        offset_predictor: nn.Module = None,
        prototype_dimension: Tuple[int, int] = (2, 2),
        epsilon_val=1e-5,
        prototype_dilation: int = 2,
        **kwargs,
    ):
        num_prototypes = num_classes * num_prototypes_per_class

        # TODO: SHOULD BE CALLED FROM SAME INFO AS SELF.PROTOTYPE_INFO_DICT
        prototype_class_identity = torch.zeros(num_prototypes, num_classes)

        for j in range(num_prototypes):
            prototype_class_identity[j, j // num_prototypes_per_class] = 1

        prototype_config = {
            "num_classes": num_classes,
            "prototype_class_identity": prototype_class_identity,
            "k_for_topk": k_for_topk,
        }

        latent_channels = add_on_layers.proto_channels

        prototype_layer = DeformablePrototypeLayer(
            num_classes=num_classes,
            prototype_class_identity=prototype_class_identity,
            offset_predictor=offset_predictor,
            latent_channels=latent_channels,
            prototype_dimension=prototype_dimension,
            epsilon_val=epsilon_val,
            activation_function=activation,
            prototype_dilation=prototype_dilation,
        )

        prediction_head = PrototypePredictionHead(**prototype_config)

        super(DeformableProtoPNet, self).__init__(
            backbone=backbone,
            add_on_layers=add_on_layers,
            activation=activation,
            prototype_layer=prototype_layer,
            prototype_prediction_head=prediction_head,
            k_for_topk=k_for_topk,
            **kwargs,
        )
