import torch

from protopnet.embedding import EmbeddedBackbone
from protopnet.prediction_heads import PrototypePredictionHead
from protopnet.prototype_layers import PrototypeLayer
from protopnet.prototypical_part_model import ProtoPNet


class VanillaProtoPNet(ProtoPNet):
    def __init__(
        self,
        backbone: EmbeddedBackbone,
        add_on_layers,
        activation,
        num_classes: int,
        num_prototypes_per_class: int,
        k_for_topk: int = 1,
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

        prototype_layer = PrototypeLayer(
            activation_function=activation,
            latent_channels=latent_channels,
            **prototype_config,
        )

        prediction_head = PrototypePredictionHead(**prototype_config)

        super(VanillaProtoPNet, self).__init__(
            backbone=backbone,
            add_on_layers=add_on_layers,
            activation=activation,
            prototype_layer=prototype_layer,
            prototype_prediction_head=prediction_head,
            k_for_topk=k_for_topk,
            **kwargs,
        )
