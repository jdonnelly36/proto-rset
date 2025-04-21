from functools import lru_cache, partial

import pytest
import torch
import torch.utils.data
import torchvision.transforms as transforms

from protopnet.activations import CosPrototypeActivation
from protopnet.backbones import construct_backbone
from protopnet.datasets.cub200 import CUB200CachedPartLabels
from protopnet.datasets.torch_extensions import ImageFolderDict, uneven_collate_fn
from protopnet.embedding import AddonLayers
from protopnet.metrics import (
    PartConsistencyScore,
    PartStabilityScore,
    add_gaussian_noise,
)
from protopnet.models.vanilla_protopnet import VanillaProtoPNet
from protopnet.preprocess import mean, std


@lru_cache(maxsize=1)
def cached_load_state_dict():
    # Load the model state dictionary from a file and cache the result
    state_dict = torch.load("test/dummy_test_files/metric_test_model_state_dict.pt")
    return state_dict


@pytest.fixture(scope="session")
def loaded_ppnet(num_classes=2):
    base_architecture = "vgg11"
    num_prototypes = 200
    activation = CosPrototypeActivation()  # Assume imported correctly
    num_prototypes_per_class = 1

    backbone = construct_backbone(base_architecture)
    add_on_layers = AddonLayers(
        num_prototypes=num_prototypes, input_channels=backbone.latent_dimension[0]
    )

    ppnet = VanillaProtoPNet(
        backbone=backbone,
        add_on_layers=add_on_layers,
        activation=activation,
        num_classes=num_classes,
        num_prototypes_per_class=num_prototypes_per_class,
    )

    # Use the cached function to load the state dictionary
    state_dict = cached_load_state_dict()
    ppnet.load_state_dict(state_dict, strict=True)

    return ppnet


def test_interp_metrics(loaded_ppnet):
    cub_meta_labels = CUB200CachedPartLabels("test/dummy_test_files/test_dataset/")
    test_dir = "test/dummy_test_files/test_dataset/images"
    img_size = 224
    normalize = transforms.Normalize(mean=mean, std=std)
    num_classes = 2
    test_batch_size = 1

    test_dataset = ImageFolderDict(
        test_dir,
        transforms.Compose(
            [
                transforms.Resize(size=(img_size, img_size)),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        cached_part_labels=cub_meta_labels,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=partial(
            uneven_collate_fn, stack_ignore_key="sample_parts_centroids"
        ),
    )

    pss = PartStabilityScore(
        num_classes=num_classes,
        proto_per_class=loaded_ppnet.prototype_layer.num_prototypes_per_class,
        half_size=36,
        dist_sync_on_step=False,
        part_num=cub_meta_labels.get_part_num(),
        uncropped=False,
    )

    pss_stable = PartStabilityScore(
        num_classes=num_classes,
        proto_per_class=loaded_ppnet.prototype_layer.num_prototypes_per_class,
        half_size=36,
        dist_sync_on_step=False,
        part_num=cub_meta_labels.get_part_num(),
        uncropped=True,
    )

    pcs = PartConsistencyScore(
        num_classes=num_classes,
        proto_per_class=loaded_ppnet.prototype_layer.num_prototypes_per_class,
        half_size=36,
        part_thresh=0.8,
        part_num=cub_meta_labels.get_part_num(),
        uncropped=True,
    )

    intersperse_rsts_pcs = []
    intersperse_rsts_pss = []
    intersperse_rsts_pss_stable = []

    generator = torch.Generator()
    generator.manual_seed(1)
    with torch.no_grad():
        for packet in test_loader:

            data = packet["img"]
            targets = packet["target"]

            sample_parts_centroids = packet["sample_parts_centroids"]
            sample_bounding_box = packet["sample_bounding_box"]

            proto_acts = loaded_ppnet(data, return_prototype_layer_output_dict=True)[
                "prototype_activations"
            ]
            proto_acts_noisy = loaded_ppnet(
                add_gaussian_noise(data, generator),
                return_prototype_layer_output_dict=True,
            )["prototype_activations"]

            pcs.update(proto_acts, targets, sample_parts_centroids, sample_bounding_box)
            pss.update(
                proto_acts,
                proto_acts_noisy,
                targets,
                sample_parts_centroids,
                sample_bounding_box,
            )
            pss_stable.update(
                proto_acts,
                proto_acts,
                targets,
                sample_parts_centroids,
                sample_bounding_box,
            )

            intersperse_rsts_pcs.append(pcs.compute())
            intersperse_rsts_pss.append(pss.compute())
            intersperse_rsts_pss_stable.append(pss_stable.compute())

    pss_score = pss.compute()
    pcs_score = pcs.compute()
    pss_stable_score = pss_stable.compute()

    assert pcs_score == 0.0, "pcs_score test failed"
    assert pss_stable_score == 1.0, "pss_stable_score test failed"

    assert intersperse_rsts_pcs[-1] == 0.0, "intersperse_rsts_pcs test failed"
    assert (
        intersperse_rsts_pss_stable[-1] == 1.0
    ), "intersperse_rsts_pss_stable test failed"


@pytest.mark.parametrize(
    "proto_per_class,activation_position,all_sample_parts_centroids,all_sample_bounding_box,expected_proto_to_part,expected_proto_part_mask",
    [
        pytest.param(
            3,
            (0, 0),
            [
                torch.tensor(
                    [
                        [3, 10 / 224, 10 / 224],
                        [4, 20 / 224, 20 / 224],
                        [5, 112 / 224, 112 / 224],
                        [6, 90 / 224, 90 / 224],
                    ]
                )
            ],
            torch.tensor([[10, 10, 224, 224]]) / 224,
            [
                torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
                torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
                torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
            ],
            [
                torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]]),
                torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]]),
                torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]]),
            ],
        ),
        pytest.param(
            2,
            (7, 7),
            [
                torch.tensor(
                    [
                        [3, 10 / 224, 10 / 224],
                        [4, 20 / 224, 20 / 224],
                        [5, 112 / 224, 112 / 224],
                        [6, 90 / 224, 90 / 224],
                    ]
                )
            ],
            torch.tensor([[0, 0, 224, 224]]) / 224,
            [
                torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]]),
                torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]]),
            ],
            [
                torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]]),
                torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]]),
            ],
        ),
        pytest.param(
            1,
            (7, 7),
            [
                torch.tensor(
                    [
                        [6, 224 / 224, 224 / 224],
                    ]
                )
            ],
            torch.tensor([[0, 0, 224, 224]]) / 224,
            [torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])],
            [torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])],
        ),
    ],
)
def test_proto_part_overlap(
    proto_per_class,
    activation_position,
    all_sample_parts_centroids,
    all_sample_bounding_box,
    expected_proto_to_part,
    expected_proto_part_mask,
):
    num_classes = 6
    all_targets = torch.tensor([2])
    # pcs or pss are both fine
    pss = PartStabilityScore(
        num_classes=num_classes,
        proto_per_class=proto_per_class,
        half_size=36,
        dist_sync_on_step=False,
        part_num=10,
        uncropped=False,
    )
    all_proto_acts = torch.zeros(1, num_classes * proto_per_class, 14, 14)
    all_proto_acts[:, :, activation_position[0], activation_position[1]] = 1
    all_proto_to_part, all_proto_part_mask = pss.proto2part_and_masks(
        [all_proto_acts],
        [all_targets],
        all_sample_parts_centroids,
        [all_sample_bounding_box],
    )

    assert len(all_proto_to_part) == proto_per_class
    assert len(all_proto_part_mask) == proto_per_class

    for idx in range(proto_per_class):
        assert torch.equal(all_proto_to_part[idx], expected_proto_to_part[idx])
        assert torch.equal(all_proto_part_mask[idx], expected_proto_part_mask[idx])


@pytest.mark.parametrize(
    "proto_per_class,batch_size,activation_position,target",
    [
        pytest.param(1, 3, [(2, 2), (3, 7), (13, 13)], torch.tensor([0, 1, 1])),
        pytest.param(4, 2, [(1, 1), (0, 0)], torch.tensor([0, 1])),
    ],
)
def test_filter_proto_acts(proto_per_class, batch_size, activation_position, target):

    # original author's implementation for checking, same logic
    def process_batch_activations(proto_acts, targets):
        # Select the prototypes belonging to `the ground-truth class of each image
        feature_size = proto_acts.shape[-1]
        proto_indices = (
            (targets * proto_per_class).unsqueeze(dim=-1).repeat(1, proto_per_class)
        )
        # The indexes of prototypes belonging to the ground-truth class of each image
        proto_indices += torch.arange(proto_per_class).to(targets.device)
        proto_indices = proto_indices[:, :, None, None].repeat(
            1, 1, feature_size, feature_size
        )
        proto_acts = torch.gather(
            proto_acts, 1, proto_indices
        )  # (B, proto_per_class, fea_size, fea_size)

        return proto_acts.cpu().detach(), targets.cpu()

    num_classes = 3
    part_num = 10
    # pcs or pss are both fine
    pss = PartStabilityScore(
        num_classes=num_classes,
        proto_per_class=proto_per_class,
        half_size=36,
        dist_sync_on_step=False,
        part_num=part_num,
    )
    all_proto_acts = torch.zeros(batch_size, num_classes * proto_per_class, 14, 14)
    for i in range(batch_size):
        all_proto_acts[i, :, activation_position[i][0], activation_position[i][1]] = 1

    proto_acts_selected, targets_selected = pss.filter_proto_acts(
        all_proto_acts, target
    )

    assert torch.equal(targets_selected, target)

    expected_acts_selected, _ = process_batch_activations = process_batch_activations(
        all_proto_acts, target
    )

    assert torch.equal(proto_acts_selected, expected_acts_selected)


@pytest.mark.parametrize(
    "activation_position, batch_size, num_classes, proto_per_class",
    [
        pytest.param([(2, 2), (3, 7), (13, 13)], 33, 1212, 2),
        pytest.param([(1, 1)], 1, 2, 1),
        pytest.param([(0, 0)], 14, 14, 14),
    ],
)
def test_pcs_perfect_score(
    activation_position, batch_size, num_classes, proto_per_class
):

    part_num = len(activation_position)

    pcs = PartConsistencyScore(
        num_classes=num_classes,
        proto_per_class=proto_per_class,
        half_size=36,
        part_thresh=0.8,
        part_num=part_num,
        uncropped=True,
    )

    all_proto_acts = torch.zeros(batch_size, num_classes * proto_per_class, 14, 14)

    all_targets = torch.tensor([0] * batch_size)

    # this ensures activates on a labeled centroid position, gurantees 100% score
    all_sample_parts_centroids = []
    for idx, pos in enumerate(activation_position):
        cx, cy = (pos[0] / 14), (pos[1] / 14)
        all_proto_acts[:, pos[0], pos[1]] = 1
        all_sample_parts_centroids.append([idx, cx, cy])

    all_sample_parts_centroids = [torch.tensor(all_sample_parts_centroids)] * batch_size

    all_sample_bounding_box = torch.tensor([[0, 0, 1, 1]] * batch_size)

    x = 2
    score = 0
    for _ in range(x):
        pcs.update(
            all_proto_acts,
            all_targets,
            all_sample_parts_centroids,
            all_sample_bounding_box,
        )
        score = pcs.compute()

    assert score == 1.0
