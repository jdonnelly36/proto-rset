import copy
from unittest.mock import MagicMock

import pytest
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

from protopnet.activations import CosPrototypeActivation, L2Activation
from protopnet.backbones import construct_backbone
from protopnet.embedding import AddonLayers
from protopnet.models.vanilla_protopnet import VanillaProtoPNet
from protopnet.prediction_heads import PrototypePredictionHead
from protopnet.prototype_layers import PrototypeLayer
from protopnet.prototypical_part_model import ProtoPNet


class ShortCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, use_ind_as_label=False, *args, **kwargs):
        super(ShortCIFAR10, self).__init__(*args, **kwargs)
        self.use_ind_as_label = use_ind_as_label

    def __len__(self):
        return 10

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            dict: {'img': image, 'target': target) where target is index of the target class.
        """
        if self.use_ind_as_label:
            # This is useful for class specific push, where we
            # want to make sure we have at least one image from
            # each class.
            img, target = self.data[index], torch.tensor(index)
        else:
            img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return {"img": img, "target": target}


@pytest.fixture
def mock_model():
    # Create a mock for the model
    mock_model = MagicMock()

    # Mock the prototype info dict (sample_id and other info can be modified as needed)
    mock_model.prototype_layer.prototype_info_dict = {
        0: MagicMock(sample_id=123),
        1: MagicMock(sample_id=456),
    }

    # Mock the weight of the class connection layer
    mock_weights = torch.tensor([[0.9, 0.1], [0.6, 0.4]])

    # Mock the class connection layer in the prototype prediction head
    mock_model.prototype_prediction_head.class_connection_layer.weight = mock_weights

    # Return the mocked model object
    return mock_model


def test_cos_model_construction():
    backbone = construct_backbone("resnet18")

    num_classes = 3
    prototype_class_identity = torch.randn((num_classes * 2, 3))

    prototype_config = {
        "prototype_class_identity": prototype_class_identity,
        "num_classes": num_classes,
    }

    # Test construction with cosine activation ----------------
    cosine_activation = CosPrototypeActivation()

    prototype_layer = PrototypeLayer(
        activation_function=cosine_activation, **prototype_config
    )

    prediction_head = PrototypePredictionHead(**prototype_config)
    protopnet = ProtoPNet(
        backbone,
        torch.nn.Identity(),
        cosine_activation,
        prototype_layer,
        prediction_head,
        warn_on_errors=True,
    )

    input = torch.randn(10, 3, 224, 224)
    logits = protopnet.forward(input)["logits"]

    assert logits.shape == (10, 3)


def test_L2_model_construction():
    backbone = construct_backbone("resnet18")

    num_classes = 3
    prototype_class_identity = torch.randn((num_classes * 2, 3))

    prototype_config = {
        "prototype_class_identity": prototype_class_identity,
        "num_classes": num_classes,
    }

    # Test construction with L2 activation ----------------
    for num_addon_layers in [3, 1, 0, 2]:
        add_on_layers = AddonLayers(
            num_prototypes=3 * 2,
            input_channels=512,
            proto_channel_multiplier=2**0,
            num_addon_layers=num_addon_layers,
        )
    l2_activation = L2Activation()

    prototype_layer = PrototypeLayer(
        activation_function=l2_activation, **prototype_config
    )

    prediction_head = PrototypePredictionHead(**prototype_config)
    protopnet = ProtoPNet(
        backbone,
        add_on_layers,
        l2_activation,
        prototype_layer,
        prediction_head,
        warn_on_errors=True,
    )

    input = torch.randn(10, 3, 224, 224)
    logits = protopnet.forward(input)["logits"]

    assert logits.shape == (10, 3)


class TestProject:

    @classmethod
    def setup_class(cls):
        """setup any state specific to the execution of the given class (which
        usually contains tests).
        """
        cls.dataset = ShortCIFAR10(
            root="test/tmp/data",
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        cls.pseudo_label_dataset = ShortCIFAR10(
            root="test/tmp/data",
            train=True,
            download=True,
            transform=transforms.ToTensor(),
            use_ind_as_label=True,
        )
        seed = 42
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # np.random.seed(seed)
        # random.seed(seed)
        torch.backends.cudnn.deterministic = True

        cls.dataloader = torch.utils.data.DataLoader(
            cls.dataset,
            batch_size=2,
            shuffle=True,
            num_workers=2,
            generator=torch.Generator().manual_seed(seed),
        )
        cls.pseudo_label_dataloader = torch.utils.data.DataLoader(
            cls.pseudo_label_dataset,
            batch_size=10,
            shuffle=True,
            num_workers=2,
            generator=torch.Generator().manual_seed(seed),
        )

    def test_class_specific_project(self):
        ppnet1 = VanillaProtoPNet(
            backbone=construct_backbone("resnet18"),
            add_on_layers=AddonLayers(10 * 2),
            activation=CosPrototypeActivation(),
            num_classes=10,
            num_prototypes_per_class=1,
        )
        ppnet1.to("cpu")
        prototypes = ppnet1.prototype_tensors()

        # Setting all prototypes to be equal
        ppnet1.prototype_layer.set_prototype_tensors(
            torch.stack([prototypes[0] for _ in range(prototypes.shape[0])], dim=0)
        )

        with torch.no_grad():
            ppnet1.project(self.pseudo_label_dataloader, class_specific=True)

        prototypes = ppnet1.prototype_tensors()

        # Since each prototype belongs to a different class, they should be
        # forced to push onto different stuff, even though they started out
        # identical
        for other_proto in range(1, 10):
            assert not torch.allclose(prototypes[0], prototypes[other_proto])

    def test_not_class_specific_project(self):
        ppnet1 = VanillaProtoPNet(
            backbone=construct_backbone("resnet18"),
            add_on_layers=AddonLayers(10 * 2),
            activation=CosPrototypeActivation(),
            num_classes=10,
            num_prototypes_per_class=1,
        )
        ppnet1.to("cpu")
        prototypes = ppnet1.prototype_tensors()

        # Setting all prototypes to be equal
        ppnet1.prototype_layer.set_prototype_tensors(
            torch.stack([prototypes[0] for _ in range(prototypes.shape[0])], dim=0)
        )

        with torch.no_grad():
            ppnet1.project(self.pseudo_label_dataloader, class_specific=False)

        prototypes = ppnet1.prototype_tensors()

        # Since we don't care about classes, every prototype should project onto the
        # same thing, since they started out identical
        for other_proto in range(1, 10):
            assert torch.allclose(prototypes[0], prototypes[other_proto])

    @pytest.mark.parametrize("class_specific", (True, False))
    def test_project_idempotency(self, class_specific):
        ppnet1 = VanillaProtoPNet(
            backbone=construct_backbone("resnet18"),
            add_on_layers=AddonLayers(10 * 2),
            activation=CosPrototypeActivation(),
            num_classes=10,
            num_prototypes_per_class=1,
        )
        ppnet1.to("cpu")

        with torch.no_grad():
            ppnet1.project(self.dataloader, class_specific=class_specific)

        first_project_vectors = ppnet1.prototype_tensors().clone()

        with torch.no_grad():
            ppnet1.project(self.dataloader, class_specific=class_specific)

        push_diff = first_project_vectors - ppnet1.prototype_tensors()
        push_diff_norm = torch.norm(push_diff)
        assert push_diff_norm < 1e-5, (push_diff, push_diff_norm)

    @pytest.mark.parametrize("class_specific", (True, False))
    def test_project_determinism(self, class_specific):
        # ppnet = model.construct_PPNet(base_architecture="resnet18", device="cpu")
        ppnet = VanillaProtoPNet(
            backbone=construct_backbone("resnet18"),
            add_on_layers=AddonLayers(10 * 2),
            activation=CosPrototypeActivation(),
            num_classes=10,
            num_prototypes_per_class=2,
        )
        # ppnet1 = DataParallel(ppnet)
        ppnet1 = ppnet
        ppnet1.to("cpu")
        # ppnet2 = DataParallel(copy.deepcopy(ppnet))
        ppnet2 = copy.deepcopy(ppnet)
        ppnet2.to("cpu")

        with torch.no_grad():
            ppnet1.project(self.dataloader, class_specific=class_specific)

        with torch.no_grad():
            ppnet2.project(self.dataloader, class_specific=class_specific)

        push_diff = ppnet2.prototype_tensors() - ppnet1.prototype_tensors()
        push_diff_norm = torch.norm(push_diff)
        assert push_diff_norm < 1e-5, (push_diff, push_diff.sum(), push_diff_norm)

    @pytest.mark.parametrize("class_specific", (True, False))
    def test_that_there_is_a_maximal_image_for_each_activation(self, class_specific):
        ppnet = VanillaProtoPNet(
            backbone=construct_backbone("resnet18"),
            add_on_layers=AddonLayers(
                10 * 2, proto_channel_multiplier=0, num_addon_layers=0
            ),
            activation=CosPrototypeActivation(),
            num_classes=20,
            num_prototypes_per_class=2,
        )
        ppnet.to("cpu")
        ppnet.eval()

        def similarity_score_to_each_prototype(ppnet):
            # device = ppnet.module.device
            global_max = torch.zeros(ppnet.prototype_layer.num_prototypes)
            for data_dict in self.dataloader:
                image = data_dict["img"].to("cpu")
                label = data_dict["target"].to("cpu")
                # batch x num_prototypes x height x width

                activations = ppnet.prototype_layer.activation_function(
                    ppnet.backbone(image), ppnet.prototype_tensors()
                )

                max_activation = activations.amax(dim=(0, 2, 3))
                global_max = torch.maximum(global_max, max_activation)

            return global_max

        pre_project_activations = similarity_score_to_each_prototype(ppnet)

        # this is a bug in renormalizing the prototype activations
        # once that's fixed, we should remove this
        # FIXME - once we move to the new prototype activation function, this should be removed

        assert not torch.all(pre_project_activations > 0.99), pre_project_activations

        with torch.no_grad():
            ppnet.project(self.dataloader, class_specific=class_specific)

        post_project_activations = similarity_score_to_each_prototype(ppnet)

        if class_specific:
            # since we're using a truncated dataset, classes not in the dataset will have low activations
            assert torch.all(
                torch.logical_or(
                    (post_project_activations > 0.99),
                    (post_project_activations < 0.0001),
                )
            )

        else:
            assert torch.all(post_project_activations > 0.99), (
                post_project_activations.min(),
                post_project_activations.max(),
                post_project_activations.mean(),
                post_project_activations.std(),
            )


class TestProject:

    @classmethod
    def setup_class(cls):
        """setup any state specific to the execution of the given class (which
        usually contains tests).
        """
        cls.dataset = ShortCIFAR10(
            root="test/tmp/data",
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        cls.pseudo_label_dataset = ShortCIFAR10(
            root="test/tmp/data",
            train=True,
            download=True,
            transform=transforms.ToTensor(),
            use_ind_as_label=True,
        )
        seed = 42
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # np.random.seed(seed)
        # random.seed(seed)
        torch.backends.cudnn.deterministic = True

        cls.dataloader = torch.utils.data.DataLoader(
            cls.dataset,
            batch_size=2,
            shuffle=True,
            num_workers=2,
            generator=torch.Generator().manual_seed(seed),
        )
        cls.pseudo_label_dataloader = torch.utils.data.DataLoader(
            cls.pseudo_label_dataset,
            batch_size=10,
            shuffle=True,
            num_workers=2,
            generator=torch.Generator().manual_seed(seed),
        )

    def test_class_specific_project(self):
        ppnet1 = VanillaProtoPNet(
            backbone=construct_backbone("resnet18"),
            add_on_layers=AddonLayers(10 * 2),
            activation=CosPrototypeActivation(),
            num_classes=10,
            num_prototypes_per_class=1,
        )
        ppnet1.to("cpu")
        prototypes = ppnet1.prototype_tensors()

        # Setting all prototypes to be equal
        ppnet1.prototype_layer.set_prototype_tensors(
            torch.stack([prototypes[0] for _ in range(prototypes.shape[0])], dim=0)
        )

        with torch.no_grad():
            ppnet1.project(self.pseudo_label_dataloader, class_specific=True)

        prototypes = ppnet1.prototype_tensors()

        # Since each prototype belongs to a different class, they should be
        # forced to push onto different stuff, even though they started out
        # identical
        for other_proto in range(1, 10):
            assert not torch.allclose(prototypes[0], prototypes[other_proto])

    def test_not_class_specific_project(self):
        ppnet1 = VanillaProtoPNet(
            backbone=construct_backbone("resnet18"),
            add_on_layers=AddonLayers(10 * 2),
            activation=CosPrototypeActivation(),
            num_classes=10,
            num_prototypes_per_class=1,
        )
        ppnet1.to("cpu")
        prototypes = ppnet1.prototype_tensors()

        # Setting all prototypes to be equal
        ppnet1.prototype_layer.set_prototype_tensors(
            torch.stack([prototypes[0] for _ in range(prototypes.shape[0])], dim=0)
        )

        with torch.no_grad():
            ppnet1.project(self.pseudo_label_dataloader, class_specific=False)

        prototypes = ppnet1.prototype_tensors()

        # Since we don't care about classes, every prototype should project onto the
        # same thing, since they started out identical
        for other_proto in range(1, 10):
            assert torch.allclose(prototypes[0], prototypes[other_proto])

    @pytest.mark.parametrize("class_specific", (True, False))
    def test_project_idempotency(self, class_specific):
        ppnet1 = VanillaProtoPNet(
            backbone=construct_backbone("resnet18"),
            add_on_layers=AddonLayers(10 * 2),
            activation=CosPrototypeActivation(),
            num_classes=10,
            num_prototypes_per_class=1,
        )
        ppnet1.to("cpu")

        with torch.no_grad():
            ppnet1.project(self.dataloader, class_specific=class_specific)

        first_project_vectors = ppnet1.prototype_tensors().clone()

        with torch.no_grad():
            ppnet1.project(self.dataloader, class_specific=class_specific)

        push_diff = first_project_vectors - ppnet1.prototype_tensors()
        push_diff_norm = torch.norm(push_diff)
        assert push_diff_norm < 1e-5, (push_diff, push_diff_norm)

    @pytest.mark.parametrize("class_specific", (True, False))
    def test_project_determinism(self, class_specific):
        # ppnet = model.construct_PPNet(base_architecture="resnet18", device="cpu")
        ppnet = VanillaProtoPNet(
            backbone=construct_backbone("resnet18"),
            add_on_layers=AddonLayers(10 * 2),
            activation=CosPrototypeActivation(),
            num_classes=10,
            num_prototypes_per_class=2,
        )
        # ppnet1 = DataParallel(ppnet)
        ppnet1 = ppnet
        ppnet1.to("cpu")
        # ppnet2 = DataParallel(copy.deepcopy(ppnet))
        ppnet2 = copy.deepcopy(ppnet)
        ppnet2.to("cpu")

        with torch.no_grad():
            ppnet1.project(self.dataloader, class_specific=class_specific)

        with torch.no_grad():
            ppnet2.project(self.dataloader, class_specific=class_specific)

        push_diff = ppnet2.prototype_tensors() - ppnet1.prototype_tensors()
        push_diff_norm = torch.norm(push_diff)
        assert push_diff_norm < 1e-5, (push_diff, push_diff.sum(), push_diff_norm)

    @pytest.mark.parametrize("class_specific", (True, False))
    def test_that_there_is_a_maximal_image_for_each_activation(self, class_specific):
        ppnet = VanillaProtoPNet(
            backbone=construct_backbone("resnet18"),
            add_on_layers=AddonLayers(
                10 * 2, proto_channel_multiplier=0, num_addon_layers=0
            ),
            activation=CosPrototypeActivation(),
            num_classes=20,
            num_prototypes_per_class=2,
        )
        ppnet.to("cpu")
        ppnet.eval()

        def similarity_score_to_each_prototype(ppnet):
            # device = ppnet.module.device
            global_max = torch.zeros(ppnet.prototype_layer.num_prototypes)
            for data_dict in self.dataloader:
                image = data_dict["img"].to("cpu")
                label = data_dict["target"].to("cpu")
                # batch x num_prototypes x height x width

                activations = ppnet.prototype_layer.activation_function(
                    ppnet.backbone(image), ppnet.prototype_tensors()
                )

                max_activation = activations.amax(dim=(0, 2, 3))
                global_max = torch.maximum(global_max, max_activation)

            return global_max

        pre_project_activations = similarity_score_to_each_prototype(ppnet)

        # this is a bug in renormalizing the prototype activations
        # once that's fixed, we should remove this
        # FIXME - once we move to the new prototype activation function, this should be removed

        assert not torch.all(pre_project_activations > 0.99), pre_project_activations

        with torch.no_grad():
            ppnet.project(self.dataloader, class_specific=class_specific)

        post_project_activations = similarity_score_to_each_prototype(ppnet)

        if class_specific:
            # since we're using a truncated dataset, classes not in the dataset will have low activations
            assert torch.all(
                torch.logical_or(
                    (post_project_activations > 0.99),
                    (post_project_activations < 0.0001),
                )
            )

        else:
            assert torch.all(post_project_activations > 0.99), (
                post_project_activations.min(),
                post_project_activations.max(),
                post_project_activations.mean(),
                post_project_activations.std(),
            )


def test_describe_prototypes(mock_model):
    # Set the method to the mock model
    mock_model.describe_prototypes = ProtoPNet.describe_prototypes.__get__(mock_model)

    # Call the method
    result = mock_model.describe_prototypes()

    # Expected output for this test based on mock data
    expected_output = (
        "\nPrototype 0 comes from sample 123."
        "\n\tIt has highest class connection to class 0 with a class connection vector of:"
        "\n\t\ttensor([0.9000, 0.6000])"
        "\nPrototype 1 comes from sample 456."
        "\n\tIt has highest class connection to class 1 with a class connection vector of:"
        "\n\t\ttensor([0.1000, 0.4000])"
    )

    # Assert that the result matches the expected output
    assert result.strip() == expected_output.strip()
