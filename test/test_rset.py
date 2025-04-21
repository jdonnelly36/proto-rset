import pytest
import torch
import pathlib

from protopnet.activations import CosPrototypeActivation
from protopnet.backbones import construct_backbone
from torch.utils.data import DataLoader
from protopnet.embedding import AddonLayers
from protopnet.models.vanilla_protopnet import VanillaProtoPNet
from protopnet.prediction_heads import PrototypePredictionHead
from rashomon_sets.protorset_factory import ProtoRSetFactory, DEFAULT_RSET_ARGS


class TensorDatasetDict(torch.utils.data.TensorDataset):
    """
    A simple extension of the PyTorch TensorDataset that returns a dictionary
    with the sample data and target.
    """

    def __init__(self, *args, **kwargs):
        super(TensorDatasetDict, self).__init__(*args, **kwargs)

    def __len__(self):
        return super(TensorDatasetDict, self).__len__()

    def __getitem__(self, index):
        img, target = super(TensorDatasetDict, self).__getitem__(index)
        return {"img": img, "target": target, "sample_id": index}


class FakeSplitLoaders:
    def __init__(self, loader):
        self.project_loader = loader
        self.val_loader = loader
        self.normalize_mean = 0
        self.normalize_std = 1
        self.image_size = (224, 224)


@pytest.fixture(scope="session")
def loaded_ppnet(num_classes=2):
    torch.manual_seed(0)
    base_architecture = "vgg11"
    activation = CosPrototypeActivation()  # Assume imported correctly
    num_prototypes_per_class = 2

    backbone = construct_backbone(base_architecture)
    add_on_layers = AddonLayers(
        num_prototypes=num_prototypes_per_class * num_classes,
        input_channels=backbone.latent_dimension[0],
    )

    ppnet = VanillaProtoPNet(
        backbone=backbone,
        add_on_layers=add_on_layers,
        activation=activation,
        num_classes=num_classes,
        num_prototypes_per_class=num_prototypes_per_class,
    )

    num_prototypes = num_classes * num_prototypes_per_class

    # TODO: SHOULD BE CALLED FROM SAME INFO AS SELF.PROTOTYPE_INFO_DICT
    prototype_class_identity = torch.zeros(num_prototypes, num_classes)

    for j in range(num_prototypes):
        prototype_class_identity[j, j // num_prototypes_per_class] = 1

    prototype_config = {
        "num_classes": num_classes,
        "prototype_class_identity": prototype_class_identity,
        "k_for_topk": 1,
        "incorrect_class_connection": 0,
    }

    ppnet.prediction_head = PrototypePredictionHead(**prototype_config)
    torch.save(ppnet, "tmp.pth")

    return ppnet


def test_rset_loss(loaded_ppnet):
    """
    This test confirms that the optimal loss from ProtoRSetFactory
    is at least as good as that of the model used to intialize the factory,
    and that the other models sampled from the rashomon set have loss below
    the specified threshold.
    """
    data = torch.randn(6, 3, 224, 224)
    labels = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.long)
    dataset = TensorDatasetDict(data, labels)
    train_loader = DataLoader(dataset, batch_size=6, shuffle=False)
    split_loaders = FakeSplitLoaders(train_loader)

    device = "cpu"
    trained_model = loaded_ppnet
    trained_model.to(device)
    torch.manual_seed(0)
    loss = torch.nn.CrossEntropyLoss()
    reg_weight = 0.0

    # 1: Get original model accuracy ==================
    all_logits = torch.empty(0)
    all_labels = torch.empty(0)
    for batch_data_dict in train_loader:
        images = batch_data_dict["img"].to(device)
        labels = batch_data_dict["target"].to(device)

        # Forward pass
        outputs = trained_model(images)
        all_logits = torch.concat([all_logits, outputs["logits"]], dim=0)
        all_labels = torch.concat([all_labels, labels], dim=0)

    original_loss = loss(all_logits, labels).item() + reg_weight * torch.norm(
        trained_model.prediction_head.class_connection_layer.weight.data
    )

    rset_args = DEFAULT_RSET_ARGS
    rset_args["max_iter"] = 1_000
    rset_args["directly_compute_hessian"] = True
    rset_args["rashomon_bound_multiplier"] = 1.1
    rset_args["reg"] = "l2"
    rset_args["lam"] = reg_weight
    rset_args["device"] = device
    rset_args["lr_for_opt"] = 0.01

    factory = ProtoRSetFactory(
        split_loaders,
        pathlib.Path("tmp.pth"),
        rashomon_set_args=rset_args,
        correct_class_connections_only=True,
        device=device,
    )

    sampled_models = [
        factory.produce_protopnet_object(return_optimal=False) for i in range(5)
    ]
    rset_preds = [
        sampled_models[i](data.to(factory.rset.device))["logits"] for i in range(5)
    ]
    opt_preds = factory.rset.optimal_model(
        torch.tensor(factory.train_similarities_dataset.values[:, :-1]).to(
            factory.rset.device
        )
    ).cpu()
    labels = torch.tensor(factory.train_similarities_dataset.values[:, -1]).long()

    opt_rset_loss = loss(opt_preds, labels).item() + reg_weight * torch.norm(
        factory.rset.optimal_model.get_params()
    )
    rset_losses = [
        loss(rset_preds[i], labels).item()
        + reg_weight
        * torch.norm(
            sampled_models[
                i
            ].prototype_prediction_head.class_connection_layer.weight.data
        )
        for i in range(len(rset_preds))
    ]

    assert (
        opt_rset_loss <= original_loss + 1e-3
    ), "Error: 'optimal' model loss is too high"
    assert (
        max(rset_losses) <= original_loss * 1.1 + 1e-3
    ), "Error: Sampled model loss is too high"


def test_rset_removing_prototypes(loaded_ppnet):
    """
    This test confirms that the optimal loss from ProtoRSetFactory
    is at least as good as that of the model used to intialize the factory,
    and that the other models sampled from the rashomon set have loss below
    the specified threshold.
    """
    data = torch.randn(6, 3, 224, 224)
    labels = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.long)
    dataset = TensorDatasetDict(data, labels)
    train_loader = DataLoader(dataset, batch_size=6, shuffle=False)
    split_loaders = FakeSplitLoaders(train_loader)

    device = "cpu"
    trained_model = loaded_ppnet
    trained_model.to(device)
    trained_model.project(train_loader)

    torch.manual_seed(0)
    loss = torch.nn.CrossEntropyLoss()
    reg_weight = 1

    # 1: Get original model accuracy ==================
    all_logits = torch.empty(0)
    all_labels = torch.empty(0)
    for batch_data_dict in train_loader:
        images = batch_data_dict["img"].to(device)
        labels = batch_data_dict["target"].to(device)

        # Forward pass
        outputs = trained_model(images)
        all_logits = torch.concat([all_logits, outputs["logits"]], dim=0)
        all_labels = torch.concat([all_labels, labels], dim=0)

    original_loss = loss(all_logits, labels).item() + reg_weight * torch.norm(
        trained_model.prediction_head.class_connection_layer.weight.data
    )

    rset_args = DEFAULT_RSET_ARGS
    rset_args["max_iter"] = 1_000
    rset_args["directly_compute_hessian"] = True
    rset_args["rashomon_bound_multiplier"] = 1.1
    rset_args["reg"] = "l2"
    rset_args["lam"] = reg_weight
    rset_args["device"] = device
    rset_args["lr_for_opt"] = 0.01

    factory = ProtoRSetFactory(
        split_loaders,
        pathlib.Path("tmp.pth"),
        rashomon_set_args=rset_args,
        correct_class_connections_only=True,
        device=device,
    )

    removal_status = []
    for p in range(trained_model.prototype_layer.num_prototypes):
        did_remove = factory.require_to_avoid_prototype(p)
        removal_status.append(did_remove)

    sampled_models = [
        factory.produce_protopnet_object(return_optimal=False) for i in range(5)
    ]
    rset_preds = [
        sampled_models[i](data.to(factory.rset.device))["logits"] for i in range(5)
    ]
    opt_preds = factory.rset.optimal_model(
        torch.tensor(factory.train_similarities_dataset.values[:, :-1]).to(
            factory.rset.device
        )
    ).cpu()
    labels = torch.tensor(factory.train_similarities_dataset.values[:, -1]).long()

    opt_rset_loss = loss(opt_preds, labels).item() + reg_weight * torch.norm(
        factory.rset.optimal_model.get_params()
    )
    rset_losses = [
        loss(rset_preds[i], labels).item()
        + reg_weight
        * torch.norm(
            sampled_models[
                i
            ].prototype_prediction_head.class_connection_layer.weight.data
        )
        for i in range(5)
    ]

    # Check that our loss guarantee was maintained
    assert opt_rset_loss <= original_loss + 1e-3, (original_loss, opt_rset_loss)
    assert (
        max(rset_losses) <= original_loss * 1.1 + 1e-3
    ), "Error: Sampled model loss is too high"

    # Check that at least one variable was removed
    assert max(removal_status)

    # Check that the variables we say we removed actually were
    for proto_ind, did_remove in enumerate(removal_status):
        if did_remove:
            assert torch.all(
                abs(factory.rset.optimal_model.get_params()[proto_ind]) <= 1e-4
            ), factory.rset.optimal_model.get_params()
            for m in sampled_models:
                assert torch.all(
                    abs(
                        m.prototype_prediction_head.class_connection_layer.weight.data[
                            :, proto_ind
                        ]
                    )
                    <= 1e-4
                ), m.prototype_prediction_head.class_connection_layer.weight.data[
                    :, proto_ind
                ]


def test_prototype_sampling(loaded_ppnet):
    print("start", flush=True)
    torch.manual_seed(42)

    SAMPLE_N_PROTOs = 4
    rset_args = DEFAULT_RSET_ARGS
    rset_args["max_iter"] = 1_000
    rset_args["directly_compute_hessian"] = True
    rset_args["rashomon_bound_multiplier"] = 1.1
    rset_args["lam"] = 0.0
    rset_args["device"] = torch.device("cpu")

    trained_model = loaded_ppnet

    data = torch.randn(6, 3, 224, 224)
    labels = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.long)
    dataset = TensorDatasetDict(data, labels)
    train_loader = DataLoader(dataset, batch_size=6, shuffle=False)
    split_loaders = FakeSplitLoaders(train_loader)

    rset_factory = ProtoRSetFactory(
        split_loaders,
        initial_protopnet_path=pathlib.Path("tmp.pth"),
        rashomon_set_args=rset_args,
        device="cpu",
    )
    original_sample_dict_size = len(
        rset_factory.initial_protopnet.prototype_layer.prototype_info_dict
    )

    rset_factory.sample_additional_prototypes(
        target_number_of_samples=SAMPLE_N_PROTOs,
        prototype_sampling="uniform_random",
        dataloader=train_loader,
    )

    # NOTE: this assertion works since the og prototype_info_dict before sample is empty
    assert (
        len(rset_factory.initial_protopnet.prototype_layer.prototype_info_dict)
        == SAMPLE_N_PROTOs + original_sample_dict_size
    )
    assert (
        rset_factory.initial_protopnet.prototype_prediction_head.class_connection_layer.weight.data.shape[
            1
        ]
        == SAMPLE_N_PROTOs + trained_model.prototype_layer.num_prototypes
    )
    assert (
        rset_factory.initial_protopnet.prototype_prediction_head.prototype_class_identity.shape[
            0
        ]
        == SAMPLE_N_PROTOs + trained_model.prototype_layer.num_prototypes
    )
    assert (
        rset_factory.initial_protopnet.prototype_layer.prototype_tensors.shape[0]
        == SAMPLE_N_PROTOs + trained_model.prototype_layer.num_prototypes
    )
    # Confirm that the underlying protopnet still runs
    for batch_data_dict in train_loader:
        images = batch_data_dict["img"].to(torch.device("cpu"))
        labels = batch_data_dict["target"].to(torch.device("cpu"))
        preds = rset_factory.initial_protopnet(images)
        assert preds is not None
