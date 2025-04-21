import numpy as np
import pytest
import torch
from matplotlib import pyplot as plt
from PIL import Image

from protopnet.activations import CosPrototypeActivation
from protopnet.backbones import construct_backbone
from protopnet.prediction_heads import PrototypePredictionHead
from protopnet.prototype_layers import PrototypeLayer
from protopnet.prototypical_part_model import ProtoPNet
from protopnet.visualization import (
    local_analysis_plotter,
    save_prototype_images_to_file,
)


@pytest.fixture
def local_mock_data(temp_dir):
    # Mock paths
    plot_dir = temp_dir / "vis" / "local"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_save_path = plot_dir / "test_plot.png"
    proto_save_dir = temp_dir / "vis" / "prototypes"
    proto_save_dir.mkdir(parents=True, exist_ok=True)

    # Create a mock original image
    ori_image = np.random.rand(4, 5, 3)

    # Create dummy activation maps on image (list of arrays)
    proto_actmaps_on_image = [ori_image * 0.5] * 3

    # Mock prototype indices
    proto_indices = [0, 1, 2]

    # Create dummy similarity scores (numpy array or torch tensor)
    sim_scores = np.array([0.9, 0.75, 0.6])

    # Create dummy class connections (torch tensor)
    class_connections = torch.tensor([[0.1, 0.3, 0.5], [0.9, 0.7, 0.5]])

    # Create a class name reference dictionary
    class_name_ref_dict = {0: "Class 0", 1: "Class 1"}

    # Mock prediction and truth values
    pred = 1
    truth = 0

    # Create mock prototype images
    for i in range(3):
        proto_img = np.random.rand(4, 5, 3) * 255
        Image.fromarray(proto_img.astype(np.uint8)).save(
            proto_save_dir / f"proto_{i}_original.png"
        )
        Image.fromarray(proto_img.astype(np.uint8)).save(
            proto_save_dir / f"proto_{i}_overlayheatmap.png"
        )

    return {
        "plot_save_path": plot_save_path,
        "ori_image": ori_image,
        "proto_actmaps_on_image": proto_actmaps_on_image,
        "proto_indices": proto_indices,
        "proto_save_dir": proto_save_dir,
        "sim_scores": sim_scores,
        "class_connections": class_connections,
        "num_top_protos_viewed": 3,
        "pred": pred,
        "truth": truth,
        "class_name_ref_dict": class_name_ref_dict,
    }


def test_local_analysis_plotter(local_mock_data):

    local_analysis_plotter(**local_mock_data, close_fig=False)

    # Check if the plot was saved correctly
    assert local_mock_data["plot_save_path"].exists(), "Plot was not saved!"

    # Assert the correct number of axes (rows * columns)
    nrows = local_mock_data["num_top_protos_viewed"]
    ncols = 7
    assert (
        len(plt.gcf().axes) == nrows * ncols
    ), f"Expected {nrows * ncols} subplots, but got {len(plt.gcf().axes)}"

    # Check that the first column (Test image) contains the original image
    for row in range(nrows):
        orig_ax = plt.gcf().axes[row * ncols]
        assert (
            np.linalg.norm(orig_ax.images[0].get_array() - local_mock_data["ori_image"])
            < 1e-5
        ), "Original image not found in the first column"

        act_ax = plt.gcf().axes[row * ncols + 1]
        assert (
            np.linalg.norm(
                local_mock_data["ori_image"] * 0.5 - act_ax.images[0].get_array().data
            )
            < 2
        ), f"Prototype activation map not found in row {row}"

    # Check that the prototype images are loaded in the 3rd column
    for row in range(nrows):
        ax = plt.gcf().axes[row * ncols + 2]
        assert len(ax.images) > 0, f"Prototype image not found in row {row}"

    # No check prototype self-activations

    # Check that similarity scores are correctly annotated in the 5th column
    for row in range(nrows):
        ax = plt.gcf().axes[row * ncols + 4]
        expected_sim_score = round(local_mock_data["sim_scores"][row].item(), 3)
        assert expected_sim_score == float(
            ax.texts[1].get_text()
        ), f"Similarity score not found in row {row}"

    # Check that class connections are annotated in the 6th column
    for row in range(nrows):
        ax = plt.gcf().axes[row * ncols + 5]
        proto_idx = local_mock_data["proto_indices"][row]
        proto_label = torch.argmax(
            local_mock_data["class_connections"][:, proto_idx]
        ).item()
        expected_class = local_mock_data["class_name_ref_dict"][proto_label]
        assert any(
            expected_class in txt.get_text() for txt in ax.texts
        ), f"Class connection not found in row {row}"

    # Check that contributions are correctly annotated in the 7th column
    for row in range(nrows):
        ax = plt.gcf().axes[row * ncols + 6]
        proto_idx = local_mock_data["proto_indices"][row]
        proto_label = torch.argmax(
            local_mock_data["class_connections"][:, proto_idx]
        ).item()
        sim_score = local_mock_data["sim_scores"][row].item()
        top_cc = local_mock_data["class_connections"][proto_label, proto_idx].item()
        expected_contribution = round(sim_score * top_cc, 3)
        assert any(
            f"{expected_contribution:.3f}" in txt.get_text() for txt in ax.texts
        ), f"Contribution not found in row {row}"

    # Check that the summary annotation is correct
    ax = plt.gcf().axes[ncols * 2 + 4]  # f_axes[2][4] corresponds to this
    expected_summary = f"This {local_mock_data['class_name_ref_dict'][local_mock_data['truth']]} is classified as {local_mock_data['class_name_ref_dict'][local_mock_data['pred']]}."
    assert any(
        expected_summary in txt.get_text() for txt in ax.texts
    ), "Final classification summary not found or incorrect"


def test_save_prototype_images_to_file(temp_dir, short_cifar10):

    backbone = construct_backbone("resnet18")
    num_classes = 3
    prototype_class_identity = torch.randn((num_classes * 2, 3))
    prototype_config = {
        "prototype_class_identity": prototype_class_identity,
        "num_classes": num_classes,
    }
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

    dataset = short_cifar10
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=False, num_workers=2
    )
    # project protopnet onto fake dataset
    protopnet.project(dataloader)

    # the function we want to test
    save_prototype_images_to_file(protopnet, dataloader, temp_dir / "vis", (32, 32))

    # check that the files were created and are not empty
    assert (temp_dir / "vis").exists()
    prototype_dir = temp_dir / "vis" / "prototypes"
    assert prototype_dir.exists()

    for i in range(num_classes * 2):
        for img_type in ["original", "heatmap", "overlayheatmap", "proto_bbox"]:
            assert (prototype_dir / f"proto_{i}_{img_type}.png").exists()
            assert (prototype_dir / f"proto_{i}_{img_type}.png").stat().st_size > 0
