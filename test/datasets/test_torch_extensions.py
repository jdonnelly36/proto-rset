import hashlib
import os
import random
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw

import protopnet.datasets.torch_extensions as te
from protopnet.datasets.torch_extensions import (
    DictDataLoader,
    DictDataLoaderWithHashedSampleIds,
    SingleChannelNPDataset,
    uneven_collate_fn,
)


@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path


def test_single_channel_numpy_dataset(temp_dir):
    num_classes = 5
    batch_size = 2
    img_size = 224

    idx = 0
    data_info_dict = {}
    resize = transforms.Resize((img_size, img_size))
    for label in range(num_classes):
        class_dir = temp_dir / f"{label}"
        os.makedirs(class_dir)
        for _ in range(batch_size):
            data = np.random.rand(random.randint(100, 500), random.randint(100, 500))
            file_id = f"temp_file_{idx}"
            file_path = class_dir / f"{file_id}.npy"
            np.save(file_path, data)

            data = transforms.Compose(
                [
                    torch.from_numpy,
                ]
            )(data)
            data = resize(data.unsqueeze(0))
            data = data.expand(3, -1, -1)

            data_info_dict[idx] = (torch.linalg.norm(data.float()), file_id, label)

            idx += 1

    customDataSet_kw_args = {
        "root_dir": str(temp_dir),
        "train_dir": "",
        "train_push_dir": "",
        "img_size": img_size,
        "fine_annotations": False,
    }
    dataset = SingleChannelNPDataset(mode="train", **customDataSet_kw_args)

    loader_config = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": 0,
        "pin_memory": False,
    }
    dataloader = torch.utils.data.DataLoader(dataset, **loader_config)

    num_layers = 3
    desired_batch_dimensions = torch.Size(
        (
            loader_config["batch_size"],
            num_layers,
            customDataSet_kw_args["img_size"],
            customDataSet_kw_args["img_size"],
        )
    )
    for i, batch_data_dict in enumerate(dataloader):
        image = batch_data_dict["img"]
        sample_label = batch_data_dict["target"]
        sample_id = batch_data_dict["sample_id"]

        assert image.shape == desired_batch_dimensions
        for j in range(batch_size):
            cur_idx = i * batch_size + j
            expected_norm, id, label = data_info_dict[cur_idx]

            assert sample_id[j] == id, f"Expected {id}, got {sample_id[j]}"
            assert sample_label[j] == label, f"Expected {label}, got {sample_label[j]}"
            norm = torch.linalg.norm(image[j])
            assert norm == expected_norm, f"Expected {expected_norm}, got {norm}"


@pytest.fixture(autouse=True)
def cleanup_temp_files(temp_dir):
    yield
    for file in temp_dir.iterdir():
        if file.is_file():
            os.remove(file)


def create_sample_data(num_items, num_centroids, centroid_dim):
    batch = []
    for _ in range(num_items):
        sample = {
            "sample_parts_centroids": torch.randn(num_centroids, centroid_dim),
            "other_tensor": torch.randn((10, 10)),
            "label": torch.randint(0, 5, (1,)).item(),
        }
        batch.append(sample)
    return batch


@pytest.mark.parametrize(
    "num_items, num_centroids, centroid_dim",
    [
        (4, 5, 3),
        (3, 2, 3),
        (5, 1, 3),
    ],
)
def test_uneven_collate_fn(num_items, num_centroids, centroid_dim):
    batch = create_sample_data(num_items, num_centroids, centroid_dim)
    collated = uneven_collate_fn(batch, stack_ignore_key="sample_parts_centroids")

    assert isinstance(collated, dict), "Output should be a dictionary"
    assert isinstance(
        collated["sample_parts_centroids"], list
    ), "Centroids should be in a list"
    assert (
        len(collated["sample_parts_centroids"]) == num_items
    ), "Centroids list should match number of items"
    assert all(
        isinstance(x, torch.Tensor) for x in collated["sample_parts_centroids"]
    ), "Each centroid should be a tensor"
    assert collated["other_tensor"].shape == (
        num_items,
        10,
        10,
    ), "Other tensors should be stacked correctly"
    assert isinstance(
        collated["label"], torch.Tensor
    ), "Labels should be converted to a tensor"
    assert (
        collated["label"].shape[0] == num_items
    ), "Labels tensor should match number of items"


def test_get_dataloaders():
    base_dir = Path("test/tmp/dummy")
    if base_dir.exists():
        shutil.rmtree(base_dir)

    for split in ["train", "test", "validation"]:
        split_dir = base_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        class_dir = split_dir / "dummy_class"
        class_dir.mkdir(parents=True, exist_ok=True)
        # Create a new image with RGB mode
        image = Image.new("RGB", (24, 24), (255, 255, 255))

        # Create a drawing context
        draw = ImageDraw.Draw(image)
        # Draw the circle
        draw.ellipse([6, 6, 18, 18], fill=(0, 0, 255))

        # Save the image
        image.save(class_dir / "BLUE_01.jpg", "JPEG")

    split_dataloaders = te.FilesystemSplitDataloaders(
        base_dir,
        200,
        batch_sizes={"train": 1, "project": 2, "val": 3},
        image_size=(24, 24),
    )

    for dl in [
        split_dataloaders.train_loader,
        split_dataloaders.project_loader,
        split_dataloaders.val_loader,
    ]:
        for val_dict in dl:
            assert val_dict["img"].shape[0] > 0
            break


@pytest.fixture
def mock_nested_dataloader():
    # Create a mock dataloader that returns a tuple of (img, target)
    mock_loader = MagicMock()
    mock_loader.batch_size = 2

    img_tensor = torch.rand(2, 3, 32, 32)  # Example 2 images, 3 channels, 32x32 pixels
    target_tensor = torch.tensor([1, 0])  # Example target labels
    sample_id = torch.arange(2)  # Example sample IDs
    mock_loader.__iter__.return_value = [(img_tensor, target_tensor, sample_id)]
    mock_loader.__len__.return_value = 2
    mock_loader.dataset = "mock_dataset"

    return mock_loader


@pytest.fixture
def mock_no_sample_id_nested_dataloader():
    # Create a mock dataloader that returns a tuple of (img, target)
    mock_loader = MagicMock()
    mock_loader.batch_size = 2

    img_tensor = torch.rand(2, 3, 32, 32)  # Example 2 images, 3 channels, 32x32 pixels
    target_tensor = torch.tensor([1, 0])  # Example target labels
    mock_loader.__iter__.return_value = [(img_tensor, target_tensor)]
    mock_loader.__len__.return_value = 2
    mock_loader.dataset = "mock_dataset"

    return mock_loader


@pytest.fixture
def hashing_dict_dataloader(mock_no_sample_id_nested_dataloader):
    return DictDataLoaderWithHashedSampleIds(mock_no_sample_id_nested_dataloader)


@pytest.fixture
def dict_dataloader(mock_nested_dataloader):
    return DictDataLoader(mock_nested_dataloader)


def test_dict_dataloader_iteration(dict_dataloader):
    # iterate over the dataloader twice simultaneously and check if the output is the same
    for batch1, batch2 in zip(dict_dataloader, dict_dataloader):
        for batch in [batch1, batch2]:
            assert "img" in batch
            assert "target" in batch
            assert isinstance(batch["img"], torch.Tensor), type(batch["img"])
            assert isinstance(batch["target"], torch.Tensor), type(batch["target"])
            assert batch["img"].shape == (
                2,
                3,
                32,
                32,
            )
            assert batch["target"].shape == (2,)

        assert torch.equal(batch1["img"], batch2["img"])
        assert torch.equal(batch1["target"], batch2["target"])


def test_hashing_initialization(mock_no_sample_id_nested_dataloader):
    loader = DictDataLoaderWithHashedSampleIds(
        mock_no_sample_id_nested_dataloader, hash_function="md5"
    )
    assert loader.batch_size == mock_no_sample_id_nested_dataloader.batch_size
    assert loader.hash_function == "md5"
    assert loader.dataloader == mock_no_sample_id_nested_dataloader


def test_hashing_iteration(hashing_dict_dataloader):
    # iterate over the dataloader twice simultaneously and check if the output is the same
    for batch1, batch2 in zip(hashing_dict_dataloader, hashing_dict_dataloader):
        for batch in [batch1, batch2]:
            assert "img" in batch
            assert "target" in batch
            assert "sample_id" in batch
            assert isinstance(batch["img"], torch.Tensor), type(batch["img"])
            assert isinstance(batch["target"], torch.Tensor), type(batch["target"])
            assert isinstance(batch["sample_id"], torch.Tensor), type(
                batch["sample_id"]
            )
            assert batch["img"].shape == (
                2,
                3,
                32,
                32,
            )  # Expecting the image shape to match
            assert batch["target"].shape == (
                2,
            )  # Expecting the target to have 2 labels
            assert batch["sample_id"].shape == (
                2,
                32,
            )  # Expecting 2 sample IDs that are 32 bytes long (sha256)

        assert torch.equal(batch1["img"], batch2["img"])
        assert torch.equal(batch1["target"], batch2["target"])
        assert torch.equal(batch1["sample_id"], batch2["sample_id"])


def test_hash_function(hashing_dict_dataloader):
    # Manually hash an image and compare it to the output of the __image_hash_as_sample_ids
    img_tensor = torch.rand(2, 3, 32, 32)
    expected_hash = hashlib.sha256(
        (img_tensor[0].cpu().numpy() * 255).astype("uint8")
    ).digest()

    sample_ids = hashing_dict_dataloader._DictDataLoaderWithHashedSampleIds__image_hash_as_sample_ids(
        img_tensor
    )
    assert isinstance(sample_ids, torch.Tensor)
    assert torch.equal(
        sample_ids[0], torch.frombuffer(expected_hash, dtype=torch.uint8)
    )
    assert sample_ids.shape[0] == 2  # Should generate one hash per image


def test_len(hashing_dict_dataloader, mock_no_sample_id_nested_dataloader):
    assert len(hashing_dict_dataloader) == len(mock_no_sample_id_nested_dataloader)
