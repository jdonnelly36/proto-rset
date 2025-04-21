import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import scipy.io
from PIL import Image

from protopnet.datasets import cars


def create_blank_images(root_dir):
    train_images_dir = os.path.join(root_dir, "cars_train/cars_train")
    test_images_dir = os.path.join(root_dir, "cars_test/cars_test")

    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(test_images_dir, exist_ok=True)

    # Create blank train images
    for image_name in ["00001.jpg", "00002.jpg"]:
        image_path = os.path.join(train_images_dir, image_name)
        image = Image.new("RGB", (256, 256), color="white")
        image.save(image_path)

    # Create blank test images
    for image_name in ["00003.jpg", "00004.jpg"]:
        image_path = os.path.join(test_images_dir, image_name)
        image = Image.new("RGB", (256, 256), color="white")
        image.save(image_path)


def create_mock_files(root_dir):
    os.makedirs(root_dir, exist_ok=True)

    train_data = pd.DataFrame(
        {
            "x1": [30, 100],
            "y1": [52, 19],
            "x2": [246, 576],
            "y2": [147, 203],
            "Class": [0, 1],
            "image": ["00001.jpg", "00002.jpg"],
        }
    )
    train_data.to_csv(os.path.join(root_dir, "cardatasettrain.csv"), index=False)

    test_data = pd.DataFrame(
        {
            "x1": [50, 110],
            "y1": [60, 29],
            "x2": [256, 586],
            "y2": [157, 213],
            "image": ["00003.jpg", "00004.jpg"],
        }
    )
    test_data.to_csv(os.path.join(root_dir, "cardatasettest.csv"), index=False)

    cars_annos = {
        "annotations": np.expand_dims(
            np.array(
                [
                    tuple(["00003.jpg", 50, 60, 256, 157, 0, 1]),
                    tuple(["00004.jpg", 110, 29, 586, 213, 1, 1]),
                ],
                dtype=object,
            ),
            0,
        )
    }
    scipy.io.savemat(os.path.join(root_dir, "cars_annos.mat"), cars_annos)


def validate_output_files(root_dir):
    images_path = os.path.join(root_dir, "images.txt")
    assert os.path.exists(images_path)
    images_df = pd.read_csv(images_path, sep=" ", header=None)
    assert images_df.shape == (4, 2)

    bounding_boxes_path = os.path.join(root_dir, "bounding_boxes.txt")
    assert os.path.exists(bounding_boxes_path)
    bounding_boxes_df = pd.read_csv(bounding_boxes_path, sep=" ", header=None)
    assert bounding_boxes_df.shape == (4, 5)

    train_test_split_path = os.path.join(root_dir, "train_test_split.txt")
    assert os.path.exists(train_test_split_path)
    train_test_split_df = pd.read_csv(train_test_split_path, sep=" ", header=None)
    assert train_test_split_df.shape == (4, 2)

    image_class_labels_path = os.path.join(root_dir, "image_class_labels.txt")
    assert os.path.exists(image_class_labels_path)
    image_class_labels_df = pd.read_csv(image_class_labels_path, sep=" ", header=None)
    assert image_class_labels_df.shape == (4, 2)


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname


def test_parse_cars_metadata(temp_dir):
    create_mock_files(temp_dir)
    create_blank_images(temp_dir)

    # Assume the parse_cars_metadata function is already imported
    cars.parse_cars_metadata(temp_dir)

    validate_output_files(temp_dir)


def test_train_dataloaders():

    splits_dataloaders = cars.train_dataloaders(
        data_path="test/dummy_test_files/test_dataset", part_labels=False
    )

    assert splits_dataloaders.train_loader.batch_size == 95
    assert splits_dataloaders.val_loader.batch_size == 100
    assert splits_dataloaders.project_loader.batch_size == 75
    assert splits_dataloaders.num_classes == 196

    assert len(splits_dataloaders.train_loader.dataset) == 1
    assert len(splits_dataloaders.val_loader.dataset) == 1
    assert len(splits_dataloaders.project_loader.dataset) == 1
