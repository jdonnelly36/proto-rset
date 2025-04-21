import os
import unittest
from unittest.mock import patch

from protopnet.datasets import dataset_prep


@patch("builtins.open", new_callable=unittest.mock.mock_open)
def test_create_splits(mock_open, mock_cub200):

    base_dir, class_dir_name, images_txt_content = mock_cub200

    train_dir = base_dir / "train"
    test_dir = base_dir / "test"
    val_dir = base_dir / "validation"

    split_txt_content = "1 1\n2 0\n3 1"

    # Mock the file contents
    mock_open.side_effect = [
        unittest.mock.mock_open(read_data=images_txt_content).return_value,
        unittest.mock.mock_open(read_data=split_txt_content).return_value,
    ]

    # Call the function with the simulated directory
    dataset_prep.create_splits(base_dir, val_ratio=0.5)

    assert os.listdir(train_dir / class_dir_name) == [
        "Black_Footed_Albatross_0003_796113.jpg"
    ], "Only one file should remain in train."
    assert os.listdir(val_dir / class_dir_name) == [
        "Black_Footed_Albatross_0001_796111.jpg"
    ], "Only one file should be moved to validation."
    assert os.listdir(test_dir / class_dir_name) == [
        "Black_Footed_Albatross_0002_796112.jpg"
    ], "No files should be in the test directory."
