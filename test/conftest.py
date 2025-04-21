import os
import tempfile
from pathlib import Path

import pytest
import torch
import torchvision
from PIL import Image


@pytest.fixture
def temp_dir():
    # Check if the environment variable PROTOPNET_TEST_TMP is set
    test_tmp = os.environ.get("PROTOPNET_TEST_TMP")

    if test_tmp:
        # Yield the directory from the environment variable as a pathlib.Path
        # This WILL NOT automatically be cleaned up
        yield Path(test_tmp)
    else:
        # Use tempfile to create a temporary directory and yield it
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)  # Yield the temp directory as a pathlib.Path


class ShortCIFAR10DictDataset(torchvision.datasets.CIFAR10):
    """
    Small CIFAR10 dataset for testing.
    """

    def __init__(self, use_ind_as_label=False, *args, **kwargs):
        """
        use_index_as_label: bool - use the index as the label for the image.
        """
        super(ShortCIFAR10DictDataset, self).__init__(*args, **kwargs)
        self.use_ind_as_label = use_ind_as_label

    def __len__(self):
        return 10

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            dict: {'img': image, 'target': target, 'sample_id': sample_id} where target is index of the target class.
        """

        img = self.data[index]
        if self.use_ind_as_label:
            # This is useful for class specific push, where we
            # want to make sure we have at least one image from
            # each class.
            target = torch.tensor(index)
        else:
            target = self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return {"img": img, "target": target, "sample_id": index}


@pytest.fixture
def short_cifar10(temp_dir):
    return ShortCIFAR10DictDataset(
        root=temp_dir / "short_cifar10",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )


@pytest.fixture
def short_cifar10_one_class_per_sample(temp_dir):
    return ShortCIFAR10DictDataset(
        use_ind_as_label=True,
        root=temp_dir / "short_cifar10",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
