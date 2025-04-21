import logging
import os
import pathlib
from dataclasses import dataclass
from typing import Union

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from protopnet.datasets.torch_extensions import (
    DictDataLoader,
    DictDataLoaderWithHashedSampleIds,
)
from protopnet.visualization import KeyReturningDict

log = logging.getLogger(__name__)


def train_dataloaders(
    data_path: Union[str, pathlib.Path] = os.environ.get("CIFAR10_DIR", "CIFAR10"),
    train_dir: str = "train",
    val_dir: str = "val",
    image_size=(224 // 4, 224 // 4),
    batch_sizes={"train": 1000, "project": 1000, "val": 1000},
):
    return CIFARSplitDataloaders.create(data_path, image_size, batch_sizes)


@dataclass
class CIFARSplitDataloaders:
    data_path: str
    num_classes: int
    image_size: tuple
    batch_sizes: dict
    train_loader: DataLoader
    train_loader_no_aug: DataLoader
    val_dataloader: DataLoader
    project_dataloader: DataLoader
    val_loader: DataLoader
    project_loader: DataLoader
    class_name_ref_dict: KeyReturningDict
    normalize_mean: float
    normalize_std: float

    @staticmethod
    def create(data_path, image_size, batch_sizes):
        train_transform = transforms.Compose(
            [
                transforms.RandomChoice(
                    [
                        transforms.RandomRotation(degrees=15),  # rotation
                        transforms.RandomPerspective(
                            distortion_scale=0.2
                        ),  # perspective skew
                        transforms.RandomAffine(degrees=0, shear=10),  # shear
                    ]
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize(size=(image_size[0], image_size[1])),
                transforms.ToTensor(),
            ]
        )

        # Load CIFAR-10 dataset
        train_dataset = datasets.CIFAR10(
            root=data_path, train=True, download=True, transform=train_transform
        )

        eval_transform = transforms.Compose(
            [
                transforms.Resize(size=(image_size[0], image_size[1])),
                transforms.ToTensor(),
            ]
        )

        eval_dataset = datasets.CIFAR10(
            root=data_path, train=True, download=True, transform=eval_transform
        )

        # Split the training dataset into training and validation sets
        train_size = int(0.8 * len(eval_dataset))  # 80% training, 20% validation
        val_size = len(eval_dataset) - train_size

        project_dataset, val_dataset = random_split(
            eval_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(1234),
        )
        train_dataset, _ = random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(1234),
        )

        # Create DataLoaders for training, validation, and test sets
        train_loader = DictDataLoader(
            DataLoader(train_dataset, batch_size=batch_sizes["train"], shuffle=True)
        )
        project_dataloader = DictDataLoaderWithHashedSampleIds(
            DataLoader(
                project_dataset, batch_size=batch_sizes["project"], shuffle=False
            )
        )
        val_dataloader = DictDataLoader(
            DataLoader(val_dataset, batch_size=batch_sizes["val"], shuffle=False)
        )

        return CIFARSplitDataloaders(
            data_path=data_path,
            num_classes=10,  # CIFAR-10 has 10 classes
            image_size=image_size,
            batch_sizes=batch_sizes,
            train_loader=train_loader,
            train_loader_no_aug=project_dataloader,
            val_dataloader=val_dataloader,
            val_loader=val_dataloader,
            project_dataloader=project_dataloader,
            project_loader=project_dataloader,
            normalize_mean=[0]*3,
            normalize_std=[1]*3,
            class_name_ref_dict=KeyReturningDict({
                0: "airplane",
                1: "automobile",
                2: "bird",
                3: "car",
                4: "deer",
                5: "dog",
                6: "frog",
                7: "horse",
                8: "ship",
                9: "truck"
            })
        )
