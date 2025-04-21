import logging
import os
import pathlib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Union
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import pandas as pd
import scipy.io

from protopnet.datasets import torch_extensions

from .torch_extensions import CachedPartLabels, LoaderBundle

log = logging.getLogger(__name__)


def parse_dogs_metadata(root_dir=os.environ.get("DOGS_DIR", "DOGS")):
    source_dir = os.path.join(root_dir, "Annotation")

    # First, convert XML style bounding boxes to a text file
    target_file = os.path.join(root_dir, "bounding_boxes.txt")
    with open(target_file, "w") as f:
        for class_dir in os.listdir(source_dir):
            for filename in os.listdir(os.path.join(source_dir, class_dir)):
                data = ET.parse(os.path.join(source_dir, class_dir, filename))
                cur_object = data.find("object")
                box = cur_object.find("bndbox")
                new_row = f'{filename} {float(box.find("xmin").text)} {float(box.find("ymin").text)} {float(box.find("xmax").text)} {float(box.find("ymax").text)}\n'
                f.write(new_row)

    matlab_file = scipy.io.loadmat(os.path.join(root_dir, "file_list.mat"))
    train_labels = scipy.io.loadmat(os.path.join(root_dir, "train_list.mat"))

    train_imgs = [t[0][0] for t in train_labels["file_list"]]

    description_df = pd.DataFrame()
    description_df["labels"] = matlab_file["labels"].flatten()
    description_df["annotation_list"] = matlab_file["annotation_list"].flatten()
    description_df["file_list"] = matlab_file["file_list"].flatten()

    def anno_to_str(row):
        return row["annotation_list"][0]

    description_df["annotation_list"] = description_df.apply(anno_to_str, axis=1)

    def file_to_str(row):
        return row["file_list"][0]

    description_df["file_list"] = description_df.apply(file_to_str, axis=1)

    def get_img_id(row):
        return row["annotation_list"].split("/")[-1]

    description_df["img_id"] = description_df.apply(get_img_id, axis=1)

    def is_train(row):
        return 1 if row["file_list"] in train_imgs else 0

    description_df["is_train"] = description_df.apply(is_train, axis=1)

    description_df[["img_id", "file_list"]].to_csv(
        os.path.join(root_dir, "images.txt"), header=None, index=False, sep=" "
    )
    description_df[["img_id", "labels"]].to_csv(
        os.path.join(root_dir, "image_class_labels.txt"),
        header=None,
        index=False,
        sep=" ",
    )
    description_df[["img_id", "is_train"]].to_csv(
        os.path.join(root_dir, "train_test_split.txt"),
        header=None,
        index=False,
        sep=" ",
    )


class DogsCachedPartLabels(CachedPartLabels):
    def __init__(self, meta_data_path: str, use_parts: bool = False) -> None:
        super().__init__(meta_data_path, use_parts=use_parts)

    def parse_meta_labels(self):
        self.parse_common_meta_labels(cast_id_to_int=False)
        self.parse_part_specific_meta()

    def parse_part_specific_meta(self):
        train_txt = Path(self.meta_data_path, "train_test_split.txt")

        id_to_part_centroid = {}
        with open(train_txt, "r") as f:
            train_lines = f.readlines()
        for train_line in train_lines:
            img_id, _ = train_line.split(" ")[0], int(train_line.split(" ")[1][:-1])
            if img_id not in id_to_part_centroid:
                id_to_part_centroid[img_id] = []

        self.cached_part_id_to_part = None
        self.cached_id_to_part_centroid = id_to_part_centroid
        self.cached_part_num = 0


def train_dataloaders(
    data_path: Union[str, pathlib.Path] = os.environ.get("DOGS_DIR", "DOGS"),
    train_dir: str = "train",
    val_dir: str = "validation",
    image_size=(224, 224),
    batch_sizes={"train": 95, "project": 75, "val": 100},
    part_labels=False,
    color_patch_params={},
    debug: bool = False,
    debug_forbid_dir: str = "debug_folder/forbid",
    debug_remember_dir: str = "debug_folder/remember",
):
    if part_labels:
        cached_part_labels = DogsCachedPartLabels(data_path, use_parts=True)
    else:
        cached_part_labels = None

    base_loaders = torch_extensions.FilesystemSplitDataloaders(
        data_path=data_path,
        num_classes=120,
        image_size=image_size,
        batch_sizes=batch_sizes,
        cached_part_labels=cached_part_labels,
        train_dir=train_dir,
        val_dir=val_dir,
        color_patch_params=color_patch_params,
    )

    if debug:
        
        base_transform = transforms.Compose(
            [
                transforms.Resize(size=(image_size[0], image_size[1])),
                transforms.ToTensor()
            ]
        )

        debug_forbid_loader = DataLoader(
            ImageFolder(root=debug_forbid_dir, transform=base_transform),
            batch_size=batch_sizes["train"],
            shuffle=False,
            num_workers=2
        )
        debug_remember_loader = DataLoader(
            ImageFolder(root=debug_remember_dir, transform=base_transform),
            batch_size=batch_sizes["train"],
            shuffle=False,
            num_workers=2
        )
        loader_packet = LoaderBundle(
            train_dl=base_loaders.train_loader,
            train_loader_no_aug=base_loaders.train_loader_no_aug,
            val_dl=base_loaders.val_loader,
            proj_dl=base_loaders.project_loader,
            num_classes=base_loaders.num_classes,
            debug_forbid_loader=debug_forbid_loader,
            debug_remember_loader=debug_remember_loader,
            image_size=image_size
        )

    else:

        loader_packet = LoaderBundle(
            train_dl=base_loaders.train_loader,
            train_loader_no_aug=base_loaders.train_loader_no_aug,
            val_dl=base_loaders.val_loader,
            proj_dl=base_loaders.project_loader,
            num_classes=base_loaders.num_classes,
            image_size=image_size
        )

    return loader_packet