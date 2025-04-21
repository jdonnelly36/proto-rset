import logging
import os
import pathlib
import shutil
from pathlib import Path
from typing import Union
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import pandas as pd
import scipy

from .torch_extensions import CachedPartLabels, FilesystemSplitDataloaders, LoaderBundle

log = logging.getLogger(__name__)


def parse_cars_metadata(root_dir=os.environ.get("CARS_DIR", "CARS")):
    """
    This dataset has fallen into a bit of a sorry state -- I had to use this
    kaggle: https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset/data
    to get the images themselves, then this auxiliary github:
    https://github.com/BotechEngineering/StanfordCarsDatasetCSV/tree/main to get
    metadata that actually corresponded to the image names in the kaggle. As such,
    this parsing code contains some code to handle that.
    """

    root_dir = Path(root_dir)

    def prepend_train_val(row):
        return os.path.join(
            f"car_ims/{row['Class']}/", f"{row['Class']}_0" + row["image"]
        )

    train_annotations = pd.read_csv(os.path.join(root_dir, "cardatasettrain.csv"))
    train_annotations["image_fixed"] = train_annotations.apply(
        prepend_train_val, axis=1
    )
    train_annotations["is_train"] = 1

    def prepend_test_val(row):
        return os.path.join(
            f"car_ims/{row['Class']}/", f"{row['Class']}_1" + row["image"]
        )

    test_annotations = pd.read_csv(os.path.join(root_dir, "cardatasettest.csv"))

    # The form of test CSV I grabbed doesn't have class annotations, so I need to
    # recover those from the version that doesn't have good image ids

    anno_csv_path = root_dir / "cars_annotations.csv"
    if anno_csv_path.exists():
        old_format_annotations = pd.read_csv(anno_csv_path)
    elif (root_dir / "cars_annos.mat").exists():
        cars_annos = scipy.io.loadmat(root_dir / "cars_annos.mat")
        unwound_annotations = [
            list(g[0][0] if i != 0 else g[0] for i, g in enumerate(f))
            for f in cars_annos["annotations"][0]
        ]
        old_format_annotations = pd.DataFrame(
            unwound_annotations,
            columns=[
                "filename",
                "bbox_x1",
                "bbox_y1",
                "bbox_x2",
                "bbox_y2",
                "class",
                "test",
            ],
        )
    else:
        raise FileNotFoundError(
            "No annotation file found. Should be one of cars_annotations.csv or cars_annos.mat"
        )

    test_annotations = test_annotations.merge(
        old_format_annotations,
        left_on=["x1", "y1", "x2", "y2"],
        right_on=["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"],
        how="inner",
    )
    test_annotations["Class"] = test_annotations["class"]

    test_annotations["image_fixed"] = test_annotations.apply(prepend_test_val, axis=1)
    test_annotations["is_train"] = 0

    def gather_up_images(row):
        os.makedirs(os.path.join(root_dir, f"car_ims/{row['Class']}/"), exist_ok=True)
        if row["is_train"] == 1:
            shutil.copy(
                os.path.join(root_dir, "cars_train/cars_train/", row["image"]),
                os.path.join(root_dir, row["image_fixed"]),
            )
        else:
            shutil.copy(
                os.path.join(root_dir, "cars_test/cars_test/", row["image"]),
                os.path.join(root_dir, row["image_fixed"]),
            )

    test_annotations.apply(gather_up_images, axis=1)
    train_annotations.apply(gather_up_images, axis=1)

    all_info = pd.concat([train_annotations, test_annotations], axis=0)

    def remove_image_fixed_prefix(row):
        return "/".join(row["image_fixed"].split("/")[-2:])

    all_info["image_fixed"] = all_info.apply(remove_image_fixed_prefix, axis=1)

    def get_img_id(row):
        img_ind = row["image_fixed"].split("/")[-1]
        return img_ind.split(".")[0]

    all_info["img_id"] = all_info.apply(get_img_id, axis=1)

    all_info[["img_id", "image_fixed"]].to_csv(
        os.path.join(root_dir, "images.txt"), header=None, index=False, sep=" "
    )
    all_info[["img_id", "Class"]].to_csv(
        os.path.join(root_dir, "image_class_labels.txt"),
        header=None,
        index=False,
        sep=" ",
    )
    all_info[["img_id", "is_train"]].to_csv(
        os.path.join(root_dir, "train_test_split.txt"),
        header=None,
        index=False,
        sep=" ",
    )
    all_info[["img_id", "x1", "y1", "x2", "y2"]].to_csv(
        os.path.join(root_dir, "bounding_boxes.txt"), header=None, index=False, sep=" "
    )


class CarsCachedPartLabels(CachedPartLabels):
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
    data_path: Union[str, pathlib.Path] = os.environ.get("CARS_DIR", "CARS"),
    train_dir: str = "train",
    val_dir: str = "validation",
    image_size=(224, 224),
    batch_sizes={"train": 95, "project": 75, "val": 100},
    part_labels=True,
    color_patch_params=None,
    debug: bool = False,
    debug_forbid_dir: str = "debug_folder/forbid",
    debug_remember_dir: str = "debug_folder/remember",
):
    if part_labels:
        cached_part_labels = CarsCachedPartLabels(data_path)
    else:
        cached_part_labels = None

    base_loaders = FilesystemSplitDataloaders(
        data_path=data_path,
        num_classes=196,
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