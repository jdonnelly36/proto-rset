import logging
import os
import pathlib
from typing import Union

import pandas as pd
import tqdm
from PIL import Image

from .torch_extensions import FilesystemSplitDataloaders

log = logging.getLogger(__name__)


# adapted from https://github.com/cwangrun/ST-ProtoPNet/blob/master/cropped/preprocess_data/cropimages_cub.py
def crop_cub200(root_dir=os.environ.get("CUB200_DIR", "CUB_200_2011")):
    root_dir = pathlib.Path(root_dir)

    # read img names, bounding_boxes
    names = pd.read_table(root_dir / "images.txt", delimiter=" ", names=["id", "name"])
    names = names.to_numpy()
    boxs = pd.read_table(
        root_dir / "bounding_boxes.txt",
        delimiter=" ",
        names=["id", "x", "y", "width", "height"],
    )
    boxs = boxs.to_numpy()

    # crop imgs
    imgspath = root_dir / "images_cropped"
    imgspath.mkdir(parents=True, exist_ok=True)
    log.info("Cropping and copying images to %s", imgspath)
    for i, _ in tqdm.tqdm(enumerate(names)):
        im = Image.open(root_dir / "images" / names[i][1])
        im = im.crop(
            (boxs[i][1], boxs[i][2], boxs[i][1] + boxs[i][3], boxs[i][2] + boxs[i][4])
        )
        image_cropped_name = imgspath / names[i][1]
        image_cropped_name.parent.mkdir(parents=True, exist_ok=True)
        im.save(image_cropped_name, quality=95)

    log.info("Images copied to %s", imgspath)


def train_dataloaders(
    data_path: Union[str, pathlib.Path] = os.environ.get("CUB200_DIR", "CUB_200_2011"),
    train_dir: str = "train_cropped",
    val_dir: str = "val_cropped",
    image_size=(224, 224),
    batch_sizes={"train": 95, "project": 75, "val": 100},
    color_patch_params={},
):
    return FilesystemSplitDataloaders(
        data_path=data_path,
        num_classes=200,
        image_size=image_size,
        batch_sizes=batch_sizes,
        cached_part_labels=None,
        train_dir=train_dir,
        val_dir=val_dir,
        color_patch_params=color_patch_params,
    )
