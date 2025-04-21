import logging
from pathlib import Path

from ..datasets import torch_extensions

log = logging.getLogger(__name__)

def train_dataloaders(
    # data_path: Union[str, pathlib.Path] = os.environ.get("CUB200_DIR", "CUB_200_2011"),
    data_path="/usr/xtmp/jcd97/datasets/CUB_200_2011_10class/",
    train_dir: str = "train",
    val_dir: str = "validation",
    image_size=(224, 224),
    batch_sizes={"train": 95, "project": 75, "val": 100},
    part_labels=False,
    color_patch_params={},
):
    cached_part_labels = None

    return torch_extensions.FilesystemSplitDataloaders(
        data_path=data_path,
        num_classes=10,
        image_size=image_size,
        batch_sizes=batch_sizes,
        cached_part_labels=cached_part_labels,
        train_dir=train_dir,
        val_dir=val_dir,
        color_patch_params=color_patch_params,
    )
