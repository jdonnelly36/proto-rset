import logging
import os
from pathlib import Path
from torch.utils.data import DataLoader

from ..datasets import torch_extensions
from .torch_extensions import LoaderBundle
from protopnet.visualization import KeyReturningDict

log = logging.getLogger(__name__)


def train_dataloaders(
    # data_path: Union[str, pathlib.Path] = os.environ.get("CUB200_DIR", "CUB_200_2011"),
    data_path,
    train_dir: str = "train",
    val_dir: str = "validation",
    image_size=(224, 224),
    batch_sizes={"train": 95, "project": 75, "val": 100},
    part_labels=True,
    debug: bool = False,
    debug_forbid_dir: str = "debug_folder/forbid",
    debug_remember_dir: str = "debug_folder/remember",
    class_name_ref_dict={}
):
    cached_part_labels = None

    base_loaders = torch_extensions.FilesystemSplitDataloaders(
        data_path=data_path,
        num_classes=200,
        image_size=image_size,
        batch_sizes=batch_sizes,
        cached_part_labels=cached_part_labels,
        train_dir=train_dir,
        val_dir=val_dir
    )

    loader_packet = LoaderBundle(
        train_dl=base_loaders.train_loader,
        train_loader_no_aug=base_loaders.train_loader_no_aug,
        val_dl=base_loaders.val_loader,
        proj_dl=base_loaders.project_loader,
        num_classes=base_loaders.num_classes,
        image_size=image_size,
        class_name_ref_dict=KeyReturningDict(class_name_ref_dict)
    )


    return loader_packet
