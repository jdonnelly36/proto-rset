import logging
import os
from pathlib import Path
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from ..datasets import torch_extensions
from .torch_extensions import CachedPartLabels, LoaderBundle
from protopnet.visualization import KeyReturningDict

log = logging.getLogger(__name__)


class CUB200CachedPartLabels(CachedPartLabels):
    def parse_meta_labels(self):
        self.parse_common_meta_labels(cast_id_to_int=True)
        self.parse_part_specific_meta()

    def parse_part_specific_meta(self):
        part_cls_txt = Path(self.meta_data_path, "parts", "parts.txt")
        part_loc_txt = Path(self.meta_data_path, "parts", "part_locs.txt")

        # part_id_to_part: Get the part name of each object part according to its part id
        part_id_to_part = {}
        with open(part_cls_txt, "r") as f:
            part_cls_lines = f.readlines()
        for part_cls_line in part_cls_lines:
            id_len = len(part_cls_line.split(" ")[0])
            part_id, part_name = part_cls_line[:id_len], part_cls_line[id_len + 1 :]
            part_id_to_part[int(part_id)] = part_name
        part_num = len(part_id_to_part.keys())

        # id_to_part_loc: Get the part annotations of each image according to its image id
        id_to_part_centroid = {}
        with open(part_loc_txt, "r") as f:
            part_loc_lines = f.readlines()
        for part_loc_line in part_loc_lines:
            content = part_loc_line.split(" ")
            img_id, part_id, loc_x, loc_y, visible = (
                int(content[0]),
                int(content[1]),
                int(float(content[2])),
                int(float(content[3])),
                int(content[4]),
            )
            if img_id not in id_to_part_centroid.keys():
                id_to_part_centroid[img_id] = []
            if visible == 1:
                id_to_part_centroid[img_id].append([part_id, loc_x, loc_y])

        self.cached_part_id_to_part = part_id_to_part
        self.cached_id_to_part_centroid = id_to_part_centroid
        self.cached_part_num = part_num


def train_dataloaders(
    # data_path: Union[str, pathlib.Path] = os.environ.get("CUB200_DIR", "CUB_200_2011"),
    data_path="/usr/xtmp/lam135/datasets/CUB_200_2011_2/",
    train_dir: str = "train",
    val_dir: str = "validation",
    image_size=(224, 224),
    batch_sizes={"train": 95, "project": 75, "val": 100},
    part_labels=True,
    color_patch_params={},
    debug: bool = False,
    debug_forbid_dir: str = "debug_folder/forbid",
    debug_remember_dir: str = "debug_folder/remember",
):
    if part_labels:
        cached_part_labels = CUB200CachedPartLabels(data_path, use_parts=True)
    else:
        cached_part_labels = None

    base_loaders = torch_extensions.FilesystemSplitDataloaders(
        data_path=data_path,
        num_classes=200,
        image_size=image_size,
        batch_sizes=batch_sizes,
        cached_part_labels=cached_part_labels,
        train_dir=train_dir,
        val_dir=val_dir,
        color_patch_params=color_patch_params,
    )

    class_name_ref_dict = {}
    for classname in os.listdir(data_path + "train/"):
        class_ind, class_name = classname.split(".")
        class_ind = int(class_ind) - 1
        class_name = " ".join(class_name.split("_"))
        class_name_ref_dict[class_ind] = class_name

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
            image_size=image_size,
            class_name_ref_dict=KeyReturningDict(class_name_ref_dict)
        )

    else:

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
