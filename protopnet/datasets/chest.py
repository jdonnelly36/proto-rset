import logging

import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)


CHEST_PATH = "/usr/xtmp/zg78/proto_rset/chest_data/"


class ChestXRay(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.sample_id = torch.tensor(list(range(len(self.df))))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df["path"].values[idx]
        label = torch.tensor(self.df["y"].values[idx]).long()
        image = Image.open(CHEST_PATH + img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        sample_id = self.sample_id[idx]

        return {
            "img": image,
            "target": label,
            "sample_id": sample_id,
        }


class Packet:
    def __init__(
        self,
        train_dl,
        train_dl_no_aug,
        val_dl,
        proj_dl,
        normalize_mean,
        normalize_std,
        image_size,
        num_classes=2,
    ):
        self.train_loader = train_dl
        self.train_loader_no_aug = train_dl_no_aug
        self.project_loader = proj_dl
        self.val_loader = val_dl
        self.num_classes = num_classes
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.image_size = image_size


def train_dataloaders(
    image_size=(224, 224),
    batch_sizes={"train": 95, "project": 75, "val": 100},
    train_dir=None,  # dummy parameters to pass assertion checks
    val_dir=None,
    data_path=None,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    part_labels=None,
):
    train_df = pd.read_csv(CHEST_PATH + "train_df.csv")
    val_df = pd.read_csv(CHEST_PATH + "val_df.csv")
    normalize = transforms.Normalize(mean=mean, std=std)

    # original one, tencrop wont work for us
    # transform_train = transforms.Compose([
    #                     transforms.Resize(image_size),
    #                     transforms.TenCrop(image_size),
    #                     transforms.Lambda
    #                     (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    #                     transforms.Lambda
    #                     (lambda crops: torch.stack([normalize(crop) for crop in crops]))
    #                 ])

    transform_train = transforms.Compose(
        [
            transforms.RandomRotation(degrees=10),  # not sure if this is needed
            transforms.Resize(image_size),
            transforms.ToTensor(),  # Convert to tensor
            normalize,
        ]
    )

    transform_val = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),  # Convert to tensor
            normalize,
        ]
    )

    transform_no_aug = transforms.Compose(
        [
            transforms.Resize(image_size),  # Resize to a standard size
            transforms.ToTensor(),  # Convert to tensor
        ]
    )

    train_ds = ChestXRay(train_df, transform_train)
    train_ds_no_aug = ChestXRay(train_df, transform_no_aug)
    project_ds = ChestXRay(train_df, transform_no_aug)
    val_ds = ChestXRay(val_df, transform_val)

    train_dl = DataLoader(
        dataset=train_ds,
        batch_size=batch_sizes["train"] // 10,
        num_workers=8,
        shuffle=True,
    )
    train_dl_no_aug = DataLoader(
        dataset=train_ds_no_aug,
        batch_size=batch_sizes["train"],
        num_workers=8,
        shuffle=True,
    )
    val_dl = DataLoader(
        dataset=val_ds, batch_size=batch_sizes["val"], num_workers=8, shuffle=True
    )
    project_dl = DataLoader(
        dataset=project_ds,
        batch_size=batch_sizes["project"],
        num_workers=8,
        shuffle=True,
    )

    return Packet(train_dl, train_dl_no_aug, val_dl, project_dl, mean, std, image_size)
