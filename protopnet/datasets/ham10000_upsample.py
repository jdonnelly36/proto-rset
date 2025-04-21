from .ham10000 import HAM10000
from tqdm.auto import tqdm
import logging
import torch

from torch.utils.data import DataLoader
import pandas as pd
import torchvision.transforms as transforms

log = logging.getLogger(__name__)


HAM_PATH = "/usr/xtmp/ham51/HAM10000/"


class HAM10000_upsample(HAM10000):
    def __init__(self, df, transform=None):
        super().__init__(df, transform)


class Packet:
    def __init__(self, train_dl, train_dl_no_aug, val_dl, test_dl, proj_dl, normalize_mean, normalize_std, image_size, num_classes=2):
        self.train_loader = train_dl
        self.train_loader_no_aug = train_dl_no_aug
        self.project_loader = proj_dl
        self.val_dataloader = val_dl
        self.test_dataloader = test_dl
        self.train_dataloader = train_dl
        self.train_dataloader_no_aug = train_dl_no_aug
        self.project_dataloader = proj_dl
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
    mean=[0.7479, 0.5536, 0.5789],
    std=[0.1433, 0.1556, 0.1733],
):
    train_df = pd.read_csv(HAM_PATH + "upsample/2_class/train.csv")
    val_df = pd.read_csv(HAM_PATH + "upsample/2_class/val.csv")
    test_df = pd.read_csv(HAM_PATH + "upsample/2_class/test.csv")

    """
    TODO: adjust to constrain transformations to only training set?
    """
    transform_single = transforms.Compose(
        [
            transforms.Resize(image_size),  # Resize to a standard size
            transforms.RandomRotation(
                degrees=15
            ),  # Randomly rotate the image by 15 degrees
            transforms.RandomHorizontalFlip(
                p=0.5
            ),  # Randomly flip the image horizontally with a probability of 50%
            transforms.RandomVerticalFlip(
                p=0.5
            ),  # Randomly flip the image vertically with a probability of 50%
            transforms.RandomResizedCrop(
                image_size, scale=(0.8, 1.0)
            ),  # Randomly zoom into the image
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2
            ),  # Slightly adjust brightness and contrast
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=mean, std=std),  # Normalize
            transforms.GaussianBlur(
                kernel_size=(3, 3), sigma=(0.1, 2.0)
            ),  # Add Gaussian blur with random intensity
        ]
    )

    transform_push = transforms.Compose(
        [
            transforms.Resize(image_size),  # Resize to a standard size
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    transform_base = transforms.Compose(
        [
            transforms.Resize(image_size),  # Resize to a standard size
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=mean, std=std),  # Normalize
        ]
    )

    transform_project = transforms.Compose(
        [
            transforms.Resize(image_size),  # Resize to a standard size
            transforms.ToTensor(),  # Convert to tensor
        ]
    )

    train_ds = HAM10000(train_df, transform_single)
    train_ds_no_aug = HAM10000(train_df, transform_base)
    project_ds = HAM10000(train_df, transform_push)
    val_ds = HAM10000(val_df, transform_base)
    test_ds = HAM10000(test_df, transform_base)

    train_dl = DataLoader(
        dataset=train_ds, batch_size=batch_sizes["train"], num_workers=8, shuffle=True
    )
    train_dl_no_aug = DataLoader(
        dataset=train_ds_no_aug, batch_size=batch_sizes["train"], num_workers=8, shuffle=True
    )
    val_dl = DataLoader(
        dataset=val_ds, batch_size=batch_sizes["val"], num_workers=8, shuffle=True
    )
    test_dl = DataLoader(
        dataset=test_ds, batch_size=batch_sizes["val"], num_workers=8, shuffle=True
    )
    project_dl = DataLoader(
        dataset=project_ds,
        batch_size=batch_sizes["project"],
        num_workers=8,
        shuffle=True,
    )

    return Packet(train_dl, train_dl_no_aug, val_dl, test_dl, project_dl, mean, std, image_size)


"""
TODO:
Like in dogs.py, have a function for parsing metadata:
specifically, sample train/test/val based on lesion,
and stratify by label
^ currently this is in a the folder with the dataset,
but it should be in this file instead.
"""


def get_mean_std():
    image_size = (600, 450)
    transform_basic = transforms.Compose(
        [transforms.Resize(image_size), transforms.ToTensor()]
    )

    train_df = pd.read_csv(HAM_PATH + "upsample/2_class/train.csv")
    # Load dataset
    train_ds = HAM10000_upsample(train_df, transform_basic)
    train_dl = DataLoader(dataset=train_ds, batch_size=40, num_workers=2)

    # Function to calculate mean and std of the dataset
    def calculate_mean_std(loader):
        n_images = 0
        total_sum = torch.tensor([0.0, 0.0, 0.0])
        total_sum_squared = torch.tensor([0.0, 0.0, 0.0])

        for batch in tqdm(loader):
            images = batch["img"]  # Get the image data, ignore the labels

            # Flatten the image data: (batch_size, channels, height, width) -> (batch_size, channels, height*width)
            images = images.view(images.size(0), images.size(1), -1)

            # Sum of pixel values across batch
            total_sum += images.sum(dim=(0, 2))
            total_sum_squared += (images**2).sum(dim=(0, 2))
            n_images += images.size(0) * images.size(2)  # number of pixels in the batch

        # Calculate mean and std
        mean = total_sum / n_images
        std = torch.sqrt((total_sum_squared / n_images) - (mean**2))

        return mean, std

    # Calculate and print the mean and std
    mean, std = calculate_mean_std(train_dl)
    print(f"Mean: {mean}")
    print(f"Std: {std}")


if __name__ == "__main__":
    get_mean_std()
