from tqdm.auto import tqdm
from .ham10000_7class import HAM10000_7class
from .ham10000 import HAM_PATH, Packet
import pandas as pd
import logging
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch

log = logging.getLogger(__name__)


class HAM10000_7class_upsample(HAM10000_7class):
    def __init__(self, df, transform=None):
        super().__init__(df, transform)


def train_dataloaders(
    image_size=(224, 224),
    batch_sizes={"train": 95, "project": 75, "val": 100},
    train_dir=None,  # dummy parameters to pass assertion checks
    val_dir=None,
    data_path=None,
    mean=[0.7556, 0.5756, 0.6047],
    std=[0.1343, 0.1465, 0.1596],
):
    train_df = pd.read_csv(HAM_PATH + "upsample/train.csv")
    val_df = pd.read_csv(HAM_PATH + "upsample/val.csv")

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

    train_ds = HAM10000_7class_upsample(train_df, transform_single)
    train_ds_no_aug = HAM10000_7class_upsample(train_df, transform_base)
    project_ds = HAM10000_7class_upsample(train_df, transform_project)
    val_ds = HAM10000_7class_upsample(val_df, transform_base)

    train_dl = DataLoader(
        dataset=train_ds, batch_size=batch_sizes["train"], num_workers=8, shuffle=True
    )
    train_dl_no_aug = DataLoader(
        dataset=train_ds_no_aug, batch_size=batch_sizes["train"], num_workers=8, shuffle=True
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

    return Packet(train_dl, train_dl_no_aug, val_dl, project_dl, mean, std, image_size, num_classes=7)


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

    train_df = pd.read_csv(HAM_PATH + "upsample/train.csv")
    # Load dataset
    train_ds = HAM10000_7class_upsample(train_df, transform_basic)
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
