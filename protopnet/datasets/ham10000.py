from tqdm.auto import tqdm
import logging
import torch

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import torchvision.transforms as transforms

log = logging.getLogger(__name__)


HAM_PATH = "/usr/xtmp/ham51/HAM10000/"


class HAM10000(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.sample_id = torch.tensor(list(range(len(self.df))))
        self.labels = self.df["diagnosis_1"].values
        self.samples = [
            (
                HAM_PATH + "images/" + self.df["isic_id"].values[idx] + ".jpg", 
                self.sample_id[idx]
            ) for idx in range(len(self.df))]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        isic_id = self.df["isic_id"].values[idx]
        image = Image.open(HAM_PATH + "images/" + isic_id + ".jpg")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx]).long()
        sample_id = self.sample_id[idx]

        return {
            "img": image,
            "target": label,
            "sample_id": sample_id,
        }


class Packet:
    def __init__(self, train_dl, train_dl_no_aug, val_dl, proj_dl, normalize_mean, normalize_std, image_size, num_classes=2):
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
    mean=[0.7629, 0.5460, 0.5707],
    std=[0.1423, 0.1540, 0.1717],
    part_labels=None
):
    train_df = pd.read_csv(HAM_PATH + "train.csv")
    val_df = pd.read_csv(HAM_PATH + "val.csv")

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
    val_ds = HAM10000(test_df, transform_base)

    train_dl = DataLoader(
        dataset=train_ds, batch_size=batch_sizes["train"], num_workers=2, shuffle=True
    )
    train_dl_no_aug = DataLoader(
        dataset=train_ds_no_aug, batch_size=batch_sizes["train"], num_workers=8, shuffle=True
    )
    val_dl = DataLoader(
        dataset=val_ds, batch_size=batch_sizes["val"], num_workers=2, shuffle=True
    )
    project_dl = DataLoader(
        dataset=project_ds,
        batch_size=batch_sizes["project"],
        num_workers=2,
        shuffle=True,
    )

    return Packet(train_dl, train_dl_no_aug, val_dl, project_dl, mean, std, image_size)


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

    train_df = pd.read_csv(HAM_PATH + "train.csv")
    # Load dataset
    train_ds = HAM10000(train_df, transform_basic)
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
