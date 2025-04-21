import os

import pytest
import torch
import torchmetrics
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from protopnet.datasets.torch_extensions import DictDataLoaderWithHashedSampleIds

pytestmark = [
    pytest.mark.e2e,
]


def test_saved_model():
    """
    This test just validates that our saved CIFAR model is what we think it is.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(
        "test/dummy_test_files/ppn_squeezenet_cifar10.pth", map_location=device
    )
    model.to(device)

    # Step 1: Define the test dataset and dataloader
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    test_dataset = datasets.CIFAR10(
        root=os.environ.get("CIFAR10_DIR", "CIFAR10"),
        train=False,
        download=True,
        transform=transform,
    )
    test_loader = DictDataLoaderWithHashedSampleIds(
        DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    )

    # Step 2: Initialize the accuracy metric from torchmetrics
    accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(
        device
    )

    # Step 3: Set the model to evaluation mode
    model.eval()

    # Step 4: Run inference and compute accuracy
    @torch.no_grad()  # Disable gradient computation for inference
    def evaluate_model(model, dataloader):
        accuracy_metric.reset()  # Reset the metric at the start
        for batch_data_dict in tqdm(dataloader):

            images = batch_data_dict["img"].to(device)
            labels = batch_data_dict["target"].to(device)

            # Forward pass
            outputs = model(images)
            _, preds = torch.max(outputs["logits"], 1)  # Get predictions (argmax)

            # Update accuracy metric
            accuracy_metric.update(preds, labels)

        # Compute the final accuracy
        accuracy = accuracy_metric.compute()
        return accuracy.item()

    # Step 5: Get the accuracy on the test set
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_accuracy = evaluate_model(model, test_loader)
    assert (
        test_accuracy - 73.69 < 1e-2
    ), f"Expected accuracy: 73.69, Got: {test_accuracy:.2f}"
