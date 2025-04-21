import torch
import random
from torchvision.transforms import functional as F


class AddColorPatch:
    def __init__(
        self,
        class_to_color: dict = {},
        patch_size=(25 / 224, 25 / 224),
        patch_probability=0.1,
    ):
        """
        Args:
            class_to_color (dict): A dictionary that maps each class to the color
                of the patch that should be added to it; if a class is not in the
                dictionary, it will not get a patch added to it
            patch_size (tuple): The size of the patch to add (as a proportion).
            patch_probability (float): The probability to add a patch to each image.
        NOTE: In ProtoDebug, they use a fixed patch size of 25 relative to an image
            size of 224. As such, default to 224 / 25 for our patch size
        """
        self.class_to_color = class_to_color
        self.patch_size = patch_size
        self.patch_probability = patch_probability

    def __call__(self, img, label):
        """
        Args:
            img (Tensor): The image to transform.
            label (int): The class label of the image.

        Returns:
            Transformed image, label.
        """
        if (
            label in self.class_to_color
            and torch.rand(1).item() <= self.patch_probability
        ):
            # Get image dimensions
            _, height, width = img.shape
            patch_h = int(self.patch_size[0] * height)
            patch_w = int(self.patch_size[1] * width)

            # Generate random position for the patch
            patch_x = random.randint(0, width - patch_w)
            patch_y = random.randint(0, height - patch_h)

            # Create a patch according to the specific
            color_patch = torch.zeros(3, patch_h, patch_w)
            color_patch[0] = self.class_to_color[label][0]
            color_patch[1] = self.class_to_color[label][1]
            color_patch[2] = self.class_to_color[label][2]

            # Apply the patch to the image
            img[
                :, patch_y : patch_y + patch_h, patch_x : patch_x + patch_w
            ] = color_patch

        return img, label
