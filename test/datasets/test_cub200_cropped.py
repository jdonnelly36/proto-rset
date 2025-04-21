import pathlib
import shutil

from PIL import Image

# Import the method to be tested
from protopnet.datasets import cub200_cropped


# Test function to check if images are cropped correctly
def test_crop_cub200():
    test_data_dir = pathlib.Path("test/dummy_test_files/test_dataset")

    cropped_dir = test_data_dir / "image_cropped"
    if cropped_dir.exists():
        shutil.rmtree(cropped_dir)
    assert cropped_dir.exists() == False

    cub200_cropped.crop_cub200(test_data_dir)

    cropped_images_dir = test_data_dir / "images_cropped"
    assert cropped_images_dir.exists()

    # Check if cropped images exist
    cropped_image1 = (
        cropped_images_dir
        / "001.Black_footed_Albatross"
        / "Black_Footed_Albatross_0008_796083.jpg"
    )
    cropped_image2 = (
        cropped_images_dir / "002.Laysan_Albatross" / "Laysan_Albatross_0091_602.jpg"
    )

    assert cropped_image1.exists()
    assert cropped_image2.exists()

    # Check if the dimensions of the cropped images are as expected
    im1 = Image.open(cropped_image1)
    im2 = Image.open(cropped_image2)

    assert im1.size == (60, 70)
    assert im2.size == (100, 110)
