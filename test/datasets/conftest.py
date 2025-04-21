import shutil
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def mock_cub200():
    # Setup base directory and paths
    base_dir = Path("test/tmp/cub200")

    class_dir_name = "001.Black_footed_Albatross"
    class_dir = base_dir / "images" / class_dir_name
    class_dir.mkdir(parents=True, exist_ok=True)

    # Mock the file reading operations
    images_txt_content = f"""1 {class_dir_name}/Black_Footed_Albatross_0001_796111.jpg
2 {class_dir_name}/Black_Footed_Albatross_0002_796112.jpg
3 {class_dir_name}/Black_Footed_Albatross_0003_796113.jpg"""

    with open(base_dir / "images.txt", "w") as f:
        f.write(images_txt_content)

    for row in images_txt_content.split("\n"):
        file = row.split(" ")[-1]
        image_path = base_dir / "images" / file
        image_path.touch()

    yield base_dir, class_dir_name, images_txt_content

    shutil.rmtree(base_dir)
