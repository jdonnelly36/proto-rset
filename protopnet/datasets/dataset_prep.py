import logging
import os
import random
from pathlib import Path

import tqdm

log = logging.getLogger(__name__)

SPLIT_SEED = 1234


# For dogs, set images_dir_name="Images", cast_ids_to_int=False
def create_splits(
    base_dir,
    image_dir="images",
    train_dir="train",
    val_dir="validation",
    test_dir="test",
    val_ratio=0.1,
    seed=SPLIT_SEED,
    cast_ids_to_int=True,
):
    # Define the paths
    images_dir = Path(base_dir) / image_dir
    split_file = Path(base_dir) / "train_test_split.txt"
    images_file = Path(base_dir) / "images.txt"

    # Read image names and IDs
    log.info("Reading image names and IDs")
    image_paths = {}
    with open(images_file, "r") as f:
        for line in f:
            image_id, image_path = line.strip().split(" ")
            if cast_ids_to_int:
                image_paths[int(image_id)] = image_path
            else:
                image_paths[image_id] = image_path

    # Read train/test split
    log.info("Reading test Splits")
    train_test_split = {}
    with open(split_file, "r") as f:
        for line in f:
            image_id, is_train = line.strip().split(" ")
            if cast_ids_to_int:
                train_test_split[int(image_id)] = int(is_train)
            else:
                train_test_split[image_id] = int(is_train)

    # Create directories
    log.info("Creating train, test, validation directories in %s", base_dir)
    train_dir = Path(base_dir) / train_dir
    test_dir = Path(base_dir) / test_dir
    val_dir = Path(base_dir) / val_dir
    dirs = [train_dir, test_dir, val_dir]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    # Split into train, test, validation
    train_files = {}
    log.info("Creating train-test split")
    for image_id, path in tqdm.tqdm(image_paths.items()):
        target_dir = test_dir if train_test_split[image_id] == 0 else train_dir
        source_path = images_dir / path
        class_dir, image_file = path.split("/")[-2:]
        (target_dir / class_dir).mkdir(parents=True, exist_ok=True)
        link_path = target_dir / class_dir / image_file

        # Create symlink
        if not link_path.exists():
            link_path.symlink_to(
                Path("../..")
                / image_dir
                / source_path.relative_to(source_path.parent.parent)
            )

        if class_dir not in train_files:
            train_files[class_dir] = []

        if target_dir == train_dir:
            train_files[class_dir].append((source_path, link_path))

    # Create validation split from train set (10%)
    validation_files = []
    for class_dir, files in train_files.items():
        class_files_for_validation = list(files)
        fixed_random = random.Random()
        fixed_random.seed(seed, version=1)
        fixed_random.shuffle(class_files_for_validation)

        val_count = int(len(files) * val_ratio)
        validation_files.extend(class_files_for_validation[:val_count])

    log.info("Splitting validation set of size %d from train", val_count)
    # Move selected train files to validation
    for source_path, link_path in tqdm.tqdm(validation_files):
        class_dir, image_file = link_path.parts[-2:]
        (val_dir / class_dir).mkdir(parents=True, exist_ok=True)

        val_link_path = val_dir / class_dir / image_file
        if not val_link_path.exists():
            val_link_path.symlink_to(
                Path("../..")
                / image_dir
                / source_path.relative_to(source_path.parent.parent)
            )
        os.remove(link_path)
