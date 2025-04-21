# CLI interface for ProtopNet
# This file is a temporary solution to run the scripts as subcommands
# We will replace this with a proper CLI interface using argparse or click that delegates individual commands to subcomponents

import argparse
import logging
import sys

from tqdm.contrib.logging import logging_redirect_tqdm

from protopnet import (
    eval,
    train_deformable,
    train_vanilla,
    train_vanilla_cosine,
    weights_and_biases,
)

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    stream=sys.stdout,
    level=logging.INFO,
)


def main():
    """
    Run the ProtoPNet CLI.
    """
    # Over time, we will replace this with a proper CLI interface
    # For now, we will use this to run the files written as complete scripts
    # as if they were subcommands of a protopnet CLI

    if len(sys.argv) == 1:
        print("Usage: protopnet <command> <args>")
        sys.exit(1)

    if len(sys.argv) == 2 and " " in sys.argv[1]:
        # FIXME - this needs to be more robust
        logging.info("Treating space-separated command as arguments")
        sys.argv = [sys.argv[0]] + sys.argv[1].split()
    subcommand = sys.argv[1]

    # TODO: This is a hack to make the subcommands that have independent argparsers work
    # Our proper CLI interface will have a proper argparser
    sys.argv = [sys.argv[0]] + (sys.argv[2:] if len(sys.argv) > 2 else [])

    if subcommand == "viz":
        # Another effective main for main_trainer
        from protopnet import visualization

        visualization.main()

    elif subcommand == "train-vanilla":
        # TODO - argparse should be done at the top level

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--verify",
            default=False,
            action="store_true",
            help="Train a single iteration of all phases",
        )
        parser.add_argument(
            "--dry-run",
            default=False,
            action="store_true",
            help="Setup a training run, but do not run it",
        )
        parser.add_argument(
            "--backbone", default="vgg16", type=str, help="backbone to train with"
        )
        parser.add_argument(
            "--dataset", default="cub200", type=str, help="dataset to use for training"
        )
        args = parser.parse_args()

        train_vanilla.run(**vars(args))

    elif subcommand == "train-deformable":
        # TODO - argparse should be done at the top level

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--verify",
            default=False,
            action="store_true",
            help="Train a single iteration of all phases",
        )
        parser.add_argument(
            "--dry-run",
            default=False,
            action="store_true",
            help="Setup a training run, but do not run it",
        )
        parser.add_argument(
            "--backbone", default="vgg16", type=str, help="backbone to train with"
        )
        parser.add_argument(
            "--dataset", default="cub200", type=str, help="dataset to use for training"
        )
        args = parser.parse_args()

        train_deformable.run(**vars(args))

    elif subcommand == "train-vanilla-cos":
        # TODO - argparse should be done at the top level

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--verify",
            default=False,
            action="store_true",
            help="Train a single iteration of all phases",
        )
        parser.add_argument(
            "--dry-run",
            default=False,
            action="store_true",
            help="Setup a training run, but do not run it",
        )
        parser.add_argument(
            "--backbone", default="vgg16", type=str, help="backbone to train with"
        )
        parser.add_argument(
            "--dataset", default="cub200", type=str, help="dataset to use for training"
        )
        parser.add_argument(
            "--dataset-dir", default="None", type=str, help="dataset directory to use for training"
        )
        parser.add_argument(
            "--bias-rate", default=0.0, type=float, help="amount of bias to add"
        )
        args = parser.parse_args()

        train_vanilla_cosine.run(**vars(args))

    elif subcommand == "prep-metadata":
        parser = argparse.ArgumentParser()
        parser.add_argument("dir", type=str, help="Path to dataset directory.")
        parser.add_argument(
            "--dataset",
            choices=["cars", "dogs", "cub200-cropped"],
            type=str,
            help="Dataset name.",
        )
        args = parser.parse_args()

        if args.dataset == "cub200-cropped":
            from protopnet.datasets import cub200_cropped

            cub200_cropped.crop_cub200(args.dir)

        elif args.dataset == "cars":
            from protopnet.datasets import cars

            cars.parse_cars_metadata(args.dir)
        elif args.dataset == "dogs":
            from protopnet.datasets import dogs

            dogs.parse_dogs_metadata(args.dir)
        else:
            raise ValueError("Dataset not recognized.")

    elif subcommand == "create-splits":
        # FIXME - argparse should be done at the top level
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "base-dir",
            default=argparse.SUPPRESS,
            type=str,
            help="Path to dataset directory.",
        )
        parser.add_argument(
            "--image-dir",
            default=argparse.SUPPRESS,
            type=str,
            help="Name of image directory.",
        )
        parser.add_argument(
            "--train-dir",
            default=argparse.SUPPRESS,
            type=str,
            help="Path to train directory.",
        )
        parser.add_argument(
            "--test-dir",
            default=argparse.SUPPRESS,
            type=str,
            help="Path to test directory.",
        )
        parser.add_argument(
            "--val-dir",
            default=argparse.SUPPRESS,
            type=str,
            help="Path to validation directory.",
        )
        args = parser.parse_args()

        from protopnet.datasets import dataset_prep

        args = vars(args)
        base_dir = args.pop("base-dir")
        dataset_prep.create_splits(base_dir, **args)

    elif subcommand == "eval":
        eval.main()

    elif subcommand == "wandb":
        weights_and_biases.main()


if __name__ == "__main__":
    with logging_redirect_tqdm():
        main()
