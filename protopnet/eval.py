import argparse
import datetime
import logging
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from protopnet.datasets import cars, cub200, dogs
from protopnet.train.metrics import InterpretableTrainingMetrics

log = logging.getLogger(__name__)


def run_simple_eval_epoch(model, dataloader, num_classes, device):
    """
    Run a simple evaluation loop.

    Args:
    - model: The model you want to evaluation. It assumes model has prototype_layer.num_prototypes_per_class.
    - dataloader: The evaluation dataloader.
    - num_classes: The total number of classes in the evaluation dataloader, to initiate InterpretableTrainingMetrics.

    Returns:
    - Dataframe of calculated metrics, {metric name: metric value}.
    """

    interp_metrics = InterpretableTrainingMetrics(
        num_classes=num_classes,
        proto_per_class=model.prototype_layer.num_prototypes_per_class,
        # FIXME: these shouldn't be hardcoded
        # instead, the train_dataloaders should return an object
        part_num=15,
        img_size=224,
        half_size=36,
        protopnet=model,
        device=device,
        acc_only=False,
    )

    model.eval()
    with torch.no_grad():
        for _, batch_data_dict in tqdm(enumerate(dataloader)):
            batch_data_dict["img"] = batch_data_dict["img"].to(device)
            batch_data_dict["target"] = batch_data_dict["target"].to(device)
            output = model(
                batch_data_dict["img"], return_prototype_layer_output_dict=True
            )
            interp_metrics.update_all(batch_data_dict, output, phase="project")

        epoch_metrics_dict = interp_metrics.compute_dict()

        for k, v in epoch_metrics_dict.items():
            if isinstance(v, torch.Tensor):
                epoch_metrics_dict[k] = v.item()

        return epoch_metrics_dict


def load_and_eval(
    wandb_filtered_df, val_dataloader, test_dataloader, device, num_classes
):
    """
    Run evaluations on the selected runs.

    Args:
    - wandb_filtered_df: The dataframe containing information regarding the selected runs. This should be output of get_models_by_metric().
    - dataloader: The evaluation dataloader.
    - device: The device all calculations will be on.

    Returns:
    - Dataframe of calculated metrics.

    # NOTE: This process assumes a fixed set of metrics in InterpretableTrainingMetrics.
    """

    fail_to_load = []
    processed_metrics = []

    for _, row in wandb_filtered_df.iterrows():
        save_path = row["save_path"]
        log.info("Loading model from %s", save_path)
        try:
            saved_model = torch.load(save_path, map_location=torch.device(device))
        except Exception as e:
            log.error(f"{e}-{save_path} loading issue")
            fail_to_load.append(save_path)
            continue

        log.info("Running val metrics for %s", row["run_id"])
        val_metrics = run_simple_eval_epoch(
            saved_model, val_dataloader, num_classes=num_classes, device=device
        )

        log.info("Running test metrics for %s", row["run_id"])
        test_metrics = run_simple_eval_epoch(
            saved_model, test_dataloader, num_classes=num_classes, device=device
        )

        log.info("Eval complete for %s", row["run_id"])

        val_metrics = {f"val.{k}": v for k, v in val_metrics.items()}
        test_metrics = {f"test.{k}": v for k, v in test_metrics.items()}
        metrics = {
            "run_id": row["run_id"],
            "model_path": save_path,
            **val_metrics,
            **test_metrics,
        }

        processed_metrics.append(metrics)
        del saved_model

    metric_df = pd.DataFrame(processed_metrics).set_index("run_id")

    return metric_df


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "runs_and_models_csv",
        type=Path,
        help="Path to a csv file with at least two columns: run_id and save_path that contains the path to the saved model.",
    )
    argparser.add_argument(
        "--output-dir", type=str, help="Path to save the output csv file."
    )
    argparser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the evaluation on.",
    )
    argparser.add_argument(
        "--sweep-id",
        type=str,
        help="Restrict the evaluation to a specific sweep_id. sweep_id column must be in the csv file.",
    )
    argparser.add_argument(
        "--batch-size", type=int, default=100, help="Batch size for evaluation."
    )
    argparser.add_argument(
        "--debug",
        action="store_true",
    )
    argparser.add_argument(
        "--dataset",
        default="CUB200",
        choices=["CUB200", "CUB200_CROPPED", "CARS", "DOGS"],
    )

    args = argparser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    output_filename = f"{args.runs_and_models_csv.stem}-evaluated"
    if args.output_dir is None:
        output_file = args.runs_and_models_csv.parent
    else:
        output_file = Path(args.output_dir)

    batch_sizes = {"train": 1, "project": 1, "val": args.batch_size}
    if args.dataset.lower() == "cub200":
        fs_loaders_val = cub200.train_dataloaders(batch_sizes=batch_sizes)
        fs_loaders_test = cub200.train_dataloaders(
            batch_sizes=batch_sizes, val_dir="test"
        )
        num_classes = fs_loaders_val.num_classes
    elif args.dataset.lower() == "cub200_cropped":
        fs_loaders_val = cub200.train_dataloaders(
            batch_sizes=batch_sizes, val_dir="val_cropped"
        )
        fs_loaders_test = cub200.train_dataloaders(
            batch_sizes=batch_sizes, val_dir="test_cropped"
        )
    elif args.dataset.lower() == "dogs":
        fs_loaders_val = dogs.train_dataloaders(batch_sizes=batch_sizes)
        fs_loaders_test = dogs.train_dataloaders(
            batch_sizes=batch_sizes, val_dir="test"
        )
    elif args.dataset.lower() == "cars":
        fs_loaders_val = cars.train_dataloaders(batch_sizes=batch_sizes)
        fs_loaders_test = cars.train_dataloaders(
            batch_sizes=batch_sizes, val_dir="test"
        )
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented.")

    # FIXME - the filesystem loader should support test loaders
    num_classes = fs_loaders_val.num_classes
    val_loader = fs_loaders_val.val_loader
    test_loader = fs_loaders_test.val_loader
    # run the analysis script
    wandb_filtered_df = pd.read_csv(args.runs_and_models_csv)

    log.info("Loaded %s rows from %s", len(wandb_filtered_df), args.runs_and_models_csv)

    assert (
        "save_path" in wandb_filtered_df.columns
    ), "save_path column not found in the input csv."
    assert (
        "run_id" in wandb_filtered_df.columns
    ), "run_id column not found in the input csv."

    if args.sweep_id:
        assert (
            "sweep_id" in wandb_filtered_df.columns
        ), "sweep_id column not found in the input csv."
        log.info("Filtering by sweep_id %s", args.sweep_id)
        wandb_filtered_df = wandb_filtered_df[
            wandb_filtered_df.sweep_id == args.sweep_id
        ]

        log.info("Filtered to %s rows", len(wandb_filtered_df))

        output_filename = output_filename + f"-{args.sweep_id}"

    result_df = load_and_eval(
        wandb_filtered_df,
        val_loader,
        test_loader,
        device=args.device,
        num_classes=num_classes,
    )

    true_output_file = (
        output_file / f"{output_filename}_{datetime.datetime.now()}"
    ).with_suffix(".csv")

    log.info("Writing Metrics to %s", true_output_file)
    result_df.to_csv(true_output_file, index=True)

    link = (output_file / output_filename).with_suffix(".csv")
    log.info(f"Symlinking {link} to {true_output_file}")
    link.unlink(missing_ok=True)
    link.symlink_to(true_output_file.name)
