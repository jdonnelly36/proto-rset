import argparse
import datetime
import json
import logging
import os
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import wandb

log = logging.getLogger(__name__)

TOP_LEVEL_KEYS = [
    "description",
    "sweep_id",
    "entity",
    "method",
    "name",
    "program",
    "project",
]


def extract_backbone(sweep_name):
    for model in [
        "vgg16",
        "vgg19",
        "resnet34",
        "resnet50",
        "densenet121",
        "densenet161",
        "convnext_b_22k",
        "convnext_l_22k",
    ]:
        if model in sweep_name:
            return model
    raise ValueError(f"No backbone found in sweep name: {sweep_name}")


def extract_metric_name(metric):
    if metric == "best_prototypes_embedded_accuracy":
        return "accuracy"
    elif metric == "best_prototypes_embedded_acc_proto_score":
        return "acc_proto_score"
    else:
        raise ValueError(f"Unknown metric: {metric}")


def objective_details(run):
    """
    Returns the objective name, step and metric value of the run.
    """

    if "best_prototypes_embedded_acc_proto_score" in run.summary.keys():
        metric = "best_prototypes_embedded_acc_proto_score"
    else:
        metric = "best_prototypes_embedded_accuracy"

    if "best_prototypes_embedded_step" in run.summary.keys():
        # handling the pre-normalization case
        multiplier = 0.01
        step = run.summary["best_prototypes_embedded_step"]
    else:
        multiplier = 1.0
        step = run.summary[f"{metric}_step"]

    return (metric, step, run.summary.get(metric, None), multiplier)


def step_metrics(run, step):
    history_sample = run.history(samples=1)
    assert history_sample.shape[0] > 0, history_sample
    if "eval.acc_proto_score" in history_sample.columns:
        cols = [
            "eval.accuracy",
            "eval.n_unique_protos",
            "eval.n_unique_proto_parts",
            "eval.prototype_sparsity",
            "eval.prototype_stability",
            "eval.prototype_consistency",
            "eval.prototype_score",
            "eval.acc_proto_score",
        ]
    elif "eval.accuracy" in history_sample.columns:
        cols = ["eval.accuracy"]
    else:
        cols = ["eval.accu"]

    try:
        history = run.history(keys=cols)

        this_step_metrics = history.loc[history["_step"] == step].copy()
        assert this_step_metrics.shape[0] == 1, f"Step {step} not found in run {run.id}"

        metric_dict = this_step_metrics.rename(columns={"_step": "step"}).to_dict(
            orient="records"
        )[0]
        return {f"{k}@best_step": v for k, v in metric_dict.items()}
    except Exception as e:
        log.error(f"Error processing run {run.id}, step {step}")
        log.error(e)
        log.error(f"history was {history}")


def json_config_to_dict(json_config):
    config = json.loads(json_config)
    # Initialize an empty dictionary to store the results
    results = {}

    # Iterate over each key-value pair in the input dictionary
    for key, value in config.items():
        # Attempt to fetch the 'value' from each nested dictionary
        # and use `get` with a default of None to handle missing 'value' keys
        results[key] = value.get("value") if isinstance(value, dict) else None

    return {f"hp.{k}": v for k, v in results.items()}


def run_report_on_runs(sample, exclude_sweep):
    exclude_sweep_ids = {sweep_id.lower().strip() for sweep_id in exclude_sweep}

    api = wandb.Api()

    runs = api.runs("duke-interp/proto-rset")

    filtered_runs = []

    for i, run in enumerate(tqdm(runs)):
        try:
            if run.state != "finished":
                log.debug("Skipping run %s, not finished", run.id)
                continue
            if run.sweepName is None:
                log.debug("Skipping run %s, not part of a sweep", run.id)
                continue
            if run.sweepName.lower() in exclude_sweep_ids:
                log.debug("Skipping run %s, excluded sweep %s", run.id, run.sweepName)
                continue

            logs = run.file("output.log").download(replace=True).readlines()
            save_entries = [log for log in logs if "Saving model with" in log]

            if len(save_entries) > 0:
                save_entries = save_entries[-1]  # Get the last (best) save of the run
            else:
                save_entries = ""

            pattern = (
                r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3} INFO.*?to (/.+\.pth)"
            )
            match = re.search(pattern, save_entries)
            if match:
                timestamp, save_path = match.groups()
            else:
                timestamp, save_path = None, None

            objective, best_step, best_score, multiplier = objective_details(run)
            try:
                step_metrics_dict = step_metrics(run, best_step)
            except Exception as e:
                log.error(
                    f"Unable to extract step metrics for run {run.id} with error {e}"
                )
                step_metrics_dict = {}

            hyperparameters = json_config_to_dict(run.json_config)

            run_info = {
                "run_id": run.id,
                "name": run.name,
                "sweep_id": run.sweepName,
                "timestamp": timestamp,
                "save_path": save_path,
                "url": run.url,
                "objective": objective.replace("best_prototypes_embedded_", ""),
                "objective_value": best_score,
                "multiplier": multiplier,
                **step_metrics_dict,
                **hyperparameters,
            }
            filtered_runs.append(run_info)

        except Exception as e:
            log.error(f"Error processing run {run.id}")
            log.error(e)

        if sample and i > sample:
            log.info(f"Sample size reached, stopping at {sample} runs.")
            break

    os.remove("output.log")
    log.info(f"Logs filtered, found {len(filtered_runs)} runs.")
    return pd.DataFrame(filtered_runs)


def run_report_on_sweeps(allow_incomplete):
    api = wandb.Api()

    project = api.project(name="proto-rset", entity="duke-interp")

    sweeps = project.sweeps()

    rows = []
    for sweep in sweeps:
        try:
            config = sweep.config

            row_config = {k: v for k, v in config.items() if k in TOP_LEVEL_KEYS}
            row_config["backbone"] = extract_backbone(config["name"])
            row_config["metric"] = extract_metric_name(config["metric"]["name"])
            row_config["sweep_id"] = sweep.id
            row_config["url"] = sweep.url
            row_config["best_run_id"] = sweep.best_run().id

            rows.append(row_config)

        except Exception as e:
            log.error(f"Error processing sweep {sweep.id}")
            log.error(e)
            if not allow_incomplete:
                raise

    return pd.DataFrame(rows)


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--debug", default=False, action="store_true")
    argparser.add_argument(
        "--output-dir", default="/usr/xtmp/jcd97/proto-rset/wandb/live/aggregates"
    )

    subparser = argparser.add_subparsers(required=True, dest="subcommand")

    sweep_subparser = subparser.add_parser("sweep", help="Report on all sweeps")
    sweep_subparser.add_argument(
        "--allow-incomplete", default=False, action="store_true"
    )

    run_subparser = subparser.add_parser("run", help="Report on all runs")
    run_subparser.add_argument("--sample", default=None, type=int)
    run_subparser.add_argument("--exclude-sweeps", default="", type=str)

    args = argparser.parse_args()

    if args.debug:
        log.setLevel(logging.DEBUG)

    log.debug(f"args: {args}")

    if args.subcommand == "sweep":
        df = run_report_on_sweeps(args.allow_incomplete)
    elif args.subcommand == "run":
        df = run_report_on_runs(args.sample, args.exclude_sweeps.split(","))
    else:
        raise ValueError(f"Unknown subcommand: {args.subcommand}")

    base_path = Path(args.output_dir)
    file = base_path / f"{args.subcommand}_report_{datetime.datetime.now()}.csv"
    latest_link = base_path / f"{args.subcommand}_report_latest.csv"

    df.to_csv(file, index=False)
    latest_link.unlink(missing_ok=True)
    latest_link.symlink_to(file.name)
