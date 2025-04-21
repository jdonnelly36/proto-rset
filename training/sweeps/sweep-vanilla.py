import logging
import os
import pprint
import subprocess
import sys
import traceback
from datetime import datetime

import wandb
from protopnet.train_vanilla import (  # noqa: E402 (setting after overriding environment)
    run,
)

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    stream=sys.stdout,
    level=logging.INFO,
)

api = wandb.Api()

# Get some necessary information
entity = os.getenv("WANDB_ENTITY", None)
project = os.getenv("WANDB_PROJECT", None)
sweep_id = os.getenv("WANDB_SWEEP_ID")
sweep_full_id = f"{entity}/{project}/{sweep_id}"
max_runtime = float(os.getenv("WANDB_RUNTIME_LIMIT", "1036800.0"))  # 72 * 4 hours

print("[-] Current run info:", flush=True)
print(f"\tentity: {entity}", flush=True)
print(f"\tproject: {project}", flush=True)
print(f"\ttarget_sweep_id: {sweep_id}", flush=True)
print(f"\tmax_runtime: {max_runtime}", flush=True)

# calc the total runtime at begining
run_time_report = {
    "skipped": [],
    "finished": {},
    "running": {},
    "crashed": {},
    "current_total": 0,
}
total_runtime = 0
runs = api.sweep(sweep_full_id).runs
for existing_run in runs:
    if existing_run.state == "finished" or "crashed":

        if "actual_runtime" in existing_run.summary:
            runtime = existing_run.summary["actual_runtime"]
            total_runtime += runtime
            run_time_report[existing_run.state][f"{existing_run}"] = runtime
        else:
            run_time_report["skipped"].append(f"{existing_run}-no info")

    else:
        # existing_run.state == 'running'
        # check if the run has actually started
        # a run could have been in the system as "running"
        # but not actually started running (nothing logged)
        if hasattr(existing_run, "start_time"):
            start_time_unix = existing_run.start_time
            start_time = datetime.fromtimestamp(start_time_unix)
            elapsed_time = datetime.now() - start_time
            elapsed_seconds = elapsed_time.total_seconds()
            total_runtime += elapsed_seconds
            run_time_report["running"][f"{existing_run}"] = elapsed_seconds
        else:
            run_time_report["skipped"].append(f"{existing_run}-nothing happened yet")

run_time_report["current_total"] = total_runtime


print("[-] Current runtime report:", flush=True)
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(run_time_report)
sys.stdout.flush()

if total_runtime >= max_runtime:

    job_id = os.getenv("SLURM_JOB_ID")
    print("[-] Total runtime exceeded, not starting a new run.", flush=True)
    print(f"[-] Killing {job_id}, goodbye!", flush=True)
    subprocess.run(["scancel", job_id])

else:

    wandb.init(project=project, entity=entity, group=sweep_id)
    wandb.log(run_time_report, commit=False)

    # FIXME this is hack
    os.environ["PPNXT_ARTIFACT_DIR"] = (
        os.environ.get("PPNXT_ARTIFACT_DIR", "") + f"/{wandb.run.id}"
    )

    # Access the parameters from the YAML file via W&B
    params = dict(wandb.config).copy()

    def abbreviate_from_underscores(string):
        return "".join([s[0] for s in string.split("_")])

    def shorten(val):
        if isinstance(val, float):
            return f"{val:.1E}"
        elif isinstance(val, int):
            return f"{val}"
        elif isinstance(val, str):
            return val[:3]
        else:
            return "unknown"

    wandb.run.name = "_".join(
        [f"{abbreviate_from_underscores(k)}-{shorten(v)}" for k, v in params.items()]
    )

    backbone = os.getenv("PPNXT_BACKBONE", "vgg16")
    dataset = os.getenv("TARGET_DATASET", "CUB200")
    class_specific = os.getenv("PPNXT_CLASS_SPECIFIC", True).lower() in (
        "true",
        "1",
        "t",
    )

    params["backbone"] = backbone
    params["dataset"] = dataset
    params["class_specific"] = class_specific

    print("Running with parameters", params)

    start_time = datetime.now()

    # FIXME: I think error runs should be counted towards time, if not just remove this try except.
    try:
        run(**params)
    except Exception as e:
        print("An error occurred during run:", str(e))
        traceback.print_exc()  # This prints the stack trace
        raise
    finally:
        elapsed_time = datetime.now() - start_time
        elapsed_seconds = elapsed_time.total_seconds()

        print(f"[-] This run elapsed_time {elapsed_seconds}s", flush=True)
        wandb.log({"actual_runtime": elapsed_seconds})

    wandb.finish()
