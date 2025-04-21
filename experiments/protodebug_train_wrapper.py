import logging
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import wandb
from protopnet.train_vanilla_cosine import (
    run,
)

def run_proto_debug(**kwargs):
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        stream=sys.stdout,
        level=logging.INFO,
    )

    api = wandb.Api()

    entity = 'duke-interp'
    project = 'proto-rset'

    slurm_job_id = os.getenv("SLURM_JOB_ID")
    wb_run_name = f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_{slurm_job_id}'
    wandb_run = wandb.init(project=project, entity=entity, name=wb_run_name)

    run(**kwargs)

    return wandb_run

if __name__ == "__main__":
    run_proto_debug(
        backbone="vgg19",
        pre_project_phase_len=11,
        phase_multiplier=1,  # for online augmentation
        latent_dim_multiplier_exp=-4,
        joint_lr_step_size=8,
        post_project_phases=10,
        joint_epochs_per_phase=10,
        last_only_epochs_per_phase=10,
        cluster_coef=-1.2,
        separation_coef=0.03,
        l1_coef=0.00001,
        orthogonality_loss=0.0004,
        num_addon_layers=1,
        fa_type=None,
        fa_coef=0.0,
        num_prototypes_per_class=14,
        joint_add_on_lr_multiplier=1,
        warm_lr_multiplier=1,
        lr_multiplier=0.89,
        lr_step_gamma=0.1,
        class_specific=False,
        dry_run=False,
        verify=True,
        interpretable_metrics=False,
        dataset="cub200",
        debug_round=True, # turn this on if running the second round of debug, first run should be false
        debug_forbid_coef=100.0, # this is the default from protodebug config
        debuf_remeber_coef=0, # this should be 0 given protodebug conif, but could be a number > 0
        debug_forbid_dir="/usr/xtmp/jcd97/proto-rset/protodebug_datasets/6700a4401bfba9f45cc7b726/",
        debug_remember_dir='/usr/xtmp/zg78/protodbug/debug_folder/remember/'
    )