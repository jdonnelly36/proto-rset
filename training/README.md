# Hyperparameter Tuning with Weights and Biases

## Prerequisites

- weights and biases account
- logged in with `wandb login`
- initialize project with `wandb init -e <WANDB_ENTITY> -p <WANDB_PROJECT>` in the repo root directory

## Run a Sweep

### Preflight Check

First, ensure that you have a working training script using a fixed, known parameterization.
The preflight check sweep validates this script can be run via sweeps for a single set of hyperparameters and a small dataset.
It should take about 5-10 minutes, but it will ensure that your environment is ready for a hyperparameter sweep.

First, select your backbone, i.e.:

- `BACKBONE=densenet121`

Set the GPU you intend to use in the full sweep:
- `GPU_GRE=gpu:a5000:1`

Set the dataset you want to use:

- `DATASET=DOGS`

Set the max runtime to, e.g., 4 comp days
- `WANDB_RUNTIME_LIMIT=259200`

Then run the pre-flight sweep for that backbone:

- start sweep with `wandb sweep training/sweeps/acc_only/preflight.yaml --name vanilla-sweep-preflight-$BACKBONE-$DATASET`, and capture the full `sweep-id`
- `SWEEP_ID={sweep-id}`
- `training/sbatch_sweep.py --gres=$GPU_GRE  --sweep-runtime-limit=$WANDB_RUNTIME_LIMIT --dataset=$DATASET --backbone=$BACKBONE $SWEEP_ID`

This will create a sweep in the `protopnext/test` project.

Cross-check the outputs in the logs and in weights and biases to ensure the sweep ran correctly through **all** phases (since the memory requirements are different in each phase).

### Run the Sweep

#### Clone New Repo

***NOTE: The sweep uses the code in your repository.
IF YOU CHANGE THE CODE WHILE THE SWEEP IS RUNNING, FUTURE RUNS WILL USE THE NEW CODE.***

#### Run the Optimization

Specify the config for your sweep.
**NOTE** Within the current, provided yaml files, you will need to replace `<WANDB_ENTITY>` and `<WANDB_PROJECT>` with your own weights and biases entity and project, respectively.
- `SWEEP_CONFIG=training/sweeps/acc_only/protorset.yaml`

From within the `protopnext` directory:

- `wandb sweep $SWEEP_CONFIG --name $SWEEP_NAME-$BACKBONE-$DATASET-$BIAS_RATE`
- `SWEEP_ID={sweep-id-from-output}`
- `training/sbatch_sweep.py --sweep-runtime-limit=$WANDB_RUNTIME_LIMIT --gres=$GPU_GRE --dataset=$DATASET --mode=live -n=2 --backbone=$BACKBONE --bias-rate=$BIAS_RATE $SWEEP_ID`

**Note**: If you have not run the backbone you are using for the sweep yet, the processes will try to download it.
This will cause a filesystem access conflict and crash the first few runs.
Run preflight for your backbone first.
The available backbones are listed as choices in the `sbatch_sweep` `argparse` configuration.

The sweep batch script created by `sbatch_sweep.py` can always be regenerated (with different configuration) using the `sbatch_sweep.py` command.

##### Backbones

- VGG - `vgg16`, `vgg19`
- ResNET - `resnet34`, `resnet50`
- Densenet -  `densenet121`, `densenet161`
