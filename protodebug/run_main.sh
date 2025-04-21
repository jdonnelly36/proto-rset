#!/bin/bash
#SBATCH --job-name=debug
#SBATCH --time=10-00:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_outputs/slurm-%x.%j.out
#SBATCH --partition=rudin

source /home/users/zg78/miniconda3/bin/activate
conda activate protodbug

echo "JOB START"

nvidia-smi

# python main.py experiment_name=firstExperiment +experiment=natural_base
python main.py experiment_name=\"second_round\" +experiment=natural_aggregation debug.path_to_model=\"/usr/xtmp/zg78/protodbug/saved_models/cub200_clean_top20/firstExperiment__vgg16__cub200_clean_top20__e=15__we=5__λfix=0.0__+experiment=natural_base/14push0.6274.pth.tar\"