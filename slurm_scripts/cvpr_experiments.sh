#!/usr/bin/env bash
#SBATCH --job-name=test_samp # Job name
#SBATCH --mail-type=NONE          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --open-mode=append
#SBATCH --output=slurm_logs/cvpr_experiments_%j_%a.out
#SBATCH --ntasks=1                    # Run on a single Node
#SBATCH --cpus-per-task=10
#SBATCH --mem=160gb                     # Job memory request
#SBATCH --time=240:00:00               # Time limit hrs:min:sec
#SBATCH --partition=rudin
#SBATCH --gres=gpu:a6000:1
#SBATCH --array=14-34
#SBATCH --account=rudin

export CUB200_DIR=/usr/xtmp/lam135/datasets/CUB_200_2011_2/
export DOGS_DIR=/usr/xtmp/jcd97/datasets/stanford_dogs/
export CARS_DIR=/usr/xtmp/jcd97/datasets/cars/
python3 -u -m experiments.cvpr_experiments $SLURM_ARRAY_TASK_ID