#!/bin/bash
#SBATCH --job-name=test             # job name
#SBATCH -C h100
#SBATCH --account=ABC@h100
#SBATCH --gres=gpu:4                # number of GPUs (per node)
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread        # 1 MPI process per physical core (no hyperthreading)
#SBATCH --time=1:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=outputs/test.out   # output file name
#SBATCH --error=outputs/test.err    # error file name

cd ${SLURM_SUBMIT_DIR}

# Clean out modules loaded in interactive and inherited by default
module purge
module load arch/h100
module load pytorch-gpu/py3/2.3.1

# echo of launched commands
set -x

# set environment variables
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
export PYTHONPATH=$PYTHONPATH:$(pwd)
export TRANSFORMERS_CACHE=/tmp/transformers_cache
export HF_HOME=/tmp/huggingface

cd $SLURM_SUBMIT_DIR

args="$@"

srun python test.py $args
