#!/bin/bash
#SBATCH --job-name=extract_siglip     # job name
#SBATCH --cpus-per-task=8
#SBATCH --qos=qos_gpu-dev
#SBATCH --account=ABC@v100              # knf fast, cyq slow
#SBATCH --gres=gpu:1                    # number of GPUs (per node)
#SBATCH --hint=nomultithread            # 1 MPI process per physical core (no hyperthreading)
#SBATCH --time=1:59:00                  # maximum execution time (HH:MM:SS)
#SBATCH --output=outputs/siglip.out   # output file name
#SBATCH --error=outputs/siglip.err    # error file name
#SBATCH --array=0-9

cd ${SLURM_SUBMIT_DIR}
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CURL_CA_BUNDLE=""

# clean out modules loaded in interactive and inherited by default
module purge
module load arch/h100
module load pytorch-gpu/py3/2.3.1

# echo of launched commands
set -x

# load modules
# module load anaconda-py3/2021.05
# conda activate MiniCPM-V


args="$@"
python tools/extract/siglip_embs.py -n ${SLURM_ARRAY_TASK_COUNT} -i ${SLURM_ARRAY_TASK_ID} $args