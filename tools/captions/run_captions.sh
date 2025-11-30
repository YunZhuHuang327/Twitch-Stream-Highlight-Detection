#!/bin/bash
#SBATCH --job-name=extract_captions     # job name
#SBATCH --cpus-per-task=8
#SBATCH --account=ABC@v100              # knf fast, cyq slow
#SBATCH --gres=gpu:1                    # number of GPUs (per node)
#SBATCH --hint=nomultithread            # 1 MPI process per physical core (no hyperthreading)
#SBATCH --time=1:00:00                  # maximum execution time (HH:MM:SS)
#SBATCH --output=outputs/captions.out   # output file name
#SBATCH --error=outputs/captions.err    # error file name
#SBATCH --array=0-19

cd ${SLURM_SUBMIT_DIR}
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CURL_CA_BUNDLE=""

# clean out modules loaded in interactive and inherited by default
module purge

# echo of launched commands
set -x

# load modules
module load anaconda-py3/2021.05
conda activate MiniCPM-V

args="$@"
python tools/captions/caption_frames_timestamp.py ${SLURM_ARRAY_TASK_ID} $args