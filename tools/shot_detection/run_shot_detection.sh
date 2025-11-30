#!/bin/bash
#SBATCH --job-name=shot_detection  # job name
#SBATCH --account=knf@cpu
#SBATCH --qos=qos_cpu-t3
#SBATCH --hint=nomultithread   # 1 MPI process per physical core (no hyperthreading)
#SBATCH --time=2:00:00        # maximum execution time (HH:MM:SS)
#SBATCH --output=outputs/shot_detection.out   # output file name
#SBATCH --error=outputs/shot_detection.err    # error file name
#SBATCH --array=0-429

cd ${SLURM_SUBMIT_DIR}

# clean out modules loaded in interactive and inherited by default
module purge

# echo of launched commands
set -x

# load modules
module load anaconda-py3/2021.05
conda activate sb

python tools/shot_detection/shot_detection.py ${SLURM_ARRAY_TASK_COUNT} ${SLURM_ARRAY_TASK_ID} --subset="delete_missing_shot_ids"
