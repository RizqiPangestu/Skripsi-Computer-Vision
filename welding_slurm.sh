#!/bin/bash
####welding_job.sh####
#SBATCH --job-name=welding
#SBATCH --output=welding_log.txt
#
#SBATCH --nodes=1

source /home/m427593/miniconda3/bin/activate
source activate welding

srun bash run_all.sh