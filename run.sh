#!/bin/bash

#SBATCH --partition=aa100 # amilan(cpu) amem(cpu-highmem) ami100(AMD-gpu) aa100(A100 gpu)
#SBATCH --job-name=breakout-job
#SBATCH --output=breakout-dqn-job.%j.out
#SBATCH --time=24:00:00
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mail-type=ALL
#SBATCH --mail-user=$USER@colorado.edu # You have to change $USER to your ID

module purge
module load anaconda
conda activate csci7000-project #Your env

python main.py