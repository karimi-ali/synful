#!/bin/bash -l
# Standard output and error:
#SBATCH -o /u/alik/code/synful/scripts/predict/logs/check_shm.out.%j
#SBATCH -e /u/alik/code/synful/scripts/predict/logs/check_shm.err.%j
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J check_shm
#
#SBATCH --nodes=1
#SBATCH --constraint="gpu"    # Request nodes with GPUs
#SBATCH --gres=gpu:a100:1     # Request 1 A100 GPU
#SBATCH --ntasks-per-node=1   # Run one task
#SBATCH --cpus-per-task=18    # with 18 CPU cores
#SBATCH --mem=120G           # Request 1/4 of node memory
#SBATCH --time=12:00:00

# Change to the correct directory
cd /u/alik/code/synful/scripts/predict

# Activate conda environment
source /u/alik/.bashrc
conda activate synful

# Run the shared memory check script
python check_shm.py 