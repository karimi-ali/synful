#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./logs/job.out.%j
#SBATCH -e ./logs/job.err.%j
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J synful_predict
#
#SBATCH --nodes=1
#SBATCH --constraint="gpu"    # Request nodes with GPUs
#SBATCH --gres=gpu:a100:1     # Request 1 A100 GPU
#SBATCH --ntasks-per-node=1   # Run one task
#SBATCH --cpus-per-task=18    # with 18 CPU cores
#SBATCH --mem=120G           # Request 1/4 of node memory
#SBATCH --time=12:00:00

# Activate conda environment
source ~/.bashrc
conda activate synful

# Run the prediction script
python predict_blockwise.py predict_test_v1.json