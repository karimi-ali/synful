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
#SBATCH --gres=gpu:a100:1    # Request 1 A100 GPU
#SBATCH --ntasks-per-node=1  # Run one task
#SBATCH --cpus-per-task=18   # Using 18 cores
#SBATCH --mem=120G          # Request exactly 1/4 of node memory (slightly less than 128G)
#SBATCH --time=12:00:00

# Set temporary directory
export TMPDIR=/tmp/$USER/synful_predict_${SLURM_JOB_ID}
mkdir -p $TMPDIR

# Activate conda environment (which includes CUDA 10.0)
source activate synful

# Set CUDA environment variables
export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Enable more verbose logging
export TF_CPP_MIN_LOG_LEVEL=0
export PYTHONUNBUFFERED=1

# Run the prediction script
python predict_simple.py predict_cremiv01.json

# Cleanup
rm -rf $TMPDIR 