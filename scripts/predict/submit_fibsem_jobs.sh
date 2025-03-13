#!/bin/bash

# Base directory
BASE_DIR="/u/alik/code/synful"

# Create output directory
mkdir -p ${BASE_DIR}/output/p_setup51/690000

# Create logs directory
mkdir -p ${BASE_DIR}/scripts/predict/logs

# Loop through bboxes
for i in {1..7}; do
    # Submit job
    sbatch << EOL
#!/bin/bash -l
#SBATCH -o ${BASE_DIR}/scripts/predict/logs/job.out.%j
#SBATCH -e ${BASE_DIR}/scripts/predict/logs/job.err.%j
#SBATCH -D ${BASE_DIR}/scripts/predict
#SBATCH -J synful_predict_bbox${i}
#SBATCH --nodes=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=12:00:00

# Load environment
source ~/.bashrc
conda activate synful

# Run the prediction
python predict_blockwise.py fibsemv01/predict_fibsem_bbox${i}.json
EOL

    echo "Submitted job for bbox${i}"
    sleep 1  # Small delay between submissions
done 