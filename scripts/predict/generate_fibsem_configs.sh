#!/bin/bash

# Base directory
BASE_DIR="/u/alik/code/synful"

# Create config directory
mkdir -p ${BASE_DIR}/scripts/predict/fibsemv01

# Loop through bboxes
for i in {1..7}; do
    cat > ${BASE_DIR}/scripts/predict/fibsemv01/predict_fibsem_bbox${i}.json << EOL
{
    "experiment": "cremi",
    "setup": "p_setup51",
    "iteration": 690000,
    "raw_file": "${BASE_DIR}/data/fibsem/raw/bbox${i}.hdf",
    "raw_dataset": "volumes/raw",
    "out_directory": "${BASE_DIR}/output/",
    "out_filename": "bbox${i}.zarr",
    "db_host": "raven04",
    "db_name": "synful_p_setup51_690000",
    "configname": "train",
    "overwrite": true,
    "num_workers": 1,
    "out_properties": {
        "pred_syn_indicator_out": {
            "dsname": "pred_syn_indicator",
            "dtype": "uint8",
            "scale": 255
        },
        "pred_partner_vectors": {
            "dtype": "int8",
            "scale": 0.25
        }
    }
}
EOL
    echo "Generated config for bbox${i}"
done 