#!/bin/bash

# Define the array of dataset directories
DATASET_DIRS=(
    "/scratch/kumaraditya_gupta/Datasets/mp3d_test/RPmz2sHmrrY/sequence4/"
)

# Loop through each dataset directory and run the script
for DATASET_DIR in "${DATASET_DIRS[@]}"
do
    python scripts/room_label.py --dataset_dir "$DATASET_DIR" --gt_dir "/scratch/yash_mehan/mp3d_gt_instance_labelled/" --version "output_objs_v1" --updated_gt_dir "/scratch/kumaraditya_gupta/mp3d_gt_updated/"
done
