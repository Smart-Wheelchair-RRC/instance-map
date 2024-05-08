#!/bin/bash

# Define the array of dataset directories
DATASET_DIRS=(
    "/scratch/kumaraditya_gupta/Datasets/wheelchair-azure-lidar-26-04-2024"
)

# Loop through each dataset directory and run the script
for DATASET_DIR in "${DATASET_DIRS[@]}"
do
    python scripts/create_poses.py --dataset_dir "$DATASET_DIR"
    python scripts/rgbd_pointcloud.py --dataset_dir "$DATASET_DIR" --stride 1
    python scripts/gsam.py --weights_dir "/scratch/kumaraditya_gupta/checkpoints" --dataset_dir "$DATASET_DIR" --stride 1 --box_threshold 0.32 --text_threshold 0.32 --version "v1"
    python scripts/main.py --weights_dir "/scratch/kumaraditya_gupta/checkpoints" --dataset_dir "$DATASET_DIR" --stride 1 --version "v1"
    python scripts/room_label.py --dataset_dir "$DATASET_DIR" --gt_dir "/scratch/yash_mehan/mp3d_gt_instance_labelled/" --version "objs_v2" --updated_gt_dir "/scratch/kumaraditya_gupta/mp3d_gt_updated"
done
