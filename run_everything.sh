#!/bin/bash

# Define the array of dataset directories
DATASET_DIRS=(
    "/scratch/kumaraditya_gupta/Datasets/mp3d_test/D7G3Y4RVNrH/sequence2/"
    "/scratch/kumaraditya_gupta/Datasets/mp3d_test/GdvgFV5R1Z5/sequence2/"
    "/scratch/kumaraditya_gupta/Datasets/mp3d_test/gTV8FGcVJC9/sequence1/"
    "/scratch/kumaraditya_gupta/Datasets/mp3d_test/QUCTc6BB5sX/sequence2/"
    "/scratch/kumaraditya_gupta/Datasets/mp3d_test/p5wJjkQkbXX/sequence2/"
    "/scratch/kumaraditya_gupta/Datasets/mp3d_test/EU6Fwq7SyZv/sequence2/"
    "/scratch/kumaraditya_gupta/Datasets/mp3d_test/oLBMNvg9in8/sequence2/"
    "/scratch/kumaraditya_gupta/Datasets/mp3d_test/E9uDoFAP3SH/sequence1/"
    "/scratch/kumaraditya_gupta/Datasets/mp3d_test/q9vSo1VnCiC/sequence3/"
)

# Loop through each dataset directory and run the script
for DATASET_DIR in "${DATASET_DIRS[@]}"
do
    python scripts/create_poses.py --dataset_dir "$DATASET_DIR"
    python scripts/rgbd_pointcloud.py --dataset_dir "$DATASET_DIR" --stride 1
    python scripts/gsam.py --weights_dir "/scratch/kumaraditya_gupta/checkpoints" --dataset_dir "$DATASET_DIR" --stride 1 --box_threshold 0.35 --text_threshold 0.35 --version "output_objs_v1"
    python scripts/main.py --weights_dir "/scratch/kumaraditya_gupta/checkpoints" --dataset_dir "$DATASET_DIR" --stride 1 --version "output_objs_v1"
    python scripts/room_label.py --dataset_dir "$DATASET_DIR" --gt_dir "/scratch/yash_mehan/mp3d_gt_instance_labelled/" --version "output_objs_v1"
done
