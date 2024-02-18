#!/bin/bash

# Define the array of dataset directories
DATASET_DIRS=(
    "/scratch/kumaraditya_gupta/Datasets/mp3d_test/RPmz2sHmrrY/sequence4/"
    "/scratch/kumaraditya_gupta/Datasets/mp3d_test/ac26ZMwG7aT/sequence1/"
    "/scratch/kumaraditya_gupta/Datasets/mp3d_test/q9vSo1VnCiC/sequence1/"
    "/scratch/kumaraditya_gupta/Datasets/mp3d_test/sT4fr6TAbpF/sequence1/"
    "/scratch/kumaraditya_gupta/Datasets/mp3d_train/Pm6F8kyY3z2/sequence2/"
    "/scratch/kumaraditya_gupta/Datasets/mp3d_train/cV4RVeZvu5T/sequence2/"
    "/scratch/kumaraditya_gupta/Datasets/mp3d_train/e9zR4mvMWw7/sequence2/"
    "/scratch/kumaraditya_gupta/Datasets/mp3d_train/jh4fc5c5qoQ/sequence2/"
    "/scratch/kumaraditya_gupta/Datasets/mp3d_train/kEZ7cmS4wCh/sequence2/"
    "/scratch/kumaraditya_gupta/Datasets/mp3d_train_2/1LXtFkjw3qL/sequence1/"
    "/scratch/kumaraditya_gupta/Datasets/mp3d_train_2/1pXnuDYAj8r/sequence2/"
    "/scratch/kumaraditya_gupta/Datasets/mp3d_train_2/29hnd4uzFmX/sequence1/"
    "/scratch/kumaraditya_gupta/Datasets/mp3d_train_2/2azQ1b91cZZ/sequence4/"
    "/scratch/kumaraditya_gupta/Datasets/mp3d_train_2/2n8kARJN3HM/sequence1/"
    "/scratch/kumaraditya_gupta/Datasets/mp3d_train_2/2t7WUuJeko7/sequence1/"
    "/scratch/kumaraditya_gupta/Datasets/mp3d_train_2/5LpN3gDmAk7/sequence1/"
    "/scratch/kumaraditya_gupta/Datasets/mp3d_train_2/5ZKStnWn8Zo/sequence2/"
    "/scratch/kumaraditya_gupta/Datasets/mp3d_train_2/5q7pvUzZiYa/sequence1/"
)

# Loop through each dataset directory and run the script
for DATASET_DIR in "${DATASET_DIRS[@]}"
do
    python scripts/room_label.py --dataset_dir "$DATASET_DIR" --gt_dir "/scratch/yash_mehan/mp3d_gt_instance_labelled/" --version "output_objs_v1"
done
