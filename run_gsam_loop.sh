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
    "/scratch/kumaraditya_gupta/Datasets/mp3d_train/sKLMLpTHeUy/sequence2/"
)

# Loop through each dataset directory and run the script
for DATASET_DIR in "${DATASET_DIRS[@]}"
do
    python scripts/gsam_bg_classes.py --weights_dir "/scratch/kumaraditya_gupta/checkpoints" --dataset_dir "$DATASET_DIR" --stride 2 --box_threshold 0.35 --text_threshold 0.35 --version "v2"
done
