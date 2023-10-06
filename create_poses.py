import os
from tqdm import tqdm

poses_file_path = "/scratch/kumaradi.gupta/run_ipad/poses.txt"
output_path = "/scratch/kumaradi.gupta/run_ipad/pose/"

poses_file = open(poses_file_path, "r")
poses_file.readline()

for line in tqdm(poses_file):
    data = line.split(" ")
    id = data[8]
    # id has a \n at the end
    id = id[:-1]
    output_file = open(output_path + id + ".txt", "a")
    output_file.write(data[1] + " " + data[2] + " " + data[3] + " " +
                      data[4] + " " + data[5] + " " + data[6] + " " + data[7])
    output_file.close()

poses_file.close()
