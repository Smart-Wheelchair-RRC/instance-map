import os

def delete_first_two_lines(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    with open(file_path, 'w') as f:
        f.writelines(lines[2:])

def main(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith(".yaml"):
            file_path = os.path.join(directory_path, filename)
            delete_first_two_lines(file_path)

if __name__ == "__main__":
    directory_path = "rtabmap_pointcloud/imgs/calib"
    main(directory_path)
