# import open3d as o3d

import os
import gc
import numpy as np
import random
import cv2
from tqdm import tqdm
import pickle
from collections import Counter

from scipy.spatial.transform import Rotation as R

import open3d as o3d


DATASET_DIRS = [
    "/scratch/kumaraditya_gupta/Datasets/mp3d_test/q9vSo1VnCiC/sequence1/",
    "/scratch/kumaraditya_gupta/Datasets/mp3d_test/sT4fr6TAbpF/sequence1/",
    "/scratch/kumaraditya_gupta/Datasets/mp3d_train/Pm6F8kyY3z2/sequence2/",
    "/scratch/kumaraditya_gupta/Datasets/mp3d_train/cV4RVeZvu5T/sequence2/",
    "/scratch/kumaraditya_gupta/Datasets/mp3d_train/e9zR4mvMWw7/sequence2/",
    "/scratch/kumaraditya_gupta/Datasets/mp3d_train/jh4fc5c5qoQ/sequence2/",
    "/scratch/kumaraditya_gupta/Datasets/mp3d_train/kEZ7cmS4wCh/sequence2/",
    "/scratch/kumaraditya_gupta/Datasets/mp3d_train/sKLMLpTHeUy/sequence2/",
]


def get_pose(img_name, pose_dir):
    pose_path = os.path.join(pose_dir, img_name + ".txt")

    # check if the pose file exists, if it doesn't, return None
    # [x, y, z, qx, qy, qz, qw]
    if not os.path.exists(pose_path):
        return None

    with open(pose_path, "r") as f:
        pose = f.read().split()
        pose = np.array(pose).astype(np.float32)

        # change pose from [x, y, z, qw, qx, qy, qz] to [x, y, z, qx, qy, qz, qw]
        pose = np.concatenate((pose[:3], pose[4:], pose[3:4]))
    return pose


def create_pcd_from_rgbd(img_files_list, imgs_dir, depth_dir, pose_dir):
    pcd_global = o3d.geometry.PointCloud()

    # For each image, load the RGB-D image and transform the point cloud to the global frame
    for i, img_file in enumerate(tqdm(img_files_list)):
        # Load RGB and Depth images
        img_id = os.path.splitext(img_file)[0]
        img_path = os.path.join(imgs_dir, img_file)
        rgb_image = o3d.io.read_image(img_path)

        depth_path = os.path.join(depth_dir, img_id + ".png")
        depth_image = o3d.io.read_image(depth_path)

        pose = get_pose(img_id, pose_dir)

        intrinsics = o3d.camera.PinholeCameraIntrinsic(
            900,  # width
            900,  # height
            450.0,  # fx
            450.0,  # fy
            450.0,  # cx
            450.0,  # cy
        )

        # Generate point cloud from RGB-D image
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_image,
            depth_image,
            depth_scale=655.35,
            depth_trunc=10.0,
            convert_rgb_to_intensity=False,
        )
        # o3d.visualization.draw_geometries([rgbd_image])
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)
        voxel_size = 0.05  # adjust this value to change the number of points
        pcd = pcd.voxel_down_sample(voxel_size)

        # Parse the pose [x, y, z, qx, qy, qz, qw]
        pos = np.array(pose[:3], dtype=float).reshape((3, 1))
        quat = pose[3:]
        rot = R.from_quat(quat).as_matrix()

        # Apply rotation correction, to match the orientation z: backward, y: upward, and x: right
        rot_ro_cam = np.eye(3)
        rot_ro_cam[1, 1] = -1
        rot_ro_cam[2, 2] = -1

        combined_rot = rot @ rot_ro_cam

        cam_height = 1.50
        pos[1] += cam_height

        # Create the pose matrix
        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = combined_rot
        pose_matrix[:3, 3] = pos.reshape(-1)

        pcd.transform(pose_matrix)

        # Additional rotation to get the points in the correct orientation
        new_rot = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
        new_rot_matrix = np.eye(4)
        new_rot_matrix[:3, :3] = new_rot

        # Apply new_rot to the point cloud using the transform function
        pcd.transform(new_rot_matrix)

        pcd = pcd.voxel_down_sample(voxel_size=0.05)

        # Add the point cloud to the global point cloud
        pcd_global += pcd

    pcd_global = pcd_global.voxel_down_sample(voxel_size=0.05)
    return pcd_global


def main():

    for dataset in DATASET_DIRS:
        dataset_path = dataset

        imgs_dir = os.path.join(dataset_path, "color/")
        depth_dir = os.path.join(dataset_path, "depth/")
        pose_dir = os.path.join(dataset_path, "pose/")
        # save_dir = os.path.join(dataset_path, "output_v1/")
        save_dir = dataset_path

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        img_files_list = [
            f for f in os.listdir(imgs_dir) if os.path.isfile(os.path.join(imgs_dir, f))
        ]
        img_files_list = sorted(img_files_list, key=lambda x: int(x.split(".")[0]))

        stride = 2
        img_files_list = img_files_list[::stride]

        pcd_global = create_pcd_from_rgbd(img_files_list, imgs_dir, depth_dir, pose_dir)

        o3d.io.write_point_cloud(
            os.path.join(save_dir, "pointcloud_aligned.ply"), pcd_global
        )


if __name__ == "__main__":
    main()
