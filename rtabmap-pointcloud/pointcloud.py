import numpy as np
import open3d as o3d
import pandas as pd
import yaml
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# Loading data from text file into pandas dataframe
data = pd.read_csv(
    "rtabmap_pointcloud/poses_cam_id.txt",
    delim_whitespace=True,
    comment="#",
    names=["timestamp", "x", "y", "z", "qx", "qy", "qz", "qw", "id"],
)

# Create an empty point cloud
pcd_global = o3d.geometry.PointCloud()

# For each pose
for i, row in tqdm(data.iterrows()):
    # Load RGB and Depth images
    rgb_image = o3d.io.read_image(f'rtabmap_pointcloud/imgs/rgb/{int(row["id"])}.png')
    depth_image = o3d.io.read_image(
        f'rtabmap_pointcloud/imgs/depth/{int(row["id"])}.png'
    )

    depth_scale = 0.001
    depth_data = np.asarray(depth_image) * depth_scale
    depth_image = o3d.geometry.Image(float(depth_data))

    # Load camera intrinsics from the calibration file
    with open(f'rtabmap_pointcloud/imgs/calib/{int(row["id"])}.yaml', "r") as f:
        calibration = yaml.safe_load(f)
    camera_matrix = calibration["camera_matrix"]["data"]
    camera_matrix = np.array(camera_matrix).reshape((3, 3))
    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        640,
        480,
        camera_matrix[0, 0],
        camera_matrix[1, 1],
        camera_matrix[0, 2],
        camera_matrix[1, 2],
    )

    # Generate point cloud from RGB-D image
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_image, depth_image
    )
    o3d.visualization.draw_geometries([rgbd_image])
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)
    # o3d.io.write_point_cloud("rtabmap_pointcloud/single_pcd.ply", pcd)
    # Sample points
    # voxel_size = 0.001  # adjust this value to change the number of points
    # pcd = pcd.voxel_down_sample(voxel_size)

    # Transform point cloud using camera pose
    pose = np.eye(4)
    pose[:3, :3] = R.from_quat([row["qx"], row["qy"], row["qz"], row["qw"]]).as_matrix()
    pose[:3, 3] = [row["x"], row["y"], row["z"]]
    pcd.transform(pose)
    o3d.visualization.draw_geometries([pcd])
    # print(pcd)

    # Add the point cloud to the global point cloud
    pcd_global += pcd

pcd_global = pcd_global.voxel_down_sample(voxel_size=0.05)
print(pcd_global)
# Visualize the global point cloud
o3d.visualization.draw_geometries([pcd_global])
o3d.io.write_point_cloud("rtabmap_pointcloud/pointcloud.ply", pcd_global)
