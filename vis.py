import os
import numpy as np
import open3d as o3d


def get_pose(img_name):
    pose_dir = "/scratch/kumaradi.gupta/run_kinect_wheel_1/pose/"
    pose_path = os.path.join(pose_dir, img_name + '.txt')

    # check if the pose file exists, if it doesn't, return None
    # [x, y, z, qx, qy, qz, qw]
    if not os.path.exists(pose_path):
        return None

    with open(pose_path, 'r') as f:
        pose = f.read().split()
        pose = np.array(pose).astype(np.float32)
    return pose


def visualize_and_capture(img_id, scene_obj_nodes, params):

    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)

    # Clear existing geometries
    vis.clear_geometries()

    # Add point clouds to the visualizer
    for node_id, node_data in scene_obj_nodes.items():
        pcd = node_data['pcd']
        vis.add_geometry(pcd)

    W, H = params['img_size']

    cam_pose = get_pose(img_id)
    # Extract translation and quaternion
    t = cam_pose[0:3]
    q = cam_pose[3:]

    # Create 4x4 transformation matrix
    R = o3d.geometry.get_rotation_matrix_from_quaternion(q)
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = t

    # Set camera parameters using Kinect camera matrix
    K = params['cam_mat']
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(W, H, fx, fy, cx, cy)  # Assuming 1920x1080 resolution

    # Set the camera pose
    view_control = vis.get_view_control()
    view_control.convert_from_pinhole_camera_parameters(T, intrinsic)

    # Capture and save image
    image = vis.capture_screen_float_buffer(do_render=True)
    o3d.io.write_image(os.path.join(params['save_folder'], f"{img_id}.png"), np.asarray(image)*255)
