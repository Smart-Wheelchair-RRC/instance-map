import os
import numpy as np
from tqdm import tqdm
import pickle

import open3d as o3d

import torch

from utils import *
from vis_live import visualize_and_capture
from scene_graph import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

params = {'init_img_id': '835',  # initialize the scene with this image

          'cam_mat': get_kinect_cam_mat(),  # camera matrix
          'device': device,  # device to run the code on

          'img_size': (2048, 1536),  # image size
          'weights_dir': "/scratch/kumaradi.gupta/checkpoints",
          'imgs_dir': "/scratch/kumaradi.gupta/run_kinect_wheel_1/rgb",
          'depth_dir': "/scratch/kumaradi.gupta/run_kinect_wheel_1/depth/",
          'pose_dir': "/scratch/kumaradi.gupta/run_kinect_wheel_1/pose/",
          'save_dir': '/scratch/kumaradi.gupta/kinect_output_imgs/',
          'img_dict_dir': "/scratch/kumaradi.gupta/kinect_img_dict.pkl",

          'voxel_size': 0.025,  # voxel size for downsampling point clouds
          'eps': 0.075,  # eps for DBSCAN
          'min_samples': 10,  # min_samples for DBSCAN
          'embed_type': 'dino_embed',  # embedding type to use for visual similarity

          'sim_threshold': 0.95,  # threshold for aggregate similarity while running update_scene_nodes
          'alpha': 0,  # weight for visual similarity while computing aggregate similarity

          'merge_overlap_method': 'nnratio',  # metric to use for merging overlapping nodes
          'merge_overlap_thresh': 0.95,  # threshold for overlap ratio while merging nodes in scene
          'merge_visual_thresh': 0.75,  # threshold for visual similarity while merging nodes in scene
          'merge_overall_thresh': 0.95,  # threshold for overall similarity while merging nodes in scene

          'obj_min_points': 30,  # minimum number of points in a node while filtering scene nodes
          'obj_min_detections': 3,  # minimum number of detections in a node while filtering scene nodes

          'icp_threshold_multiplier': 1.5,  # threshold multiplier for ICP
          'icp_max_iter': 2000  # maximum number of iterations for ICP
          }

with open(params['img_dict_dir'], 'rb') as f:
    img_dict = pickle.load(f)

# for id in img_dict.keys():
#     print(id, len(img_dict[id]['objs']))

# Initialize the scene with the first image
print("Initializing the scene...")
scene_obj_nodes = init_scene_nodes(img_dict[params['init_img_id']], params['init_img_id'], params)
print("Number of objs in the scene: ", len(scene_obj_nodes))
visualize_and_capture(params['init_img_id'], scene_obj_nodes, params)

# # Iterate over the images and update the scene
# print("Iterating and updating the scene...")
# counter = 0
# for img_id, img_data in tqdm(img_dict.items()):
#     if len(img_data['objs']) == 0 or img_id == params['init_img_id']:
#         continue

#     scene_obj_nodes = update_scene_nodes(img_id, img_data, scene_obj_nodes, params)

#     counter += 1
#     if counter % 25 == 0:
#         scene_obj_nodes = merge_scene_nodes(scene_obj_nodes, params)

#     # visualize_and_capture(img_id, scene_obj_nodes, params)

# scene_obj_nodes = filter_scene_nodes(scene_obj_nodes, params)
# print("Number of objs in the scene: ", len(scene_obj_nodes))
# for node_id, node_data in scene_obj_nodes.items():
#     o3d.io.write_point_cloud(f'/scratch/kumaradi.gupta/kinect_pcds/{node_id}.pcd', node_data['pcd'])
