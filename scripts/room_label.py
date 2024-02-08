import numpy as np
import os
import pickle
import argparse
import random
from tqdm import tqdm

import cv2
import open3d as o3d
from scipy.spatial import cKDTree

parser = argparse.ArgumentParser(description="Script parameters")

parser.add_argument(
    "--dataset_dir",
    type=str,
    default="/scratch/kumaraditya_gupta/Datasets/mp3d_train/sKLMLpTHeUy/sequence2/",
    help="Directory for dataset",
)
parser.add_argument(
    "--gt_dir",
    type=str,
    default="/scratch/yash_mehan/mp3d_gt_instance_labelled/",
    help="Directory for GT data",
)
parser.add_argument(
    "--version", type=str, default="output_objs_v1", help="Version string"
)

args = parser.parse_args()

dataset_dir = args.dataset_dir
version = args.version
output_dir = os.path.join(dataset_dir, f"output_{version}")
gt_dir = args.gt_dir

ALL_ROOM_TYPES = [
    "bathroom",
    "bedroom",
    "closet",
    "dining",
    "entryway",
    "family",
    "garage",
    "hallway",
    "library",
    "laundry",
    "kitchen",
    "living",
    "meeting",
    "lounge",
    "office",
    "porch",
    "recreation",
    "stairs",
    "toilet",
    "utility",
    "tv",
    "gym",
    "outdoor",
    "balcony",
    "other",
    "bar",
    "classroom",
    "sauna",
    "junk",
    "none",
]


def generate_pastel_color():
    # generate (r, g, b) tuple of random numbers between 0.5 and 1, truncate to 2 decimal places
    r = round(random.uniform(0.4, 1), 2)
    g = round(random.uniform(0.4, 1), 2)
    b = round(random.uniform(0.4, 1), 2)
    color = np.array([r, g, b])
    return color


def load_gt_data(gt_dir, dataset_dir):
    dataset_name = dataset_dir.split("/")[-3]
    gt_file_name = f"{dataset_name}_xyz_rgb_o_r_inst.npy"
    gt = np.load(os.path.join(gt_dir, gt_file_name))  # (N, 9)
    return gt


def get_room_masks(gt):
    # gt shape is (N, 9) and format [x, y, z, r, g, b, obj, room_label_id, instance_number]
    unique_room_ids = np.unique(
        [
            "{}_{}".format(int(room_label), int(instance_num))
            for room_label, instance_num in gt[:, [7, 8]]
        ]
    )

    # Initialize a dictionary to store mask coordinates for each room
    room_masks = {}

    # Iterate over each unique room ID and plot
    for room_id in unique_room_ids:
        room_label, instance_num = map(int, room_id.split("_"))

        # Filter points belonging to the current room
        room_points = gt[(gt[:, 7] == room_label) & (gt[:, 8] == instance_num)]
        x, y, z = room_points[:, 0], room_points[:, 1], room_points[:, 2]
        room_masks[room_id] = (x, y, z)

    return room_masks


def get_kd_tree(room_masks):
    # Create a KD-tree for each room
    kd_trees = {}
    for room_id, (x, y, z) in room_masks.items():
        points = np.stack((x, y, z), axis=-1)
        kd_trees[room_id] = cKDTree(points)

    return kd_trees


def assign_room_to_obj(obj_nodes_dict, kd_trees):
    # Initialize a dictionary to store the room assignment for each object
    room_obj_assignments = {}

    # Iterate through each object
    for node_id, node_data in tqdm(obj_nodes_dict.items()):
        obj_bbox = node_data["bbox"]  # 8x3 array of bbox points

        # Initialize a dictionary to keep the sum of minimum distances for each room
        room_distance_sum = {room_id: 0 for room_id in kd_trees.keys()}

        # Iterate through the 8 points of the bounding box
        for point in obj_bbox:
            # Initialize a dictionary to keep the minimum distance for each room for this point
            room_min_distance = {}

            # Check the distance of the point to each room
            for room_id, kd_tree in kd_trees.items():
                distance, _ = kd_tree.query(
                    point[:3]
                )  # Consider x, y, z for 3D KD-tree
                room_min_distance[room_id] = distance

            # Add the minimum distance to the room's total distance sum
            for room_id, distance in room_min_distance.items():
                room_distance_sum[room_id] += distance

        # Assign the object to the room with the minimum sum of distances
        assigned_room = min(room_distance_sum, key=room_distance_sum.get)

        if assigned_room not in room_obj_assignments:
            room_obj_assignments[assigned_room] = []
        room_obj_assignments[assigned_room].append(node_id)

        node_data["room_id"] = assigned_room

        room_label_id = assigned_room.split("_")[0]
        node_data["room_label"] = ALL_ROOM_TYPES[int(room_label_id)]

    return room_obj_assignments, obj_nodes_dict


def get_top30_objs_per_room(room_obj_assignments, obj_nodes_dict):
    room_obj_assignments_top30 = {}

    for room_id, obj_ids in room_obj_assignments.items():
        if len(obj_ids) <= 30:
            room_obj_assignments_top30[room_id] = obj_ids
        else:
            obj_points_counts = []

            for obj_id in obj_ids:
                pcd_path = obj_nodes_dict[obj_id]["pcd"]
                pcd = o3d.io.read_point_cloud(pcd_path)
                obj_points_counts.append((obj_id, len(pcd.points)))

            # Sort the list based on the number of points in descending order
            obj_points_counts.sort(key=lambda x: x[1], reverse=True)
            top_30_obj_ids = [obj_id for obj_id, _ in obj_points_counts[:30]]
            room_obj_assignments_top30[room_id] = top_30_obj_ids

    return room_obj_assignments_top30


def save_roomwise_pcds(room_obj_assignments, obj_nodes_dict):
    # generate colors equal to the length of room_labels
    room_id_to_color = {}
    for room_id in room_obj_assignments.keys():
        room_id_to_color[room_id] = generate_pastel_color()

    save_path = os.path.join(output_dir, "pcds_roomwise")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    room_pcds = {}

    # iterate over obj_nodes_dict and get the pcd for each node. Save the pcd with a color based on room label
    for node_id, node_data in obj_nodes_dict.items():
        obj_pcd_path = node_data["pcd"]
        obj_pcd = o3d.io.read_point_cloud(obj_pcd_path)
        obj_room_id = node_data["room_id"]

        color = room_id_to_color[obj_room_id]
        obj_pcd.paint_uniform_color(color)
        o3d.io.write_point_cloud(os.path.join(save_path, f"{node_id}.pcd"), obj_pcd)

        if obj_room_id not in room_pcds:
            room_pcds[obj_room_id] = []
        room_pcds[obj_room_id].append(obj_pcd)

    # room_pcds contains the list of pcds for each room label
    # we can merge them and save them in a separate folder
    save_path = os.path.join(output_dir, "pcds_roomwise_merged")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for room_id, pcd_list in room_pcds.items():
        merged_pcd = o3d.geometry.PointCloud()
        for pcd in pcd_list:
            merged_pcd += pcd
        o3d.io.write_point_cloud(os.path.join(save_path, f"{room_id}.pcd"), merged_pcd)


def main():
    print(f"Working on dataset {dataset_dir} with version {version}...")

    print("Loading obj_nodes.pkl...")
    obj_nodes_path = os.path.join(output_dir, "scene_obj_nodes.pkl")
    with open(obj_nodes_path, "rb") as file:
        obj_nodes_dict = pickle.load(file)

    print("Loading img_dict.pkl...")
    img_dict_path = os.path.join(output_dir, "img_dict.pkl")
    with open(img_dict_path, "rb") as file:
        img_dict = pickle.load(file)

    gt = load_gt_data(gt_dir, dataset_dir)

    print("Getting room masks...")
    room_masks = get_room_masks(gt)

    print("Creating KD-trees for each room...")
    kd_trees = get_kd_tree(room_masks)

    print("Assigning rooms to objects...")
    room_obj_assignments, obj_nodes_dict = assign_room_to_obj(obj_nodes_dict, kd_trees)
    room_obj_assignments = get_top30_objs_per_room(room_obj_assignments, obj_nodes_dict)

    save_roomwise_pcds(room_obj_assignments, obj_nodes_dict)

    # save obj_node_dict.pkl
    with open(obj_nodes_path, "wb") as file:
        pickle.dump(obj_nodes_dict, file)

    # save room_obj_assignments.pkl
    with open(os.path.join(output_dir, "room_obj_assignments.pkl"), "wb") as file:
        pickle.dump(room_obj_assignments, file)

    print("Done!")
    return


if __name__ == "__main__":
    main()
