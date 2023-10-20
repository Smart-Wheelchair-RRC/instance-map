import pickle
import open3d as o3d
import numpy as np
import torch
import torch.nn.functional as F
import open_clip
import matplotlib
from collections import Counter
import random

CLIP_SIM_THRESHOLD = 0.80

def generate_pastel_color():
    # generate (r, g, b) tuple of random numbers between 0.5 and 1, truncate to 2 decimal places
    r = round(random.uniform(0.4, 1), 2)
    g = round(random.uniform(0.4, 1), 2)
    b = round(random.uniform(0.4, 1), 2)
    color = np.array([r, g, b])
    return color

def reset_colors(vis):
    for _, pcd, original_color, _, _ in point_clouds:
        pcd.paint_uniform_color(original_color)
        vis.update_geometry(pcd)
    return

def color_by_clip_sim(vis):
    text_query = input("Enter your query: ")
    text_queries = [text_query]

    text_queries_tokenized = clip_tokenizer(text_queries).to(device)
    text_query_ft = clip_model.encode_text(text_queries_tokenized)
    text_query_ft = text_query_ft / text_query_ft.norm(dim=-1, keepdim=True)
    text_query_ft = text_query_ft.squeeze()

    objects_clip_fts = torch.stack([torch.tensor(scene_obj_nodes[node_id]['clip_embed'])
                                   for node_id, _, _, _, _ in point_clouds]).to(device)
    similarities = F.cosine_similarity(text_query_ft.unsqueeze(0), objects_clip_fts, dim=-1)
    normalized_similarities = (similarities - similarities.min()) / (similarities.max() - similarities.min())

    cmap = matplotlib.colormaps.get_cmap("turbo")
    similarity_colors = cmap(normalized_similarities.detach().cpu().numpy())[..., :3]

    for i, (_, pcd, _, _, _) in enumerate(point_clouds):
        pcd.colors = o3d.utility.Vector3dVector(
            np.tile(similarity_colors[i], (len(pcd.points), 1))
        )
        vis.update_geometry(pcd)
    return


def instance_coloring_callback(vis):
    target_node_id = input("Enter the node_id to color: ")

    try:
        target_node_id = int(target_node_id)
    except ValueError:
        print("Invalid node_id. Please enter an integer.")
        return

    for node_id, pcd, _, _, _ in point_clouds:
        if node_id == target_node_id:
            unique_color = generate_pastel_color()
            pcd.paint_uniform_color(unique_color)
        else:
            colors = np.array([0.5, 0.5, 0.5])
            pcd.paint_uniform_color(colors)

        vis.update_geometry(pcd)
    return


def clip_similarity_find_obj(vis):
    text = input("Enter the text to find similarity: ")
    text_queries = [text]
    text_queries_tokenized = clip_tokenizer(text_queries).to(device)
    text_features = clip_model.encode_text(text_queries_tokenized)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    text_features = text_features.squeeze()

    ground_truths_node_ids = []
    ground_truth_tag_ids = []
    highlighted_ids = []

    matching_count = 0  # Counter for matching objects
    objects_clip_fts = torch.stack([torch.tensor(scene_obj_nodes[node_id]['clip_embed'])
                                   for node_id, _, _, _, _ in point_clouds]).to(device)
    similarities = F.cosine_similarity(text_features.unsqueeze(0), objects_clip_fts, dim=-1)
    normalized_similarities = (similarities - similarities.min()) / (similarities.max() - similarities.min())

    # sort the normalized similarities and print them
    sorted_similarities, sorted_indices = torch.sort(normalized_similarities, descending=True)
    # print("Sorted similarities: ", sorted_similarities)

    # color the objects with similarity greater than CLIP_SIM_THRESHOLD
    for i, (node_id, pcd, color, gt_tag, gt_tag_id) in enumerate(point_clouds):
        if normalized_similarities[i] > CLIP_SIM_THRESHOLD:
            matching_count += 1
            highlighted_ids.append(node_id)
            pcd.paint_uniform_color(generate_pastel_color())
        else:
            pcd.paint_uniform_color([0.5, 0.5, 0.5])

        vis.update_geometry(pcd)

        for tag in gt_tag:
            if tag == text:
                ground_truths_node_ids.append(node_id)
                ground_truth_tag_ids.append(gt_tag_id)
                break

    gt_tag_id_counter = Counter(ground_truth_tag_ids)
    ground_truths_count = len(gt_tag_id_counter)

    # Get FP and FN from highlighted_ids and ground_truths
    FP = [node_id for node_id in highlighted_ids if node_id not in ground_truths_node_ids]
    FN = [node_id for node_id in ground_truths_node_ids if node_id not in highlighted_ids]

    print("------------------------------------")
    print(f"Ground truth count: {ground_truths_count}")
    print("------------------------------------")
    print(f'GT Node IDs: {ground_truths_node_ids}')
    print(f'Highlighted IDs: {highlighted_ids}')
    print("------------------------------------")
    print(f"Ground truth count (including ghosts): {len(ground_truths_node_ids)}")
    print(f"Detected object count: {matching_count}")
    print(f"False Positives: {len(FP)}")
    print(f"False Negatives: {len(FN)}")
    print("------------------------------------")
    return


# Initialize the CLIP model
clip_model_name = "ViT-H-14"
print("Initializing CLIP model...")
device = "cpu"
if clip_model_name == "ViT-H-14":
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k")
    clip_model = clip_model.to(device)
    clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
elif clip_model_name == "ViT-B-32":
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-B-32", "laion2b_s34b_b79k")
    clip_model = clip_model.to(device)
    clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
else:
    raise NotImplementedError(f"CLIP model {clip_model_name} not implemented.")
print("Done initializing CLIP model.")

# Load the scene object nodes
all_datasets_path = "/Users/kumaraditya/Desktop/Datasets"
dataset_path = f"{all_datasets_path}/run_kinect_wheel_2/output_v1.2"
scene_obj_nodes_path = f"{dataset_path}/scene_obj_nodes.pkl"
with open(scene_obj_nodes_path, "rb") as f:
    scene_obj_nodes = pickle.load(f)

# node_ids_to_remove = [2294, 3045, 1610, 208, 538, 1157]
# for node_id in node_ids_to_remove:
#     if node_id in scene_obj_nodes:
#         del scene_obj_nodes[node_id]

# Initialize the visualizer
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(window_name="Point Cloud Visualizer", width=1280, height=720)

# Change the background color
view_control = vis.get_view_control()
opt = vis.get_render_option()
opt.background_color = np.asarray([0.1, 0.1, 0.1])  # Setting to dark gray

# Load point clouds and store original colors
point_clouds = []
for node_id, node_data in scene_obj_nodes.items():
    pcd_path = node_data['pcd']
    pcd_path = pcd_path.replace('/scratch/kumaradi.gupta/Datasets', all_datasets_path)
    pcd = o3d.io.read_point_cloud(pcd_path)
    pcd = pcd.voxel_down_sample(voxel_size=0.05)
    original_color = np.asarray(pcd.colors)[0].tolist()

    gt_tag, gt_tag_id = node_data['gt_tag'], node_data['gt_tag_id']
    point_clouds.append((node_id, pcd, original_color, gt_tag, gt_tag_id))
    vis.add_geometry(pcd)

# Register key callback
vis.register_key_callback(ord("F"), clip_similarity_find_obj)
vis.register_key_callback(ord("I"), instance_coloring_callback)
vis.register_key_callback(ord("G"), color_by_clip_sim)
vis.register_key_callback(ord("R"), reset_colors)

# Run the visualizer
vis.run()
# vis.destroy_window()
