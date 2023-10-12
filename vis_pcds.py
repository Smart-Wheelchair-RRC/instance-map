import pickle
import open3d as o3d
import numpy as np
import torch
import torch.nn.functional as F
import open_clip
import matplotlib

CLIP_SIM_THRESHOLD = 0.3


def reset_colors(vis):
    for node_id, pcd, original_colors in point_clouds:
        pcd.colors = o3d.utility.Vector3dVector(original_colors)
        vis.update_geometry(pcd)


def color_by_clip_sim(vis):
    text_query = input("Enter your query: ")
    text_queries = [text_query]

    text_queries_tokenized = clip_tokenizer(text_queries).to(device)
    text_query_ft = clip_model.encode_text(text_queries_tokenized)
    text_query_ft = text_query_ft / text_query_ft.norm(dim=-1, keepdim=True)
    text_query_ft = text_query_ft.squeeze()

    objects_clip_fts = torch.stack([torch.tensor(scene_obj_nodes[node_id]['clip_embed'])
                                   for node_id, _, _ in point_clouds]).to(device)
    similarities = F.cosine_similarity(text_query_ft.unsqueeze(0), objects_clip_fts, dim=-1)
    normalized_similarities = (similarities - similarities.min()) / (similarities.max() - similarities.min())

    cmap = matplotlib.colormaps.get_cmap("turbo")
    similarity_colors = cmap(normalized_similarities.detach().cpu().numpy())[..., :3]

    for i, (_, pcd, _) in enumerate(point_clouds):
        pcd.colors = o3d.utility.Vector3dVector(
            np.tile(similarity_colors[i], (len(pcd.points), 1))
        )
        vis.update_geometry(pcd)


def instance_coloring_callback(vis):
    target_node_id = input("Enter the node_id to color: ")

    for node_id, pcd, _ in point_clouds:
        if node_id == target_node_id:
            unique_color = np.random.rand(3)
            unique_color = np.tile(unique_color, (len(pcd.points), 1))
            pcd.colors = o3d.utility.Vector3dVector(unique_color)
        else:
            colors = np.array([0.5, 0.5, 0.5])
            colors = np.tile(colors, (len(pcd.points), 1))
            pcd.colors = o3d.utility.Vector3dVector(colors)

        vis.update_geometry(pcd)


def clip_similarity_find_obj(vis):
    text = input("Enter the text to find similarity: ")
    text_queries = [text]
    text_queries_tokenized = clip_tokenizer(text_queries).to(device)
    text_features = clip_model.encode_text(text_queries_tokenized)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    text_features = text_features.squeeze()

    matching_count = 0  # Counter for matching objects
    sim_scores = []

    for node_id, pcd, _ in point_clouds:
        clip_embed = scene_obj_nodes[node_id]['clip_embed']
        similarity = torch.nn.functional.cosine_similarity(
            torch.tensor(clip_embed, device=device),
            text_features,
            dim=-1
        ).item()

        sim_scores.append(similarity)

        if similarity < CLIP_SIM_THRESHOLD:
            colors = np.array([0.5, 0.5, 0.5])
        else:
            colors = np.random.rand(3)
            matching_count += 1  # Increment the counter if it's a match

        expanded_colors = np.tile(colors, (len(pcd.points), 1))
        pcd.colors = o3d.utility.Vector3dVector(expanded_colors)
        vis.update_geometry(pcd)
    
    # sort the scores in descending order
    sim_scores = np.array(sim_scores)
    sorted_indices = np.argsort(sim_scores)[::-1]
    sim_scores = sim_scores[sorted_indices]
    print(sim_scores)

    print(f"{matching_count} objects match the text '{text}'")

# Key callbacks


def key_callback(vis, action, mods):
    if action != o3d.visualization.KeyActionType.KEY_DOWN:
        return False

    if vis.is_key_pressed("f"):
        clip_similarity_find_obj(vis)
        return True

    if vis.is_key_pressed("i"):
        instance_coloring_callback(vis)
        return True

    if vis.is_key_pressed("g"):
        color_by_clip_sim(vis)
        return True

    if vis.is_key_pressed("r"):
        reset_colors(vis)
        return True

    return False


# Initialize the CLIP model
print("Initializing CLIP model...")
device = "cpu"
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k")
clip_model = clip_model.to(device)
clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
print("Done initializing CLIP model.")

# Load the scene object nodes
scene_obj_nodes_path = "/home/interns/Desktop/scene_obj_nodes.pkl"
with open(scene_obj_nodes_path, "rb") as f:
    scene_obj_nodes = pickle.load(f)

node_ids_to_remove = [2294, 3045, 1610, 208, 538, 1157]
for node_id in node_ids_to_remove:
    if node_id in scene_obj_nodes:
        del scene_obj_nodes[node_id]

# Initialize the visualizer
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(window_name="Point Cloud Visualizer", width=1280, height=720)

# Load point clouds and store original colors
point_clouds = []
for node_id, node_data in scene_obj_nodes.items():
    pcd_path = node_data['pcd']
    pcd_path = pcd_path.replace('/scratch/kumaradi.gupta/Datasets/run_kinect_wheel_2/pcds', '/home/interns/Desktop/pcds')
    pcd = o3d.io.read_point_cloud(pcd_path)
    pcd = pcd.voxel_down_sample(voxel_size=0.05)
    original_colors = np.asarray(pcd.colors)
    point_clouds.append((node_id, pcd, original_colors))
    vis.add_geometry(pcd)

# Register key callback
vis.register_key_callback(ord("F"), clip_similarity_find_obj)
vis.register_key_callback(ord("I"), instance_coloring_callback)
vis.register_key_callback(ord("G"), color_by_clip_sim)
vis.register_key_callback(ord("R"), reset_colors)

# Run the visualizer
vis.run()
# vis.destroy_window()
