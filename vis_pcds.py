import pickle
import open3d as o3d
import numpy as np
import torch
import torch.nn.functional as F
import open_clip
import matplotlib

CLIP_SIM_THRESHOLD = 0.85


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
                                   for node_id, _, _ in point_clouds]).to("cuda")
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
            pcd.colors = o3d.utility.Vector3dVector(unique_color)
        else:
            colors = np.array([0.5, 0.5, 0.5])
            pcd.colors = o3d.utility.Vector3dVector(colors)

        vis.update_geometry(pcd)


def clip_similarity_callback(vis):
    text = input("Enter the text to find similarity: ")
    text_queries = [text]
    text_queries_tokenized = clip_tokenizer(text_queries).to("cuda")
    text_features = clip_model.encode_text(text_queries_tokenized)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    text_features = text_features.squeeze()

    matching_count = 0  # Counter for matching objects

    for node_id, pcd, _ in point_clouds:
        clip_embed = scene_obj_nodes[node_id]['clip_embed']
        similarity = torch.nn.functional.cosine_similarity(
            torch.tensor(clip_embed, device=device),
            text_features,
            dim=-1
        ).item()

        if similarity < CLIP_SIM_THRESHOLD:
            colors = np.array([0.5, 0.5, 0.5])
        else:
            colors = np.random.rand(3)
            matching_count += 1  # Increment the counter if it's a match

        pcd.colors = o3d.utility.Vector3dVector(colors)
        vis.update_geometry(pcd)

    print(f"{matching_count} objects match the text '{text}'")

# Key callbacks


def key_callback(vis, action, mods):
    if action != o3d.visualization.KeyActionType.KEY_DOWN:
        return False

    if vis.is_key_pressed("f"):
        # Your existing clip_similarity_callback
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
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k")
clip_model = clip_model.to(device)
clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
print("Done initializing CLIP model.")

# Load the scene object nodes
scene_obj_nodes_path = "/path/to/scene_obj_nodes.pkl"
with open(scene_obj_nodes_path, "rb") as f:
    scene_obj_nodes = pickle.load(f)

# Initialize the visualizer
vis = o3d.visualization.Visualizer()
vis.create_window()

# Load point clouds and store original colors
point_clouds = []
for node_id, node_data in scene_obj_nodes.items():
    pcd = o3d.io.read_point_cloud(node_data['pcd'])
    original_colors = np.asarray(pcd.colors)
    point_clouds.append((node_id, pcd, original_colors))
    vis.add_geometry(pcd)

# Register key callback
vis.register_key_callback(key_callback)

# Run the visualizer
vis.run()
vis.destroy_window()
