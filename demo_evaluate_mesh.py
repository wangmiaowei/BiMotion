import numpy as np
import open3d as o3d
import os

from models.VAEs.vae_bspline import VariationalAutoEncoder
from models.Diffusions.transformer_modified import BsplineVarianceDiT

from omegaconf import OmegaConf
import torch

from models.b_spline_utils import (
    make_clamped_uniform_knots,
    bspline_basis_matrix_cox,
    B_spline_compute
)


from utils.rf_diffusion import rf_sample

# Configuration files
vae_config_path = "configs/vae_animate.yml"
diffusion_config_path = "configs/diffusion.yml"




# ------------------------------------------------------------
# Normalize mesh vertices and compute normals
# ------------------------------------------------------------
def normalize_vertices_faces(mesh_or_vertices, triangles=None, scale=1.8):
    """
    Normalize mesh to a centered bounding box with a fixed scale.

    Args:
        mesh_or_vertices:
            - open3d TriangleMesh OR
            - ndarray (N, 3) vertices
        triangles:
            Face indices (M, 3) if vertices are provided
        scale:
            Target scale factor (default = 1.8)

    Returns:
        norm_vertices : (N,3) normalized vertices
        faces         : (M,3) triangle indices
        vertex_normals: (N,3) normals of normalized mesh
        transform     : dict with parameters for inverse transform
    """

    # Accept either mesh object or raw arrays
    if isinstance(mesh_or_vertices, o3d.geometry.TriangleMesh):
        mesh = mesh_or_vertices
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles).astype(np.int64)
    else:
        vertices = np.asarray(mesh_or_vertices, dtype=np.float64)
        if triangles is None:
            raise ValueError("Triangles must be provided when input is vertices.")
        faces = np.asarray(triangles, dtype=np.int64)

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)

    # Compute bounding box
    bbox = mesh.get_axis_aligned_bounding_box()
    b_min = np.asarray(bbox.get_min_bound(), dtype=np.float64)
    b_max = np.asarray(bbox.get_max_bound(), dtype=np.float64)

    extent = float(np.max(b_max - b_min))
    center = (b_max + b_min) / 2.0
    scale_factor = float(scale) / extent

    # Normalize vertices
    norm_vertices = (vertices - center) * scale_factor

    # Build normalized mesh to compute normals
    norm_mesh = o3d.geometry.TriangleMesh()
    norm_mesh.vertices = o3d.utility.Vector3dVector(norm_vertices)
    norm_mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    norm_mesh.compute_vertex_normals()

    vertex_normals = np.asarray(norm_mesh.vertex_normals)

    transform = {
        "center": center,
        "extent": extent,
        "scale": float(scale),
        "scale_factor": scale_factor
    }

    return norm_vertices, faces, vertex_normals, transform


# ------------------------------------------------------------
# Recover original vertex scale from normalized vertices
# ------------------------------------------------------------
def denormalize_vertices(norm_vertices, transform):
    """
    Convert normalized vertices back to original space.

    Args:
        norm_vertices : (T, N, 3)
        transform     : dict with scale_factor and center

    Returns:
        orig_vertices : (T, N, 3)
    """
    scale_factor = transform["scale_factor"]
    center = transform["center"]

    orig_vertices = norm_vertices / scale_factor + center[None, None, :]
    return orig_vertices


# ------------------------------------------------------------
# Save mesh sequence to disk
# ------------------------------------------------------------
def build_save_mesh_sequence(sequence_vertices, faces, full_save_path):
    """
    Save each frame as a PLY mesh.

    Args:
        sequence_vertices : (T, N, 3)
        faces             : (M, 3)
        full_save_path    : folder path

    Returns:
        True if successful
    """
    try:
        for i in range(sequence_vertices.shape[0]):
            frame_mesh = o3d.geometry.TriangleMesh()
            frame_mesh.vertices = o3d.utility.Vector3dVector(sequence_vertices[i])
            frame_mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
            frame_mesh.compute_vertex_normals()

            o3d.io.write_triangle_mesh(
                os.path.join(full_save_path, f"frame_{i:04d}.ply"),
                frame_mesh
            )
        return True
    except Exception as e:
        print(f"Error saving mesh sequence: {e}")
        return False


# ------------------------------------------------------------
# Load checkpoint with compatibility handling
# ------------------------------------------------------------
def load_compatible_checkpoint(model, ckpt_path, device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model_weights = checkpoint['model_state_dict']
    else:
        model_weights = checkpoint

    # Remove DataParallel prefix if present
    if list(model_weights.keys())[0].startswith('module.'):
        model_weights = {k.replace('module.', ''): v for k, v in model_weights.items()}

    model.load_state_dict(model_weights)
    print("Model weights loaded successfully.")
    return model


# ------------------------------------------------------------
# Set random seed for reproducibility
# ------------------------------------------------------------
def set_seed(seed):
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



# ---------------------------------------------------------
# Argument parser
# ---------------------------------------------------------
def set_parse():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=3407)

    parser.add_argument(
        "--input_meshes_folder",
        type=str,
        default="visualize_experiments/input_meshs",
        help="Folder containing input .ply .obj files"
    )

    parser.add_argument(
        "--mesh_name",
        type=str,
        default="dragon",
        help="Mesh name without extension"
    )

    parser.add_argument("--generated_seq_length", type=int, default=20)

    parser.add_argument("--vae_path", type=str, default="")
    parser.add_argument("--dit_path", type=str, default="")

    parser.add_argument(
        "--save_path",
        type=str,
        default="visualize_experiments/results",
        help="Output folder"
    )

    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--guidance_scale", type=float, default=3.0)

    return parser.parse_args()

# -------------------------------------------------
# Utility: automatically detect mesh file (.ply or .obj)
# -------------------------------------------------
def auto_detect_mesh_path(input_meshes_folder, mesh_name):
    """
    Automatically find a mesh file with .ply or .obj extension.

    Args:
        input_meshes_folder (str): folder containing meshes
        mesh_name (str): mesh file name without extension

    Returns:
        str: full path to mesh file
    """
    ply_path = os.path.join(input_meshes_folder, mesh_name + ".ply")
    obj_path = os.path.join(input_meshes_folder, mesh_name + ".obj")

    if os.path.exists(ply_path):
        return ply_path
    elif os.path.exists(obj_path):
        return obj_path
    else:
        raise FileNotFoundError(
            f"No mesh found for {mesh_name} (.ply or .obj) in {input_meshes_folder}"
        )


# -------------------------------------------------
# Main
# -------------------------------------------------
if __name__ == "__main__":

    # -----------------------
    # Parse arguments
    # -----------------------
    opt = set_parse()

    # -----------------------
    # Device setup
    # -----------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(opt.seed)

    # -----------------------
    # Detect mesh path
    # -----------------------
    mesh_path = auto_detect_mesh_path(opt.input_meshes_folder, opt.mesh_name)

    predicted_frame_length = opt.generated_seq_length
    animation_name = opt.mesh_name
    prompt = opt.prompt

    vae_checkpoint_path = opt.vae_path
    dit_checkpoint_path = opt.dit_path

    # -----------------------
    # Load normalization statistics
    # -----------------------
    deformation_mean = torch.load("dataset/data_statistics/motion_mean.pt").float().to(device)
    deformation_std  = torch.load("dataset/data_statistics/motion_std.pt").float().to(device)
    static_mean = torch.load("dataset/data_statistics/static_mean.pt").float().to(device)
    static_std = torch.load("dataset/data_statistics/static_std.pt").float().to(device)

    # -----------------------
    # Load models
    # -----------------------
    VAE_config = OmegaConf.load(vae_config_path)
    VAE_model = VariationalAutoEncoder(**VAE_config["model"]).to(device)
    VAE_model = load_compatible_checkpoint(VAE_model, vae_checkpoint_path, device)
    VAE_model.eval()
    print("✅ DyMeshVAE loaded")

    DIT_config = OmegaConf.load(diffusion_config_path)
    diffusion_model = BsplineVarianceDiT(**DIT_config["model"]).to(device)
    diffusion_model = load_compatible_checkpoint(diffusion_model, dit_checkpoint_path, device)
    diffusion_model.eval()
    print("✅ Diffusion model loaded")

    # -----------------------
    # Output folder
    # -----------------------
    mesh_sequence_save_folder = opt.save_path
    full_save_path = os.path.join(
        mesh_sequence_save_folder,
        animation_name,
        f"Meshes_length_{predicted_frame_length}"
    )
    os.makedirs(full_save_path, exist_ok=True)

    # -----------------------
    # Load mesh
    # -----------------------
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    norm_v, faces, norm_n, transform = normalize_vertices_faces(mesh)
    print("✅ Mesh loaded and normalized")

    # -----------------------
    # Build B-spline basis
    # -----------------------
    knots = make_clamped_uniform_knots(16, 3)
    denom_clamp = 1e-6
    full_timesteps = torch.linspace(0.0, 1.0, steps=predicted_frame_length)
    B_s = bspline_basis_matrix_cox(3, knots, full_timesteps).to(device)

    # -------------------------------------------------
    # Inference
    # -------------------------------------------------
    with torch.no_grad():

        vertices_tensor = torch.tensor(norm_v, dtype=torch.float32).unsqueeze(0).to(device)
        norm_tensor = torch.tensor(norm_n, dtype=torch.float32).unsqueeze(0).to(device)
        vertices_repeat = vertices_tensor.unsqueeze(0).repeat(1, 16, 1, 1)

        kl, x, idx, pc0_embed_ori, x0, posterior = VAE_model.encode(
            vertices_tensor, norm_tensor, vertices_repeat
        )

        x0_latent, xt_latent = x.chunk(2, dim=-1)

        # Normalize latent space
        x0_latent = (x0_latent - static_mean) / static_std
        xt_latent = (xt_latent - deformation_mean) / deformation_std
        x_start = torch.cat([x0_latent, xt_latent], dim=-1)

        # Diffusion sampling
        samples = rf_sample(
            model=diffusion_model,
            shape=x_start.shape,
            text_prmpt=[prompt],
            guidance_scale=3.0,
            device=device,
            f0=x0_latent,
        )

        x0_recon = samples[:, :, :x0_latent.shape[-1]]
        xt_recon = samples[:, :, x0_latent.shape[-1]:]

        # De-normalize latents
        x0_recon = x0_recon * static_std + static_mean
        xt_recon = xt_recon * deformation_std + deformation_mean
        x = torch.cat([x0_recon, xt_recon], dim=-1)

        outputs = VAE_model.decode(x, vertices_tensor, norm_tensor, pc0_embed_ori)

        P_pred_bspline = B_spline_compute(
            B_s,
            outputs[0],
            torch.ones_like(outputs[0, :, :, 0]),
            denom_clamp
        )

        pred_deformed_pc_bspline = (
            P_pred_bspline.unsqueeze(0)
            + vertices_tensor.repeat(predicted_frame_length, 1, 1).unsqueeze(0)
        )

    # -------------------------------------------------
    # Save mesh sequence
    # -------------------------------------------------
    full_vertices_positions = torch.cat(
        [vertices_tensor.unsqueeze(1), pred_deformed_pc_bspline],
        dim=1
    ).squeeze(0).cpu().numpy()

    recovered_v = denormalize_vertices(full_vertices_positions, transform)

    flag = build_save_mesh_sequence(recovered_v, faces, full_save_path)

    if flag:
        print(f"✅ All frames saved to: {full_save_path}")
    else:
        print("❌ Error saving mesh sequence")