from omegaconf import OmegaConf
import torch
import numpy as np
from pytorch3d.structures import Meshes
import os

from models.VAEs.vae_bspline import VariationalAutoEncoder
from models.Diffusions.transformer_modified import BsplineVarianceDiT
from models.b_spline_utils import (
    make_clamped_uniform_knots,
    bspline_basis_matrix_cox,
    B_spline_compute
)

from utils.rf_diffusion import rf_sample
from utils.render import (
    clear_scene,
    import_glb,
    get_all_vertices,
    get_all_faces,
    drive_mesh_with_trajs_frames,
    drive_mesh_with_trajs_frames_five_views
)
from utils.mesh_utils import merge_vertices_with_indices


import logging
logging.getLogger("bpy").setLevel(logging.ERROR)


# Configuration files
vae_config_path = "configs/vae_animate.yml"
diffusion_config_path = "configs/diffusion.yml"


# ---------------------------------------------------------
# Normalize mesh to fit inside a unit bounding box
# ---------------------------------------------------------
def normalize_vertices_faces(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    scale: float = 1.8,
    eps: float = 1e-12,
):
    """
    Normalize a mesh to fit inside a bounding box of size `scale`.

    Args:
        vertices: (N,3) vertex positions
        faces: (F,3) face indices
        scale: target bounding box size
        eps: numerical stability

    Returns:
        norm_vertices: normalized vertices
        faces: faces (unchanged)
        vertex_normals: computed vertex normals
        transform: transformation metadata
    """

    if vertices.dim() != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must be shape (N,3)")
    if faces.dim() != 2 or faces.shape[1] != 3:
        raise ValueError("faces must be shape (F,3)")

    device = vertices.device
    dtype = vertices.dtype

    # Compute bounding box
    b_min = torch.min(vertices, dim=0).values
    b_max = torch.max(vertices, dim=0).values

    extent_vec = b_max - b_min
    extent = float(torch.max(extent_vec).item())

    # Compute scale factor
    if extent <= eps:
        scale_factor = 1.0
    else:
        scale_factor = float(scale) / extent

    # Center of bounding box
    center = (b_max + b_min) / 2.0

    # Normalize vertices
    norm_vertices = (vertices - center) * scale_factor

    # Build PyTorch3D mesh
    mesh = Meshes(
        verts=[norm_vertices],
        faces=[faces.long().to(device)]
    )

    # Compute vertex normals
    v_normals = mesh.verts_normals_packed()

    transform = {
        "center": center.detach().cpu(),
        "extent": extent,
        "scale": scale,
        "scale_factor": scale_factor
    }

    return norm_vertices, faces.long(), v_normals, transform


# ---------------------------------------------------------
# Restore normalized vertices to original space
# ---------------------------------------------------------
def denormalize_vertices(norm_vertices: torch.Tensor, transform):
    """
    Restore normalized vertices to original coordinates.
    """

    sf = float(transform["scale_factor"])
    center = torch.as_tensor(
        transform["center"],
        dtype=norm_vertices.dtype,
        device=norm_vertices.device
    )

    if norm_vertices.dim() == 2:
        return norm_vertices / sf + center
    else:
        expand_shape = [1] * (norm_vertices.dim() - 1) + [3]
        center_view = center.view(*expand_shape)
        return norm_vertices / sf + center_view


# ---------------------------------------------------------
# Load model checkpoint with compatibility handling
# ---------------------------------------------------------
def load_compatible_checkpoint(model, ckpt_path, device):

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    print(f"Loading checkpoint from: {ckpt_path}")

    checkpoint = torch.load(
        ckpt_path,
        map_location=device,
        weights_only=False
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model_weights = checkpoint["model_state_dict"]
    else:
        model_weights = checkpoint

    # Remove "module." prefix if trained with DataParallel
    if list(model_weights.keys())[0].startswith("module."):
        model_weights = {
            k.replace("module.", ""): v
            for k, v in model_weights.items()
        }

    model.load_state_dict(model_weights)

    print("Model weights loaded successfully.")

    return model


# ---------------------------------------------------------
# Fix random seed for reproducibility
# ---------------------------------------------------------
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
        "--input_glbs_folder",
        type=str,
        default="visualize_experiments/input_glbs",
        help="Folder containing input glb files"
    )

    parser.add_argument(
        "--glb_name",
        type=str,
        default="dragon",
        help="Object name without extension"
    )

    parser.add_argument("--generated_seq_length", type=int, default=16)

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

    parser.add_argument("--azi", type=float, default=0.0)
    parser.add_argument("--ele", type=float, default=0.0)

    parser.add_argument(
        "--export_format",
        type=str,
        default="none",
        choices=["none", "abc", "fbx", "glb"]
    )

    return parser.parse_args()


# ---------------------------------------------------------
# Main Inference Pipeline
# ---------------------------------------------------------
if __name__ == "__main__":

    # Parse arguments
    opt = set_parse()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fix random seed
    set_seed(opt.seed)

    import time

    # -----------------------------------------------------
    # Load normalization statistics
    # -----------------------------------------------------
    deformation_mean = torch.load(
        "dataset/data_statistics/motion_mean.pt",
        weights_only=True
    ).float().to(device)

    deformation_std = torch.load(
        "dataset/data_statistics/motion_std.pt",
        weights_only=True
    ).float().to(device)

    static_mean = torch.load(
        "dataset/data_statistics/static_mean.pt",
        weights_only=True
    ).float().to(device)

    static_std = torch.load(
        "dataset/data_statistics/static_std.pt",
        weights_only=True
    ).float().to(device)


    # -----------------------------------------------------
    # Load VAE
    # -----------------------------------------------------
    VAE_config = OmegaConf.load(vae_config_path)

    VAE_model = VariationalAutoEncoder(
        **VAE_config["model"]
    ).to(device)

    VAE_model = load_compatible_checkpoint(
        VAE_model,
        opt.vae_path,
        device
    )

    VAE_model.eval()

    print("VAE model loaded successfully!")


    # -----------------------------------------------------
    # Load Diffusion Model
    # -----------------------------------------------------
    DIT_config = OmegaConf.load(diffusion_config_path)

    diffusion_model = BsplineVarianceDiT(
        **DIT_config["model"],
    ).to(device)

    diffusion_model = load_compatible_checkpoint(
        diffusion_model,
        opt.dit_path,
        device
    )

    diffusion_model.eval()

    print("Diffusion model loaded successfully!")


    # -----------------------------------------------------
    # Build B-spline basis
    # -----------------------------------------------------
    knots = make_clamped_uniform_knots(16, 3)

    denom_clamp = 1e-6

    full_timesteps = torch.linspace(
        0.0, 1.0,
        steps=opt.generated_seq_length
    )

    B_s = bspline_basis_matrix_cox(
        3,
        knots,
        full_timesteps
    ).cuda()


    # -----------------------------------------------------
    # Create output directory
    # -----------------------------------------------------
    save_video_folder = os.path.join(
        opt.save_path,
        opt.glb_name
    )

    os.makedirs(save_video_folder, exist_ok=True)


    # -----------------------------------------------------
    # Load and preprocess mesh
    # -----------------------------------------------------
    glb_file_path = os.path.join(
        opt.input_glbs_folder,
        opt.glb_name + ".glb"
    )

    clear_scene()

    t1 = time.time()

    mesh_objects = import_glb(glb_file_path)

    all_vertices = get_all_vertices(mesh_objects)
    all_faces = get_all_faces(mesh_objects)

    merged_verts, merged_faces, all_indices = merge_vertices_with_indices(
        all_vertices,
        all_faces
    )

    norm_v, faces, norm_n, transform = normalize_vertices_faces(
        merged_verts,
        merged_faces
    )

    print("Mesh loaded and normalized successfully!")


    # -----------------------------------------------------
    # Inference
    # -----------------------------------------------------
    VAE_model.num_traj = max(
        512,
        norm_v.shape[0] // 8
    )

    with torch.no_grad():

        vertices_tensor = torch.tensor(
            norm_v,
            dtype=torch.float32
        ).clone().detach().unsqueeze(0).to(device)

        norm_tensor = torch.tensor(
            norm_n,
            dtype=torch.float32
        ).clone().detach().unsqueeze(0).to(device)

        vertices_repeat = vertices_tensor.unsqueeze(0).repeat(1, 16, 1, 1)


        # Encode
        kl, x, idx, pc0_embed_ori, x0, posterior = VAE_model.encode(
            vertices_tensor,
            norm_tensor,
            vertices_repeat
        )


        # Split latent
        x0_latent, xt_latent = x.chunk(2, dim=-1)


        # Normalize latent
        x0_latent = (x0_latent - static_mean) / static_std
        xt_latent = (xt_latent - deformation_mean) / deformation_std

        x_start = torch.cat([x0_latent, xt_latent], dim=-1)


        # Diffusion sampling
        samples = rf_sample(
            model=diffusion_model,
            shape=x_start.shape,
            text_prmpt=[opt.prompt, ""],
            guidance_scale=opt.guidance_scale,
            device=device,
            f0=x0_latent
        )


        # Recover latent
        x0_recon = samples[:, :, :x0_latent.shape[-1]]
        xt_recon = samples[:, :, x0_latent.shape[-1]:]


        # Denormalize latent
        x0_recon = x0_recon * static_std + static_mean
        xt_recon = xt_recon * deformation_std + deformation_mean

        x = torch.cat([x0_recon, xt_recon], dim=-1)


        # Decode
        outputs = VAE_model.decode(
            x,
            vertices_tensor,
            norm_tensor,
            pc0_embed_ori
        )


        # -------------------------------------------------
        # Compute B-spline trajectories
        # -------------------------------------------------
        P_pred_bspline = B_spline_compute(
            B_s,
            outputs[0, :, :, :],
            torch.ones_like(outputs[0, :, :, 0]),
            denom_clamp
        )

        pred_deformed_pc = (
            P_pred_bspline +
            vertices_tensor.repeat(opt.generated_seq_length, 1, 1)
        )

        pred_deformed_pc = torch.cat(
            [vertices_tensor, pred_deformed_pc],
            dim=0
        )


        # Prepare trajectories
        trajs = [
            1.0* pred_deformed_pc[:, idx].cpu()
            for idx in all_indices
        ]


        # Render animation
        drive_mesh_with_trajs_frames(
            mesh_objects,
            trajs,
            save_video_folder,
            azi=opt.azi,
            ele=opt.ele,
            export_format=opt.export_format
        )
        
        # Alternative option (much slower):
        # drive_mesh_with_trajs_frames_five_views(mesh_objects, trajs, save_video_folder)


    # -----------------------------------------------------
    # Performance statistics
    # -----------------------------------------------------
    t2 = time.time()

    print("Inference Time:", t2 - t1, "seconds")

    print(
        f"Current GPU Memory: "
        f"{torch.cuda.memory_allocated() / 1024**2:.2f} MB"
    )

    print(
        f"Peak GPU Memory: "
        f"{torch.cuda.max_memory_allocated() / 1024**2:.2f} MB"
    )
