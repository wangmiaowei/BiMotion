"""
Dataset utilities for spline-based motion training.

This file provides:
- Infinite dataloader generator
- Random rotation helpers
- JSON loading utility
- VolumeDataset (main dataset class)
"""

# =========================================================
# Standard / Third-party Imports
# =========================================================
import os
import json
import random
import numpy as np
import torch

from pathlib import Path
from torch.utils.data import DataLoader, Dataset


# =========================================================
# Dataloader factory
# =========================================================

def load_data(
    *,
    data_dir,
    batch_size,
    deterministic=False,
    train=True,
    start_idx=-1,
    end_idx=-1,
    txt_file="",
    num_pts=4096,
    num_workers=8,
    text_condition=False,
    **kwargs,
):
    """
    Create an infinite dataloader generator.

    Parameters
    ----------
    data_dir : str
        Root dataset directory
    txt_file : str
        File containing relative paths to samples
    num_pts : int
        Number of points sampled per object
    deterministic : bool
        Disable shuffle if True

    Returns
    -------
    generator
        Infinite generator yielding batches
    """

    print("Using spline sequences")

    if not data_dir:
        raise ValueError("Data directory not specified")

    # -----------------------------------------------------
    # Load file list
    # -----------------------------------------------------
    with open(txt_file) as f:
        raw_files = f.read().splitlines()

    all_files = [os.path.join(data_dir, x) for x in raw_files]

    if start_idx >= 0 and end_idx >= 0 and start_idx < end_idx:
        all_files = all_files[start_idx:end_idx]

    print("Number of sequences:", len(all_files))

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------
    dataset = VolumeDataset(
        data_dir=data_dir,
        motion_paths=all_files,
        train=train,
        num_pts=num_pts,
        text_condition=text_condition,
    )

    # -----------------------------------------------------
    # DataLoader
    # -----------------------------------------------------
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not deterministic,
        num_workers=num_workers,
        drop_last=not deterministic,
        pin_memory=True,
    )

    # Infinite iterator
    while True:
        yield from loader


# =========================================================
# Geometry helpers
# =========================================================

def random_rotation_matrix(device="cpu"):
    """
    Generate a random 3D rotation matrix (uniform on SO(3)).
    """
    theta = torch.rand(1, device=device) * 2 * torch.pi
    phi = torch.rand(1, device=device) * 2 * torch.pi
    z = torch.rand(1, device=device) * 2 - 1

    axis = torch.stack(
        [
            torch.sqrt(1 - z**2) * torch.cos(phi),
            torch.sqrt(1 - z**2) * torch.sin(phi),
            z,
        ],
        dim=-1,
    )

    cos_t, sin_t = torch.cos(theta), torch.sin(theta)

    K = torch.tensor(
        [
            [0, -axis[0, 2], axis[0, 1]],
            [axis[0, 2], 0, -axis[0, 0]],
            [-axis[0, 1], axis[0, 0], 0],
        ],
        device=device,
    )

    R = torch.eye(3, device=device) * cos_t + (1 - cos_t) * axis.T @ axis + sin_t * K
    return R


def random_rotation_matrix_z(device="cpu"):
    """Random rotation around Z axis."""
    theta = torch.rand(1, device=device) * 2 * torch.pi
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)

    return torch.tensor(
        [[cos_t, -sin_t, 0.0], [sin_t, cos_t, 0.0], [0.0, 0.0, 1.0]],
        device=device,
    ).squeeze()


# =========================================================
# IO helpers
# =========================================================

def load_json(path):
    """Safely load a JSON file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================
# Main Dataset
# =========================================================

class VolumeDataset(Dataset):
    """
    Dataset for dynamic point cloud sequences with spline motion.

    Returns
    -------
    static_pc : [N, 3]
    static_nm : [N, 3]
    moving_bspline : [T, N, 3]
    traj_sampled : [T, N, 3]
    """

    def __init__(
        self,
        data_dir,
        motion_paths,
        train=True,
        text_condition=False,
        num_pts=4096,
        slow_thresh=3.0,
    ):
        super().__init__()

        self.local_motion = list(motion_paths)
        self.train = train
        self.num_pts = num_pts
        self.text_condition = text_condition
        self.slow_thresh = slow_thresh

        # Load caption metadata
        self.ObjaverseXL_caption = load_json(
            os.path.join(data_dir, "captions", "captions_gpt5XL.json")
        )
        self.ObjaverseV1_caption = load_json(
            os.path.join(data_dir, "captions", "captions_gpt5V1.json")
        )
        self.DT4D_caption = load_json(
            os.path.join(data_dir, "captions", "captions_gpt5DT4D.json")
        )

    def __len__(self):
        return len(self.local_motion)

    def __getitem__(self, idx, retry=0):
        """
        Load a single training sample.
        Includes retry logic for corrupted files.
        """

        MAX_RETRY = 5
        root_path = self.local_motion[idx]
        sub_file = os.path.basename(root_path)

        try:
            # -------------------------------------------------
            # Randomly pick a partition file
            # -------------------------------------------------
            part_id = random.randint(1, 10)
            file_path = os.path.join(root_path, f"train_data_part{part_id}.npz")

            data = np.load(file_path, allow_pickle=True, mmap_mode="r")

            static_pc_np = data["static_pc"]
            static_nm_np = data["static_nm"]
            traj_sampled_np = data["traj_sampled_fields"]
            moving_bspline_np = data["bspline"]

            N = static_pc_np.shape[0]
            if N < self.num_pts:
                raise ValueError(f"N={N} < required {self.num_pts}")

            # -------------------------------------------------
            # Random point sampling
            # -------------------------------------------------
            ind = np.random.choice(N, size=self.num_pts, replace=False)

            static_pc = torch.from_numpy(static_pc_np[ind])
            static_nm = torch.from_numpy(static_nm_np[ind])
            moving_bspline = torch.from_numpy(moving_bspline_np[:, ind])
            traj_sampled = torch.from_numpy(traj_sampled_np[:, ind])

            # Convert trajectory to displacement
            traj_sampled = traj_sampled - static_pc.unsqueeze(0)

            # -------------------------------------------------
            # Noise augmentation
            # -------------------------------------------------
            static_pc += torch.randn_like(static_pc) * 0.004
            static_nm += torch.randn_like(static_nm) * 0.004
            static_nm = static_nm / static_nm.norm(dim=-1, keepdim=True)

            # -------------------------------------------------
            # Output
            # -------------------------------------------------
            if self.text_condition:
                # Caption sampling (dataset-specific)
                if "DT4D" in root_path:
                    caps = data.get("captions", [""])
                    caps = [c for c in caps if isinstance(c, str) and c.strip()]

                    values = self.DT4D_caption.get(sub_file, [])
                    caps_add = [v["caption"] for v in values if v.get("validated")]

                    caps_total = caps + caps_add
                    caption = random.choice(caps_total) if caps_total else ""

                else:
                    values = self.ObjaverseXL_caption.get(sub_file, [])
                    caps = [v["caption"] for v in values if v.get("validated")]
                    caption = random.choice(caps) if caps else ""

                return static_pc, static_nm, moving_bspline, caption

            else:
                return static_pc, static_nm, moving_bspline, traj_sampled, root_path

        except Exception as e:
            print(f"[DATA ERROR] {root_path}: {e}")

            if retry < MAX_RETRY:
                new_idx = np.random.randint(0, len(self))
                return self.__getitem__(new_idx, retry + 1)

            raise RuntimeError(
                f"Dataset failed after {MAX_RETRY} retries at {root_path}"
            )
