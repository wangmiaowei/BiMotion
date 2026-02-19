# Source: AnimateAnyMesh (ICCV 2025)

import torch
import numpy as np
import time
from pytorch3d.io import save_obj


def mesh_preprocess(vertices, faces, max_length=4096):
    """
    Normalize and pad mesh data to a fixed length.

    Args:
        vertices (Tensor): Shape (T, V, 3) or (1, T, V, 3). Vertex coordinates.
        faces (Tensor): Shape (F, 3) or (1, F, 3). Face indices.
        max_length (int): Maximum number of vertices after padding.

    Returns:
        vertices (Tensor): Normalized and padded vertices.
        vertices_color (Tensor): Padded vertex colors (placeholder).
        faces (Tensor): Padded faces.
        valid_mask (Tensor): Boolean mask indicating valid vertices.
        valid_length (Tensor): Number of valid vertices.
        adj_matrix (Tensor): Adjacency matrix of the mesh.
    """

    # If batched, remove batch dimension
    if vertices.ndim == 4:
        vertices, faces = vertices[0], faces[0]

    # Center the mesh using the first frame
    center = (vertices[0].max(dim=0)[0] + vertices[0].min(dim=0)[0]) / 2
    vertices = vertices - center

    # Normalize scale
    v_max = vertices[0].abs().max()
    vertices = vertices / v_max

    # Mask for valid vertices
    valid_mask = torch.ones(vertices.shape[1], dtype=torch.bool)

    # Ensure faces are int64
    faces = torch.tensor(faces, dtype=torch.int64)

    faces_max_length = int(max_length * 2.5)

    # Pad vertices to fixed length
    vertices = torch.cat(
        [vertices, torch.zeros(vertices.shape[0], max_length - vertices.shape[1], 3)],
        dim=1
    )

    # NOTE: vertices_color is assumed to exist externally
    vertices_color = torch.cat(
        [vertices_color, -1.0 * torch.ones(max_length - vertices_color.shape[0], 3)],
        dim=0
    )

    # Pad faces with invalid indices (-1)
    faces = torch.cat(
        [faces, -1 * torch.ones(faces_max_length - faces.shape[0], 3).to(torch.int64)],
        dim=0
    )

    # Extend valid mask
    valid_mask = torch.cat(
        [valid_mask, torch.zeros(max_length - valid_mask.shape[0], dtype=torch.bool)]
    )[None]

    valid_length = valid_mask.sum(dim=-1)

    # Compute adjacency matrix (external function)
    adj_matrix = get_adjacency_matrix(vertices[0][None], faces[None], valid_length)

    return vertices, vertices_color, faces, valid_mask, valid_length, adj_matrix


def find_indices_in_merged(vertices_list, merged_vertices):
    """
    Find indices of each vertex set within a merged vertex tensor.

    Args:
        vertices_list (list[Tensor]): List of (Ni, 3) vertex tensors.
        merged_vertices (Tensor): Combined vertices (N_total, 3).

    Returns:
        indices_list (list[Tensor]): Indices mapping each original
                                     vertex set to the merged tensor.
    """
    indices_list = []

    for vertices in vertices_list:
        # Compare each vertex with merged vertices
        matches = (merged_vertices.unsqueeze(1) == vertices.unsqueeze(0))
        matches = matches.all(dim=2)

        indices = matches.nonzero()[:, 0]
        indices = indices.reshape(vertices.shape[0])
        indices_list.append(indices)

    return indices_list


def merge_vertices_with_indices(vertices_list, faces_list):
    """
    Merge multiple meshes into one without deduplication.

    Args:
        vertices_list (list[Tensor]): List of vertex tensors (Ni, 3).
        faces_list (list[Tensor]): List of face tensors (Fi, 3).

    Returns:
        merged_vertices (Tensor): Concatenated vertices.
        merged_faces (Tensor): Concatenated faces.
        indices_list (list[Tensor]): Index ranges for each original mesh.
    """

    # Concatenate all vertices and faces
    all_vertices = torch.cat(vertices_list, dim=0)
    all_faces = torch.cat(faces_list, dim=0)

    indices_list = []
    start_idx = 0

    # Create sequential indices for merged vertices
    inverse_indices = torch.arange(len(all_vertices))
    print("viss:", inverse_indices.shape)

    # Record index ranges for each mesh
    for vertices in vertices_list:
        indices_list.append(inverse_indices[start_idx:start_idx + len(vertices)])
        start_idx += len(vertices)

    return all_vertices, all_faces, indices_list
