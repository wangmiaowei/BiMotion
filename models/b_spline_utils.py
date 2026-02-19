# b_spline_stable.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import numpy as np
import open3d as o3d
import imageio
import os 
from einops import rearrange
from pytorch3d.ops import knn_points
from pytorch3d.ops import sample_farthest_points as p3d_fps


def make_clamped_uniform_knots(num_ctrl, degree):
    M = num_ctrl; p = degree; K = M + p + 1
    knots = torch.zeros(K)
    knots[p:K-p] = torch.linspace(0.0, 1.0, steps=K - 2*p)
    knots[K-p:] = 1.0
    return knots

def bspline_basis_matrix_cox(degree, knots, t):
    device = t.device
    knots = knots.to(device=device, dtype=t.dtype)
    K = knots.shape[0]
    d = int(degree)
    T = t.shape[0]
    M = K - d - 1
    if M <= 0:
        return torch.zeros((T, 0), dtype=t.dtype, device=device)
    diffs = knots[1:] - knots[:-1]
    eps = torch.finfo(knots.dtype).eps
    pos_idx = torch.nonzero(diffs > eps).squeeze(-1)
    last_pos = K - 2 if pos_idx.numel() == 0 else int(pos_idx[-1].item())
    B_prev = torch.zeros((T, K - 1), dtype=t.dtype, device=device)
    for i in range(K - 1):
        left = knots[i]; right = knots[i + 1]
        if i == last_pos:
            mask = (t >= left) & (t <= right)
        else:
            mask = (t >= left) & (t < right)
        B_prev[:, i] = mask.to(t.dtype)
    for r in range(1, d + 1):
        cols = K - r - 1
        if cols <= 0:
            B_prev = B_prev[:, :0]; break
        B_next = torch.zeros((T, cols), dtype=t.dtype, device=device)
        for i in range(cols):
            denom1 = knots[i + r] - knots[i]
            denom2 = knots[i + r + 1] - knots[i + 1]
            if denom1 > 0:
                term1 = ((t - knots[i]) / denom1) * B_prev[:, i]
            else:
                term1 = torch.zeros(T, dtype=t.dtype, device=device)
            if denom2 > 0:
                term2 = ((knots[i + r + 1] - t) / denom2) * B_prev[:, i + 1]
            else:
                term2 = torch.zeros(T, dtype=t.dtype, device=device)
            B_next[:, i] = term1 + term2
        B_prev = B_next
    return B_prev



def B_spline_compute_batch_sharedB(B_s, C_s, w_s=None, denom_clamp=1e-6):
    
    """
    Efficient batched rational B-spline reconstruction with shared basis B_s.
    Minimal branching version — expects C_s to be (B, M, N, 3).
    w_s can be None or same-shape tensor (B, M, N, 3) or (B, M, N, 1).
    Returns (B, T, N, 3).
    """

    # --- lightweight sanity checks (not heavy branching) ---
    assert B_s.dim() == 2, f"B_s must be (T,M), got {B_s.shape}"
    assert C_s.dim() == 4 and C_s.shape[3] == 3, f"C_s must be (B,M,N,3), got {C_s.shape}"
    T, M = B_s.shape
    B, M_c, N, C = C_s.shape
    assert M_c == M, f"M mismatch: B_s M={M}, C_s M={M_c}"

    # --- prepare weight matrix w of shape (B, M, N) ---
    # if w_s is None, create ones of shape (B,M,N)
    if w_s is None:
        w = C_s.new_ones((B, M, N))
    else:
        # assume w_s is (B,M,N,3) or (B,M,N,1) — take mean over last dim to get scalar weight
        # this single line handles both cases (and also works if you pass torch.ones_like(pred_delta))
        w = w_s.mean(dim=-1).squeeze(-1) if w_s.dim() == 4 else w_s  # minimal branching

    # ensure w shape
    assert w.shape == (B, M, N), f"w must be (B,M,N) after processing, got {w.shape}"

    # --- weighted control points (B, M, N, 3) ---
    W_C = w.unsqueeze(-1) * C_s  # (B, M, N, 3)

    # permute for einsum convenience:
    W_C_perm = W_C.permute(0, 2, 1, 3)
    w_perm = w.permute(0, 2, 1)

    # denom: (B, T, N, 1) computed via einsum over m
    denom = torch.einsum('tm,bnm->btn', B_s, w_perm).unsqueeze(-1).clamp_min(denom_clamp)

    # numerator positions: (B, T, N, 3)
    num_pos = torch.einsum('tm,bnmc->btnc', B_s, W_C_perm)

    # final
    P_pred = num_pos / denom  # (B, T, N, 3)
    return P_pred



def B_spline_compute(B_s, C_s, w_s, denom_clamp=1e-6):
    """
    Compute rational B-spline reconstruction.

    Expected shapes:
      B_s : (T, M)         -- basis matrix for current sequence (time x num_ctrl)
      C_s : (M, N, 3)      -- control points for this sequence (num_ctrl x num_points x 3)
      w_s : (M, N, 1)      -- weights per control per point
    Returns:
      P_pred : (T, N, 3)   -- predicted trajectory (time x num_points x 3)
    """

    # reshape w_s to (N, M) for consistency
    w_s = w_s.squeeze(-1).permute(1, 0)   # (M,N,1) -> (M,N) -> (N,M)

    # numer: (T,1,M) * (1,N,M) -> (T,N,M)
    numer = B_s.unsqueeze(1) * w_s.unsqueeze(0)

    # denom: sum over control dim -> (T, N, 1)
    denom = numer.sum(dim=2, keepdim=True).clamp_min(denom_clamp)

    # rational basis R(t,n,m): (T, N, M)
    R = numer / denom

    # P_pred: einsum over m, matching C_s shape (M, N, 3)
    # 'tnm,mnc->tnc' -> (T,N,3)
    P_pred = torch.einsum('tnm,mnc->tnc', R, C_s)

    return P_pred