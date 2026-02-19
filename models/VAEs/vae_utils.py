import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
import numpy as np
import pytorch3d.ops as ops
from einops import rearrange
from typing import Optional, List
from accelerate import Accelerator

try:
    # try flash-attn style API first (common in flash-attn v2 / ops)
    from flash_attn.ops import memory_efficient_attention as _me_attention
    _ME_BACKEND = "flash_attn_ops"
except Exception:
    _me_attention = None
    try:
        # try xformers (also exposes memory_efficient_attention)
        from xformers.ops import memory_efficient_attention as _me_attention
        _ME_BACKEND = "xformers"
    except Exception:
        _me_attention = None
        _ME_BACKEND = "torch"  # fallback


# helper
def _use_me_attention(mask):
    """
    Decide whether to use memory-efficient attention backend.
    Many ME implementations have limited support for boolean masks / attn_bias.
    Here we use ME backend only if mask is None (safe); otherwise fallback to torch.sdp.
    You can extend this to accept certain bias forms if you know your backend supports them.
    """
    if _ME_BACKEND == "torch":
        return False
    # if mask provided, prefer safe fallback
    if mask is not None:
        return False
    return True

    
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def bspline_basis_matrix_cox(degree, knots, t):
    """
    Cox-de Boor, device/dtype safe.
    degree: int
    knots: 1D torch tensor of length K
    t: 1D torch tensor of sample parameters length T
    returns: (T, M) where M = K - degree - 1
    """
    device = t.device
    knots = knots.to(device=device, dtype=t.dtype)
    K = knots.shape[0]
    d = int(degree)
    T_len = t.shape[0]
    M = K - d - 1
    if M <= 0:
        return torch.zeros((T_len, 0), dtype=t.dtype, device=device)

    eps = torch.finfo(knots.dtype).eps
    diffs = knots[1:] - knots[:-1]
    pos_idx = torch.nonzero(diffs > eps).squeeze(-1)
    last_pos = K - 2 if pos_idx.numel() == 0 else int(pos_idx[-1].item())

    # degree-0 basis
    B_prev = torch.zeros((T_len, K - 1), dtype=t.dtype, device=device)
    for i in range(K - 1):
        left = knots[i]; right = knots[i + 1]
        if i == last_pos:
            mask = (t >= left) & (t <= right)
        else:
            mask = (t >= left) & (t < right)
        B_prev[:, i] = mask.to(t.dtype)

    # recurrence (r = 1..d)
    for r in range(1, d + 1):
        cols = K - r - 1
        if cols <= 0:
            B_prev = B_prev[:, :0]
            break
        B_next = torch.zeros((T_len, cols), dtype=t.dtype, device=device)
        for i in range(cols):
            denom1 = (knots[i + r] - knots[i])
            denom2 = (knots[i + r + 1] - knots[i + 1])

            # denom1, denom2 are scalar tensors; use .item() test (cheap since K small)
            if abs(denom1.item()) > eps:
                term1 = ((t - knots[i]) / denom1) * B_prev[:, i]
            else:
                term1 = torch.zeros(T_len, dtype=t.dtype, device=device)
            if abs(denom2.item()) > eps:
                term2 = ((knots[i + r + 1] - t) / denom2) * B_prev[:, i + 1]
            else:
                term2 = torch.zeros(T_len, dtype=t.dtype, device=device)
            B_next[:, i] = term1 + term2
        B_prev = B_next
    return B_prev 


def make_clamped_uniform_knots(num_ctrl, degree, device, dtype):
    """
    Return clamped uniform knots as torch tensor.
    """
    M = num_ctrl; p = degree; K = M + p + 1
    knots = torch.zeros(K, dtype=dtype, device=device)
    # interior spans count = K - 2p
    if K - 2 * p > 0:
        knots[p:K - p] = torch.linspace(0.0, 1.0, steps=K - 2 * p, device=device, dtype=dtype)
    knots[K - p:] = 1.0
    return knots


# -------------------------
# helper: build A and D on torch (regularized normal eqn)
# -------------------------
def build_A_and_D_torch(nc: int, n_trunc: int, degree: int, knot_mode: str,
                        device, dtype, custom_knots=None, reg: float = 1e-6):
    """
    Returns:
      A (n_trunc, nc) torch tensor
      D (nc, n_trunc) torch tensor computed by regularized normal equations:
        D = (A^T A + reg I)^{-1} A^T
    """
    p = degree
    if knot_mode == 'custom' and custom_knots is not None:
        knots = torch.as_tensor(custom_knots, dtype=dtype, device=device)
        assert knots.numel() == nc + p + 1
    else:
        # clamped uniform knots
        knots = make_clamped_uniform_knots(nc, degree=p, device=device, dtype=dtype)

    u0 = knots[p].item()
    u1 = knots[-p - 1].item()
    if n_trunc == 1:
        u = torch.tensor([(u0 + u1) / 2.0], dtype=dtype, device=device)
    else:
        u = torch.linspace(u0, u1, steps=n_trunc, dtype=dtype, device=device)

    A = bspline_basis_matrix_cox(degree, knots, u)  # (n_trunc, nc) on device
    # compute D via regularized normal eq: D = solve(AtA + reg I, At)
    At = A.transpose(0, 1)           # (nc, n_trunc)
    AtA = At @ A                     # (nc, nc)
    # add reg
    AtA_reg = AtA + (reg * torch.eye(AtA.shape[0], device=device, dtype=dtype))
    # solve linear system
    D = torch.linalg.solve(AtA_reg, At)  # (nc, n_trunc)
    return A, D

# --------------------------
# Improved BSplineMatrixEncoder (complete)
class BSplineMatrixEncoder_SynthesisFirst(nn.Module):
    """
    Build synthesis matrix S from A_0..A_{L-1}:
      x = S * y  where y = [c_L; d_0; d_1; ...]
    Then build analysis B = pinv(S) so y = B x.

    forward: x (B, T_in, N, 3) -> pad to T_eff if needed -> for each (B,N) series compute y = B @ x_series
             then fuse via MLP -> (B, N, E)
    """
    def __init__(self,
                 T: int,
                 N: int,
                 embed_dim: int = 128,
                 min_coarse: int = 2,
                 pad_mode: str = "repeat",
                 degree: int = 3,
                 knot_mode: str = "clamped",
                 custom_knots: dict = None,
                 reg: float = 1e-6,
                 step: Optional[int] = None,
                 device=None,
                 dtype=torch.float32):
        super().__init__()
        assert pad_mode in ("repeat", "reflect")
        self.T_in = T
        self.N = N
        self.embed_dim = embed_dim
        self.min_coarse = min_coarse
        self.pad_mode = pad_mode
        self.degree = degree
        self.knot_mode = knot_mode
        self.custom_knots = custom_knots or {}
        self.reg = reg
        self.step = step
        self.dtype = dtype
        self.device_build = torch.device("cpu") if device is None else device

        # effective odd T
        self.T_eff = T if (T % 2 == 1) else (T + 1)

        # build levels (pairs (n_trunc, Nc)) — same logic as before
        if self.step is None:
            levels = []
            n = self.T_eff
            while True:
                Nc = (n + 1) // 2
                Nf_expected = 2 * Nc - 1
                n_trunc = Nf_expected if Nf_expected <= n else n

                
                if Nc < self.min_coarse or Nc == n:
                    break
                levels.append((max(2,int(n_trunc)), int(Nc)))
                n = Nc
        else:
            n_vals: List[int] = []
            n = self.T_eff
            n_vals.append(n)
            while True:
                next_n = n - self.step
                if next_n < self.min_coarse:
                    next_n = self.min_coarse
                if next_n >= n:
                    break
                n_vals.append(next_n)
                if next_n == self.min_coarse:
                    break
                n = next_n
            levels = []
            for j in range(len(n_vals) - 1):
                levels.append((int(n_vals[j]), int(n_vals[j + 1])))
    

        levels =  [(17, 15), (15, 13), (13, 11), (11, 9), (9, 7), (7, 5), (5, 4)]

        self.levels = levels
        self.n_levels = len(levels)
        if self.n_levels == 0:
            # fallback: trivial single-level identity
            self.register_buffer('S_time', torch.eye(self.T_eff, dtype=self.dtype, device=self.device_build))
            self.register_buffer('B_time', torch.linalg.pinv(self.S_time))
            self.row_sizes = torch.tensor([self.T_eff], dtype=torch.long)
        else:
            # Build A^{(l)} for l=0..L-1 (A_l maps n_trunc_l x Nc_l)
            A_list = []
            for (n_trunc, Nc) in self.levels:
                # allow custom knots keyed by Nc
                knots = None
                if self.knot_mode == 'custom' and Nc in self.custom_knots:
                    knots = self.custom_knots[Nc]
                A_t, D_t = build_A_and_D_torch(Nc, n_trunc, degree=self.degree,
                                               knot_mode=self.knot_mode, device=self.device_build,
                                               dtype=self.dtype, custom_knots=knots, reg=self.reg)
                # A_t has shape (n_trunc, Nc)
                A_list.append(A_t.clone())

            # Now compute P_j = product_{k=0}^{j-1} A_k with P_0 = I_{n0}
            n0 = self.levels[0][0]  # expected equal to T_eff in halving case
            P_list = []
            P_prev = torch.eye(n0, dtype=self.dtype, device=self.device_build)  # P_0
            P_list.append(P_prev)  # P_0
            for j in range(1, self.n_levels + 1):
                # P_j = P_{j-1} @ A_{j-1}
                Ajm1 = A_list[j - 1]  # shape (n_{j-1}, n_j)
                Pj = P_prev @ Ajm1    # shapes: (n0, n_{j-1}) @ (n_{j-1}, n_j) -> (n0, n_j)
                P_list.append(Pj)
                P_prev = Pj


            blocks = [P_list[-1]] + P_list[:-1]  # [P_L, P_0, P_1, ..., P_{L-1}]
            S_big = torch.cat(blocks, dim=1)  # (n0, M)
            self.register_buffer('S_time', S_big)

            # compute analysis B = pinv(S) (use torch.linalg.pinv for robustness)
            B_time = torch.linalg.pinv(S_big)  # (M, n0)
            self.register_buffer('B_time', B_time)

            # store row_sizes: [len(c_L)=n_L, len(d0)=n0, len(d1)=n1, ...]
            row_sizes = [self.levels[-1][1]]  # final coarse length n_L (Nc_{L-1}) = n_L
            for (n_tr, Nc) in self.levels:
                row_sizes.append(n_tr)
            self.register_buffer('row_sizes', torch.tensor(row_sizes, dtype=torch.long))

            # register A_list for debugging
            for idx, A_t in enumerate(A_list):
                self.register_buffer(f'A_{idx}', A_t)

        # fusion MLP
        M = int(self.B_time.shape[0])
        self.fusion = nn.Sequential(
            nn.Linear(M * 3, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )

    def _pad_to_Teff(self, x):
        B, T_in, N, C = x.shape
        if T_in == self.T_eff:
            return x
        if T_in == self.T_eff - 1:
            if self.pad_mode == "repeat":
                last = x[:, -1:, :, :]
                x_pad = torch.cat([x, last], dim=1)
                return x_pad
            else:
                reflect = x[:, -2:-1, :, :]
                x_pad = torch.cat([x, reflect], dim=1)
                return x_pad
        else:
            raise ValueError(f"Input T={T_in} not supported (expected {self.T_in} or {self.T_eff}).")

    def forward(self, x):
        """
        x: (B, T_in, N, 3) -> returns (B, N, embed_dim)
        """
        B_batch, T_in, N_in, C = x.shape
        #assert N_in == self.N and C == 3
        x = self._pad_to_Teff(x)
        B_batch, T, N, C = x.shape
        BN = B_batch * N

        # operate on first n0 rows: if levels exist, n0 = self.levels[0][0], else n0 = T_eff
        n0 = int(self.levels[0][0]) if self.n_levels > 0 else self.T_eff
        # take first n0 samples (should be equal to T after pad in halving-case)
        x_trunc = x[:, :n0, :, :].contiguous()  # (B, n0, N, 3)

        x_series = x_trunc.permute(0, 2, 1, 3).contiguous().view(BN, n0, 3)  # (BN, n0, 3)
        x_perm = x_series.permute(0, 2, 1)  # (BN, 3, n0)

        B_time = self.B_time.to(x.device)
        out = torch.matmul(x_perm, B_time.T)  # (BN, 3, M)
        out = out.permute(0, 2, 1).contiguous()  # (BN, M, 3)

        out_flat = out.reshape(BN, -1)
        fused = self.fusion(out_flat)  # (BN, embed_dim)
        fused = fused.view(B_batch, N, self.embed_dim)
        return fused

class PreNorm_qkv(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, fn):
        super().__init__()
        self.fn = fn
        self.norm_q = nn.LayerNorm(query_dim)
        self.norm_k = nn.LayerNorm(key_dim)
        self.norm_v = nn.LayerNorm(value_dim)

    def forward(self, x, key, value, **kwargs):
        x = self.norm_q(x)
        key = self.norm_k(key)
        value = self.norm_v(value)
        return self.fn(x, key, value, **kwargs)
    
class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)
    
class Attention_qkv(nn.Module):
    def __init__(self, query_dim, key_dim = None, value_dim = None, heads = 8, dim_head = 64):
        super().__init__()
        key_dim = default(key_dim, query_dim)
        value_dim = default(value_dim, key_dim)
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(key_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(value_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)
        self._zero_init_out_layer()

    def _zero_init_out_layer(self):
        nn.init.zeros_(self.to_out.weight)
        if self.to_out.bias is not None:
            nn.init.zeros_(self.to_out.bias)

    def forward(self, x, key=None, value=None, mask=None):
        """
        x: (B, N, query_dim)
        key/value: optional (B, Nk, key_dim)
        mask: attn_mask - keep as in torch.sdp format if you want torch fallback
        """
        h = self.heads
        key = default(key, x)
        value = default(value, key)

        # linear projections
        q = self.to_q(x)
        k = self.to_k(key)
        v = self.to_v(value)

        # reshape to (B, h, N, dim_head)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        b, heads, n, d = q.shape

        # if memory-efficient backend available and mask is None -> use it
        if _use_me_attention(mask):
            # collapse batch and heads: (B*h, N, d)
            q_ = q.reshape(b * heads, n, d)
            k_ = k.reshape(b * heads, -1, d)
            v_ = v.reshape(b * heads, -1, d)

            out_me = _me_attention(q_, k_, v_)  # (B*h, S, d)

            # reshape back to (B, h, N, d)
            out = out_me.view(b, heads, n, d)
            out = rearrange(out, 'b h n d -> b n (h d)')
            out = self.to_out(out)
            return out

        # fallback: use torch scaled dot product (supports mask)
        if mask is not None and mask.ndim == 3:
            mask_ = mask
        else:
            mask_ = mask

        # use PyTorch's scaled_dot_product_attention (fast path, supports attn_mask)
        with torch.backends.cuda.sdp_kernel(enable_math=False):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask_,
            )
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class NormEmbed(nn.Module):
    """Linear-based normal embedding.
       Map (B,N,3) or (B,T,N,3) -> same hidden_dim (or user-specified out_dim).
       Optionally apply LayerNorm and small MLP.
    """
    def __init__(self, hidden_dim=48, out_dim=None, use_layernorm=True, mlp=False):
        super().__init__()
        out_dim = hidden_dim if out_dim is None else out_dim
        self.in_dim = 3
        self.out_dim = out_dim
        self.linear = nn.Linear(self.in_dim, out_dim)
        self.use_layernorm = use_layernorm
        self.norm = nn.LayerNorm(out_dim) if use_layernorm else nn.Identity()
        # optional small mlp (nonlinearity) for expressivity
        if mlp:
            self.mlp = nn.Sequential(
                nn.Linear(out_dim, out_dim),
                nn.GELU(),
                nn.Linear(out_dim, out_dim),
            )
        else:
            self.mlp = None

    def forward(self, x):
        # x: (B,N,3) or (B,T,N,3)
        orig_dtype = x.dtype
        if x.dim() == 3:
            B, N, D = x.shape
            out = self.linear(x)  # broadcasts on last dim -> (B,N,out_dim)
            if self.mlp is not None:
                out = self.mlp(out)
            out = self.norm(out)
        elif x.dim() == 4:
            B, T, N, D = x.shape
            # flatten then linear then reshape
            flat = x.reshape(B*T*N, D)
            out = self.linear(flat)
            if self.mlp is not None:
                out = self.mlp(out)
            out = self.norm(out)
            out = out.reshape(B, T, N, self.out_dim)
        else:
            raise ValueError("NormEmbed expects input of dim 3 or 4")
        return out.to(orig_dtype)



class PointNormEmbed(nn.Module):
    """Combine PointEmbed + NormEmbed into a single embedding suitable for attention.
       - mode: 'concat' (default) or 'sum'
       - gate: if True, learn a scalar gate to weight normal branch (applied when concat or sum)
       - proj_dim: final output dim for attention (if None, use hidden_dim)
    """
    def __init__(self,
                 hidden_dim=48,
                 proj_dim=None,
                 mode='sum',
                 use_norm_layer=True,
                 norm_mlp=False,
                 use_gate=True):
        super().__init__()
        assert mode in ('concat', 'sum'), "mode must be 'concat' or 'sum'"
        self.mode = mode
        self.hidden_dim = hidden_dim
        self.point = PointEmbed(dim=hidden_dim)
        self.norm = NormEmbed(hidden_dim=hidden_dim, use_layernorm=use_norm_layer, mlp=norm_mlp)
        self.use_gate = use_gate

        if mode == 'concat':
            in_proj_dim = hidden_dim * 2
        else:  # sum
            in_proj_dim = hidden_dim

        self.proj_dim = proj_dim if proj_dim is not None else hidden_dim
        # final projection to desired attention dimension
        #self.proj = nn.Linear(in_proj_dim, self.proj_dim)
        self.out_norm = nn.LayerNorm(self.proj_dim)


    def forward(self, pos, normals):
        # pos, normals: same shape (B,N,3) or (B,T,N,3)
        pe = self.point(pos)    # (..., hidden_dim)
        ne = self.norm(normals) # (..., hidden_dim)

        cos_sim = (1 + F.cosine_similarity(pe, ne, dim=-1))/2 
        combined = pe + cos_sim.unsqueeze(-1) * ne
        out = self.out_norm(combined)
        return out


class PointEmbed(nn.Module):
    def __init__(self, hidden_dim=48, dim=128):
        super().__init__()
        assert hidden_dim % 6 == 0
        self.embedding_dim = hidden_dim
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6), e,
                        torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6), e]),
        ])
        self.register_buffer('basis', e)  # 3 x 16
        self.mlp = nn.Linear(self.embedding_dim+3, dim)
        self.out_norm = nn.LayerNorm(dim)
    @staticmethod
    def embed(input, basis):
        projections = torch.einsum(
            'bnd,de->bne', input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
        return embeddings
    
    def forward(self, input):
        # input: B x N x 3
        embed = self.mlp(torch.cat([self.embed(input, self.basis), input], dim=2)) # B x N x C
        
        return self.out_norm(embed)

class DiagonalGaussianDistribution(object):
    def __init__(self, mean, logvar, deterministic=False):
        self.mean = mean
        self.logvar = logvar
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.mean.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.mean.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.mean(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2])
            else:
                return 0.5 * torch.mean(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)
    

class SyncAttentionPreNorm(nn.Module):
    def __init__(self, dim, attn_module):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.norm_v1 = nn.LayerNorm(dim)
        self.norm_v2 = nn.LayerNorm(dim)
        self.attn = attn_module

    def forward(self, q_stream, k_stream, v1_stream, v2_stream):
        q = self.norm_q(q_stream)
        k = self.norm_k(k_stream)
        v1 = self.norm_v1(v1_stream)
        v2 = self.norm_v2(v2_stream)
        return self.attn(q, k, v1, v2)

class SyncFFNPreNorm(nn.Module):
    def __init__(self, dim, ffn_module):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = ffn_module

    def forward(self, x1, x2):
        return self.ffn(self.norm1(x1), self.norm2(x2))
    

class SyncAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v1 = nn.Linear(dim, inner_dim, bias=False)
        self.to_v2 = nn.Linear(dim, inner_dim, bias=False)
        self.to_out1 = nn.Linear(inner_dim, dim)
        self.to_out2 = nn.Linear(inner_dim, dim)

    def forward(self, q_stream, k_stream, v1_stream, v2_stream, mask=None):
        """
        q_stream, k_stream, v1_stream, v2_stream: (B, N, dim)
        mask: optional attention mask; if provided we fallback to PyTorch SDP
        returns: out1, out2 each (B, N, dim)
        """
        h = self.heads
        # linear projections
        q = self.to_q(q_stream)
        k = self.to_k(k_stream)
        v1 = self.to_v1(v1_stream)
        v2 = self.to_v2(v2_stream)

        # reshape to (B, h, N, d)
        q, k, v1, v2 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v1, v2))
        b, heads, n, d = q.shape

        # concat v across last dim -> (b, h, n, 2d)
        v_cat = torch.cat([v1, v2], dim=-1)

        # try using memory-efficient attention if available and mask is None
        if _use_me_attention(mask):
            try:
                # collapse batch & heads -> (b*h, n, d) for q/k and (b*h, n, 2d) for v
                q_ = q.reshape(b * heads, n, d)
                k_ = k.reshape(b * heads, n, d)
                v_ = v_cat.reshape(b * heads, n, 2 * d)

                # call backend; some backends may require equal dim for v, others accept different
                out_me = _me_attention(q_, k_, v_)  # expect (b*h, n, 2d) or may raise

                # reshape back to (b, h, n, 2d)
                out = out_me.view(b, heads, n, 2 * d)
                # split back into two halves and produce outputs
                out1, out2 = out.chunk(2, dim=-1)  # each (b, h, n, d)
                out1 = rearrange(out1, 'b h n d -> b n (h d)')
                out2 = rearrange(out2, 'b h n d -> b n (h d)')
                return self.to_out1(out1), self.to_out2(out2)
            except Exception:
                # memory-efficient backend failed (maybe incompatible v dim) -> fallback below
                pass

        # fallback: use PyTorch scaled_dot_product_attention which supports different v dim and mask
        # scaled_dot_product_attention expects (b, h, n, d) tensors; it will compute attention weights and apply to v_cat
        with torch.backends.cuda.sdp_kernel(enable_math=False):
            out_cat = F.scaled_dot_product_attention(q, k, v_cat, attn_mask=mask)
        # out_cat: (b, h, n, 2d)
        out1, out2 = out_cat.chunk(2, dim=-1)  # each (b, h, n, d)
        out1 = rearrange(out1, 'b h n d -> b n (h d)')
        out2 = rearrange(out2, 'b h n d -> b n (h d)')
        return self.to_out1(out1), self.to_out2(out2)


class SyncFeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.ffn1 = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )
        self.ffn2 = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )
    def forward(self, x1, x2):
        return self.ffn1(x1), self.ffn2(x2)
