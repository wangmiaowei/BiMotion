import math
import torch
import torch.nn as nn
from typing import *
import torch.nn.functional as F
from .full_attn import scaled_dot_product_attention




class AdaLayerNormZero(nn.Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim, norm_type="layer_norm", bias=True):
        super().__init__()
        
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=bias)

        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb=None):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp

class MultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (F.normalize(x.float(), dim = -1) * self.gamma * self.scale).to(x.dtype)

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int]=None,
        type: Literal["self", "cross"] = "self",
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        qkv_bias: bool = True,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        assert channels % num_heads == 0
        assert type in ["self", "cross"], f"Invalid attention type: {type}"
        assert attn_mode in ["full", "windowed"], f"Invalid attention mode: {attn_mode}"
        assert type == "self" or attn_mode == "full", "Cross-attention only supports full attention"
        
        if attn_mode == "windowed":
            raise NotImplementedError("Windowed attention is not yet implemented")
        
        self.channels = channels
        self.head_dim = channels // num_heads
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self.num_heads = num_heads
        self._type = type
        self.attn_mode = attn_mode
        self.window_size = window_size
        self.shift_window = shift_window
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm

        if self._type == "self":
            self.to_qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        else:
            self.to_q = nn.Linear(channels, channels, bias=qkv_bias)
            self.to_kv = nn.Linear(self.ctx_channels, channels * 2, bias=qkv_bias)
            
        if self.qk_rms_norm:
            self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            
        self.to_out = nn.Linear(channels, channels)

        if use_rope:
            self.rope = RotaryPositionEmbedder(channels)
    
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, C = x.shape
        if self._type == "self":
            qkv = self.to_qkv(x)
            qkv = qkv.reshape(B, L, 3, self.num_heads, -1)
            if self.use_rope:
                q, k, v = qkv.unbind(dim=2)
                q, k = self.rope(q, k, indices)
                qkv = torch.stack([q, k, v], dim=2)
            if self.attn_mode == "full":
                if self.qk_rms_norm:
                    q, k, v = qkv.unbind(dim=2)
                    q = self.q_rms_norm(q)
                    k = self.k_rms_norm(k)
                    h = scaled_dot_product_attention(q, k, v)
                else:
                    h = scaled_dot_product_attention(qkv)
            elif self.attn_mode == "windowed":
                raise NotImplementedError("Windowed attention is not yet implemented")
        else:
            Lkv = context.shape[1]
            q = self.to_q(x)
            kv = self.to_kv(context)
            q = q.reshape(B, L, self.num_heads, -1)
            kv = kv.reshape(B, Lkv, 2, self.num_heads, -1)
            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                k, v = kv.unbind(dim=2)
                k = self.k_rms_norm(k)
                h = scaled_dot_product_attention(q, k, v)
            else:
                h = scaled_dot_product_attention(q, kv)
        h = h.reshape(B, L, -1)
        h = self.to_out(h)
        return h



class CondAdaLayerNormZero(nn.Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim, norm_type="layer_norm", bias=True):
        super().__init__()
        
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 2 * embedding_dim, bias=bias)

        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb=None):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa = emb.chunk(2, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x

class CogXAttentionBlock(nn.Module):
    def __init__(
        self,
        width,
        heads,
        init_scale=1.0,
    ):
        super().__init__()


        self.attn_self = MultiHeadAttention(
            width,
            num_heads=heads,
            type="self",
            attn_mode="full",
            window_size=None,
            shift_window=None,
            qkv_bias=True,
            use_rope=False,
            qk_rms_norm=True,
        )
        
        self.text_cross_attn = MultiHeadAttention(
            width,
            ctx_channels=width,
            num_heads=heads,
            type="cross",
            attn_mode="full",
            qkv_bias=True,
            qk_rms_norm=True,
        )

        
        self.rest_cross_attn = MultiHeadAttention(
            width,
            ctx_channels=width,
            num_heads=heads,
            type="cross",
            attn_mode="full",
            qkv_bias=True,
            qk_rms_norm=True,
        )

        

        self.mlp = MLP(width=width, init_scale=init_scale)
        
        self.adaln_x = AdaLayerNormZero(width)
        self.adaln_rest = CondAdaLayerNormZero(width)
        self.adaln_text = CondAdaLayerNormZero(width)

        self.ln_x1 = nn.LayerNorm(width)
        self.ln_x2 = nn.LayerNorm(width)
        self.ln_x3 = nn.LayerNorm(width)
 

    def forward(self, x,rest_emb, text_emb, t_emb):

        # adaln for x layenorma+adpative
        norm_x, gate_msa_x, shift_mlp_x, scale_mlp_x, gate_mlp_x = self.adaln_x(x, emb=t_emb)
        
        # adaln for text
        norm_text  = self.adaln_text(text_emb, emb=t_emb)
        
        # adaln for rest
        norm_rest  = self.adaln_rest(rest_emb, emb=t_emb)

        # self attention 
        h = self.attn_self(norm_x) 
        h = h * gate_msa_x.unsqueeze(1) 
        x = x + h

        # Text cross attention
        h = self.ln_x1(x)
        h = self.text_cross_attn(h,norm_text)
        x = x + h

        # Rest cross attention 
        h = self.ln_x2(x)
        h = self.rest_cross_attn(h,norm_rest)
        x = x + h

        # feedforward by MLP
        h = self.ln_x3(x)
        h = h * (1 + scale_mlp_x[:, None]) + shift_mlp_x[:, None]
        h = self.mlp(h)
        h = gate_mlp_x[:, None] * h


        x = x + h

        return x


class MLP(nn.Module):
    def __init__(self, width, init_scale=0.25):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * 4)
        self.c_proj = nn.Linear(width * 4, width)
        self.gelu = nn.GELU()
        init_linear(self.c_fc, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


def init_linear(l, stddev):
    nn.init.normal_(l.weight, std=stddev)
    if l.bias is not None:
        nn.init.constant_(l.bias, 0.0)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].to(timesteps.dtype) * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
