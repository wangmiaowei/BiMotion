import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
import numpy as np
import pytorch3d.ops as ops
from einops import rearrange
from typing import Optional, List
from accelerate import Accelerator
from .vae_utils import *

class VariationalAutoEncoder(nn.Module):
    def __init__(
        self,
        *,
        enc_depth=8, 
        dec_depth=8,
        dim=280,     
        output_dim=3*16,
        latent_dim=32,
        heads=8,
        dim_head=64,
        T=16,
        num_inputs=8192,
        num_traj=512,
        n_layers=1, 
    ):
        super().__init__()

        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.num_traj = num_traj
        self.T = T
        self.latent_dim = latent_dim
        
        # Layer definition
        get_enc_attn = lambda: SyncAttentionPreNorm(dim, SyncAttention(dim=dim, heads=1, dim_head=dim))
        get_enc_ffn = lambda: SyncFFNPreNorm(dim, SyncFeedForward(dim=dim))
        get_dec_attn = lambda: SyncAttentionPreNorm(dim, SyncAttention(dim=dim, heads=heads, dim_head=dim_head))
        get_dec_ffn = lambda: SyncFFNPreNorm(dim, SyncFeedForward(dim=dim))

        ### Encoder

        
        self.point_embed = PointNormEmbed(hidden_dim=dim)

        self.traj_embed  = BSplineMatrixEncoder_SynthesisFirst(T=16, N=8192, embed_dim=dim, min_coarse=4, pad_mode="repeat", degree=3, knot_mode='clamped',step =3)

        # Encoder attn & ffn
        self.enc_blocks = nn.ModuleList([])
        for _ in range(enc_depth):
            self.enc_blocks.append(nn.ModuleList([
                get_enc_attn(),
                get_enc_ffn(),
            ]))
        
        ### VAE sample
        self.mean_fc_x0 = nn.Linear(dim, latent_dim)
        self.mean_fc_xt = nn.Linear(dim, latent_dim)
        self.logvar_fc_xt = nn.Linear(dim, latent_dim)
        
        ### Decoder
        # projector
        self.proj_x0 = nn.Linear(latent_dim, dim)
        self.proj_xt = nn.Linear(latent_dim, dim)
        # Decoder attn & ffn
        self.dec_blocks = nn.ModuleList([])
        for _ in range(dec_depth):
            self.dec_blocks.append(nn.ModuleList([
                get_dec_attn(), 
                get_dec_ffn(),
            ]))
        # Output layers
        self.fc_query = nn.Linear(dim, dim*2)
        self.decoder_final_ca = PreNorm_qkv(dim*2, dim, dim, Attention_qkv(dim*2, dim, dim, heads=1, dim_head=dim*2))
        self.to_outputs = nn.Linear(dim*2, output_dim)

    def encode(self, static_pc,static_nm, nublas_pc, faces=None, valid_mask=None, adj_matrix=None):
        pc0_embed = self.point_embed(static_pc,static_nm) 
        pct_embed = self.traj_embed(nublas_pc)
        pc0_embed_ori = pc0_embed
        pct_embed_ori = pct_embed

        # Sample & Gather
        with torch.no_grad():
            _, idx = ops.sample_farthest_points(points=pc0_embed, K=self.num_traj)

        pc0_embed = torch.gather(pc0_embed, 1, idx.unsqueeze(-1).expand(-1, -1, pc0_embed.shape[-1]))
        pct_embed = torch.gather(pct_embed, 1, idx.unsqueeze(-1).expand(-1, -1, pct_embed.shape[-1]))
        # Enc CA & FFN
        for enc_attn, enc_ffn in self.enc_blocks:
            # CA
            attn_res_0, attn_res_t = enc_attn(
                q_stream=pc0_embed, 
                k_stream=pc0_embed_ori, 
                v1_stream=pc0_embed_ori, 
                v2_stream=pct_embed_ori
            )

            pc0_embed = pc0_embed + attn_res_0
            pct_embed = pct_embed + attn_res_t

            # FFN
            ffn_res_0, ffn_res_t = enc_ffn(pc0_embed, pct_embed)
            pc0_embed = pc0_embed + ffn_res_0
            pct_embed = pct_embed + ffn_res_t
        
        # VAE
        x0 = self.mean_fc_x0(pc0_embed)
        mean = self.mean_fc_xt(pct_embed)
        logvar = self.logvar_fc_xt(pct_embed)
        
        posterior = DiagonalGaussianDistribution(mean, logvar)
        xt = posterior.sample()
        kl = posterior.kl()
        x = torch.cat([x0, xt], dim=-1)

        return kl, x, idx, pc0_embed_ori,x0,posterior

    def decode(self, x, static_pc,static_nm, pc0_embed_ori):
        # Projection
        x0_latent, xt_latent = x.chunk(2, dim=-1)
        x0 = self.proj_x0(x0_latent)
        xt = self.proj_xt(xt_latent)
        
        # Dec SA & FFN
        for dec_attn, dec_ffn in self.dec_blocks:
            # SA
            attn_res_0, attn_res_t = dec_attn(
                q_stream=x0, 
                k_stream=x0, 
                v1_stream=x0, 
                v2_stream=xt
            )
            x0 = x0 + attn_res_0
            xt = xt + attn_res_t
            # FFN
            ffn_res_0, ffn_res_t = dec_ffn(x0, xt)
            x0 = x0 + ffn_res_0
            xt = xt + ffn_res_t

        # Final CA & Projection
        query_embed = self.fc_query(pc0_embed_ori)
        latents = self.decoder_final_ca(query_embed, key=x0, value=xt)
        outputs = self.to_outputs(latents)
        outputs = outputs.view(x.shape[0], static_pc.shape[1], -1, 3).permute(0, 2, 1, 3)
        
        return outputs

    def forward(self, static_pc, static_nm,nublas_pc, faces=None, valid_mask=None, adj_matrix=None, num_traj=None, just_encode=False, just_decode=False, samples=None):
        if num_traj is not None:
            self.num_traj = num_traj

        kl, x, idx, pc0_embed_ori,_,_ = self.encode(static_pc,static_nm, nublas_pc, faces=faces, valid_mask=valid_mask, adj_matrix=adj_matrix)
        
        if just_encode: return x
        
        recon_pc = self.decode(samples if just_decode else x, static_pc,static_nm, pc0_embed_ori)

        if just_decode: return recon_pc

        return {'logits': recon_pc, 'kl': kl, 'idx_temp': idx, 'latent': x}



