import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import math
from typing import Iterable, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
)


from .dif_util import *



class Transformer_cogx(nn.Module):
    def __init__(
        self,
        width,
        layers,
        heads,
        init_scale=0.25,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        init_scale = init_scale * math.sqrt(1.0 / width)
        self.resblocks = nn.ModuleList(
            [
                CogXAttentionBlock(
                    width=width,
                    heads=heads,
                    init_scale=init_scale,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor,rest_emb: torch.Tensor, text_emb: torch.Tensor, t_emb: torch.Tensor):
        for block in self.resblocks:
            x = block(x,rest_emb, text_emb, t_emb)
        return x


class BsplineVarianceDiT(nn.Module):
    def __init__(
        self,
        input_channels=64,
        output_channels=64,
        width=512,
        layers=12,
        heads=8,
        init_scale=0.25,
        cond_drop_prob=0.1,
        **kwargs,
    ):
        super().__init__()

        self.cond_drop_prob = cond_drop_prob
        
        self.time_embed = MLP(
            width=width, init_scale=init_scale * math.sqrt(1.0 / width)
        )
        self.backbone = Transformer_cogx(
            width=width,
            layers=layers,
            heads=heads,
            init_scale=init_scale,
        )

        self.ln_pre = nn.LayerNorm(width)
        self.ln_pre_text = nn.LayerNorm(width)
        self.ln_pre_rest = nn.LayerNorm(width)
        self.ln_post = nn.LayerNorm(width)
        

        # ✅ Load the online Hugging Face model
        self.clip_text_model = CLIPTextModel.from_pretrained(
            "sd-legacy/stable-diffusion-v1-5",
            subfolder="text_encoder"
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            "sd-legacy/stable-diffusion-v1-5",
            subfolder="tokenizer"
        )
        # 🔹 Or load from a local path (optional)
        # model_path = "/path_to_your_hf_home/hf_models/stable-diffusion-v1-5"
        # self.clip_text_model = CLIPTextModel.from_pretrained(
        #     f"{model_path}/text_encoder",
        #     local_files_only=True,
        #     use_safetensors=True
        # )
        # self.tokenizer = CLIPTokenizer.from_pretrained(
        #     f"{model_path}/tokenizer",
        #     local_files_only=True
        # )
        
        for param in self.clip_text_model.parameters():
            param.requires_grad = False





        self.clip_token_mlp = nn.Linear(768, self.backbone.width)
        self.rest_shape_mlp = nn.Linear(int(input_channels)//2, self.backbone.width)
        self.input_channels = input_channels
        self.input_proj = nn.Linear(input_channels, width)
        self.output_proj = nn.Linear(width, output_channels)
        with torch.no_grad():
            self.output_proj.weight.zero_()
            self.output_proj.bias.zero_()
        
    def _forward_cogx(self, x,rest_embed, text_emb, t_emb):
        
        # conver to 512
        h = self.input_proj(x)    
        h = self.ln_pre(h)
        rest_embed = self.ln_pre_rest(rest_embed)
        text_emb = self.ln_pre_text(text_emb)
        
        # attention 
        h = self.backbone(h,rest_emb = rest_embed ,text_emb=text_emb, t_emb=t_emb)
        
        # project back
        h = self.ln_post(h)
        h = self.output_proj(h)

        return h

    def forward(self, x, t, texts=None,rest_mask=None):
        device = x.device

        # ---- Temporal embedding ----
        t_embed = self.time_embed(timestep_embedding(t, self.backbone.width))

        # ---- Canonical shape embedding ----
        rest_embed = x[:, :, : int(self.input_channels // 2)]
        rest_embed = self.rest_shape_mlp(rest_embed)


        # ---- Text encoding ----
        with torch.no_grad():
            text_inputs = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(self.clip_text_model.device)

            text_embed = self.clip_text_model(text_inputs)[0]  # [B, 77, 768]

        text_embed = self.clip_token_mlp(text_embed).to(device)
        
        #---- Determine which samples have text ----
        has_text = torch.tensor([text != '' for text in texts],
                                dtype=torch.bool, device=device)
                                
        if self.training:
            # Randomly drop conditioning for classifier-free guidance
            drop_mask_text = (torch.rand(len(x), device=device) < self.cond_drop_prob)  # True = drop
            keep_mask = has_text & ~drop_mask_text  # keep only if has_text and not dropped
            text_embed = text_embed * keep_mask[:, None, None]  # broadcast multiply
        else:
            text_embed = text_embed * has_text[:, None, None]
        return self._forward_cogx(x, rest_embed, text_embed, t_embed)


