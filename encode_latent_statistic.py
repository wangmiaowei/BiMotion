import os
import copy
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from utils import tools  # custom utility functions
from einops import rearrange
import json
import torch.utils.cpp_extension
from omegaconf import OmegaConf
from collections import OrderedDict
import time
from tqdm import tqdm
from dataset.dataset_vae import load_data
from models.VAEs.vae_bspline import VariationalAutoEncoder
import torch.distributed as dist
from torch.optim import AdamW
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
import random


# -------------------------------
# ARGUMENT PARSER
# -------------------------------
def create_argparser():
    """
    Defines command-line arguments for evaluation / latent statistics computation.
    """
    def none_or_str(value):  
        if value.lower() == 'none':  
            return None  
        return value

    parser = argparse.ArgumentParser()
    
    # Experiment / checkpoint
    parser.add_argument("--ckpt", type=str, default=None, help="Path to pretrained checkpoint")
    
    # Mixed precision
    parser.add_argument("--use_fp16", action="store_true", help="Enable FP16 mixed precision")

    # Model config
    parser.add_argument("--config", type=str, default="configs/vae.yml", help="Path to model config YAML")

    # Data loader
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--data_dir", type=str, default="", help="Root directory of dataset")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=4)
    parser.add_argument("--txt_file", type=str, default="", help="File listing dataset items")
 
    return parser


# -------------------------------
# MAIN FUNCTION
# -------------------------------
def main():
    args = create_argparser().parse_args()
    
    # -------------------------------
    # Initialize Accelerator
    # -------------------------------
    accelerator = Accelerator(
        mixed_precision='fp16' if args.use_fp16 else 'no',
    )

    # -------------------------------
    # Load model configuration
    # -------------------------------
    model_and_diffusion_config = OmegaConf.load(args.config)
    if accelerator.is_main_process:
        print("Model and Diffusion config:", model_and_diffusion_config)

    # -------------------------------
    # Initialize VAE model
    # -------------------------------
    model = VariationalAutoEncoder(**model_and_diffusion_config['model'])
    
    # Load pretrained checkpoint if provided
    if args.ckpt is not None:
        if accelerator.is_main_process:
            print("Loading model weights from:", args.ckpt)
        new_state_dict = OrderedDict()
        state_dict = torch.load(args.ckpt, map_location="cpu")
        for k, v in state_dict.items():
            # remove 'module.' prefix if checkpoint saved with DDP wrapper
            new_state_dict[k[7:] if k.startswith("module.") else k] = v
        model.load_state_dict(new_state_dict)

    # -------------------------------
    # Determine per-process data range (for multi-GPU)
    # -------------------------------
    total_samples = args.end_idx - args.start_idx
    samples_per_process = total_samples // accelerator.num_processes
    process_start_idx = args.start_idx + samples_per_process * accelerator.process_index
    process_end_idx = process_start_idx + samples_per_process \
        if accelerator.process_index < accelerator.num_processes - 1 else args.end_idx

    # -------------------------------
    # Load validation data
    # -------------------------------
    val_data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        train=False,
        deterministic=True,
        start_idx=process_start_idx,
        end_idx=process_end_idx,
        txt_file=args.txt_file,
        text_condition=False,
        num_pts=model_and_diffusion_config['model']['num_inputs']
    ) 
    
    # Wrap model and data for accelerator
    model, val_data = accelerator.prepare(model, val_data)
    model.eval()  # evaluation mode

    # -------------------------------
    # Prepare statistics accumulation
    # -------------------------------
    samples_per_process = total_samples // accelerator.num_processes
    num_batch_per_process = samples_per_process // args.batch_size

    progress_bar = tqdm(
        range(num_batch_per_process),
        disable=not accelerator.is_main_process,
        desc=f"Process {accelerator.process_index} encoding"
    )

    out_dir = "dataset/data_statistics"
    os.makedirs(out_dir, exist_ok=True)

    # Use double precision for accumulation for numerical stability
    n_motion, mean_motion, M2_motion = 0, None, None
    n_static, mean_static, M2_static = 0, None, None

    # -------------------------------
    # Loop over batches
    # -------------------------------
    for _ in progress_bar:
        batch = next(val_data)

        with torch.no_grad():
            with accelerator.autocast():
                # Unpack batch
                micro_static_pc = batch[0].to(accelerator.device)
                micro_static_nms = batch[1].to(accelerator.device)
                micro_delta_pc = batch[2].to(accelerator.device)

                B, T, N, _ = micro_delta_pc.shape

                # Encode batch to latent space
                kl, x, idx, pc0_embed_ori, x0, posterior = model.encode(
                    micro_static_pc, micro_static_nms, micro_delta_pc
                )
                
                latent_mean = posterior.mean.cpu().detach()
                latent_std = posterior.std.cpu().detach()
                static_batch = x0.cpu().detach()

                # Skip batch if NaNs detected
                if torch.isnan(latent_mean).any() or torch.isnan(latent_std).any() or torch.isnan(static_batch).any():
                    print("NaN detected in latent, skipping batch")
                    continue
                
                # -------------------------------
                # Motion latent accumulation (Welford algorithm)
                # Please refer to https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
                # -------------------------------
                motion_batch = latent_mean.double()
                k_motion = motion_batch.shape[0]
                batch_mean_motion = motion_batch.mean(dim=0)
                batch_dev = motion_batch - batch_mean_motion.unsqueeze(0)
                Sb_motion = (batch_dev * batch_dev).sum(dim=0)

                if mean_motion is None:
                    mean_motion = torch.zeros_like(batch_mean_motion, dtype=torch.float64)
                    M2_motion = torch.zeros_like(batch_mean_motion, dtype=torch.float64)

                if n_motion == 0:
                    mean_motion, M2_motion, n_motion = batch_mean_motion.clone(), Sb_motion.clone(), k_motion
                else:
                    delta = batch_mean_motion - mean_motion
                    n_new = n_motion + k_motion
                    mean_motion += delta * (k_motion / n_new)
                    M2_motion += Sb_motion + (delta * delta) * (n_motion * k_motion / n_new)
                    n_motion = n_new

                # -------------------------------
                # Static latent accumulation
                # -------------------------------
                if static_batch is not None:
                    static_batch = static_batch.double()
                    k_static = static_batch.shape[0]
                    batch_mean_static = static_batch.mean(dim=0)
                    batch_dev_static = static_batch - batch_mean_static.unsqueeze(0)
                    Sb_static = (batch_dev_static * batch_dev_static).sum(dim=0)

                    if mean_static is None:
                        mean_static = torch.zeros_like(batch_mean_static, dtype=torch.float64)
                        M2_static = torch.zeros_like(batch_mean_static, dtype=torch.float64)

                    if n_static == 0:
                        mean_static, M2_static, n_static = batch_mean_static.clone(), Sb_static.clone(), k_static
                    else:
                        delta_s = batch_mean_static - mean_static
                        n_new_s = n_static + k_static
                        mean_static += delta_s * (k_static / n_new_s)
                        M2_static += Sb_static + (delta_s * delta_s) * (n_static * k_static / n_new_s)
                        n_static = n_new_s
    # -------------------------------
    # Final statistics
    # -------------------------------
    # Population variance (M2 / n). For sample variance use M2 / (n-1)
    if n_motion > 0:
        motion_var = M2_motion / n_motion
        motion_std = torch.sqrt(motion_var)
        motion_mean = mean_motion
    else:
        motion_mean = motion_std = None

    if n_static > 0:
        static_var = M2_static / n_static
        static_std = torch.sqrt(static_var)
        static_mean = mean_static
    else:
        static_mean = static_std = None

    # Convert to float32 and save
    if motion_mean is not None:
        motion_mean = motion_mean.float().mean(dim=0, keepdims=True)
        motion_std = motion_std.float().mean(dim=0, keepdims=True)
        torch.save(motion_mean, os.path.join(out_dir, "motion_mean.pt"))
        torch.save(motion_std,  os.path.join(out_dir, "motion_std.pt"))

    if static_mean is not None:
        static_mean = static_mean.float().mean(dim=0, keepdims=True)
        static_std = static_std.float().mean(dim=0, keepdims=True)
        torch.save(static_mean, os.path.join(out_dir, "static_mean.pt"))
        torch.save(static_std,  os.path.join(out_dir, "static_std.pt"))


# -------------------------------
# ENTRY POINT
# -------------------------------
if __name__ == "__main__":
    #torch.backends.cudnn.benchmark = True
    main()
