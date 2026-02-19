import os
import argparse
import torch
import torch.utils.cpp_extension
from omegaconf import OmegaConf
from collections import OrderedDict
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from dataset.dataset_vae import load_data
from utils import logger
from models.VAEs.vae_bspline import VariationalAutoEncoder
from models.Diffusions.transformer_modified import BsplineVarianceDiT
from utils.rf_diffusion import rf_training_losses
from torch.optim import AdamW
import copy
import time
import numpy as np

# -------------------------------
# Utilities
# -------------------------------

def get_logdir():
    """
    Returns the checkpoint directory for the current experiment.
    Creates the directory if it doesn't exist.
    """
    p = os.path.join(logger.get_dir(), "checkpoints")
    os.makedirs(p, exist_ok=True)
    return p

def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters using Exponential Moving Average (EMA).
    
    EMA update formula:
        target = target * rate + source * (1 - rate)
    
    Args:
        target_params (iterable): Target model parameters to update
        source_params (iterable): Source model parameters (current model)
        rate (float): EMA smoothing factor (closer to 1 = slower update)
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)

# -------------------------------
# Argument Parser
# -------------------------------
def create_argparser():
    """
    Defines the command-line arguments for training VAE + Diffusion models.
    """
    def none_or_str(value):  
        """Helper function to convert 'none' string to Python None"""
        if value.lower() == 'none':  
            return None  
        return value

    parser = argparse.ArgumentParser()

    # ---------- Experiment settings ----------
    parser.add_argument("--exp_name", type=str, default="/tmp/output/", help="Experiment output folder")
    parser.add_argument("--vae_ckpt", type=str, default=None, help="Checkpoint path for pretrained VAE")
    parser.add_argument("--dit_ckpt", type=str, default=None, help="Checkpoint path for pretrained Diffusion model")
    parser.add_argument("--save_interval", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--log_interval", type=int, default=100, help="Log metrics every N steps")
    parser.add_argument("--use_fp16", action="store_true", help="Enable mixed precision training")
    parser.add_argument("--use_tensorboard", action="store_true", help="Enable TensorBoard logging")

    # ---------- Model configuration ----------
    parser.add_argument("--config_vae", type=str, default="configs/vae_animate.yml", help="VAE config YAML")
    parser.add_argument("--config_dit", type=str, default="configs/diffusion.yml", help="Diffusion model config YAML")

    # ---------- Training hyperparameters ----------
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--ema_rate", type=float, default=0.9999, help="Exponential Moving Average rate")
    parser.add_argument("--uncond_p", type=float, default=0.1, help="Probability for unconditional training")
    parser.add_argument("--loss_type", type=str, default="l1", help="Loss function type")
    parser.add_argument("--static_vae_steps", type=int, default=5000, help="Steps for pretraining static VAE")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--xyz_points", type=int, default=4096, help="Number of points in point cloud")
    parser.add_argument("--knn_k", type=int, default=8, help="KNN neighbors for ARAP loss")
    parser.add_argument("--sample_timesteps", type=int, default=1)
    parser.add_argument("--auto_resume", action="store_true", help="Resume training from last checkpoint")
    parser.add_argument("--mem_ratio", type=float, default=1.0, help="Ratio of GPU memory to use")

    # ---------- Dataset arguments ----------
    parser.add_argument("--data_dir", type=str, default="", help="Root directory of dataset")
    parser.add_argument("--canonical_file", type=none_or_str, default="", help="Canonical point cloud file")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=4)
    parser.add_argument("--txt_file", type=str, default="", help="Text file listing dataset items")
    parser.add_argument("--img_feature_root", type=str, default="", help="Root directory for image features")

    return parser

# -------------------------------
# Logging Utilities
# -------------------------------
def log_loss_dict(losses):
    """
    Logs mean values of losses to the global logger.
    
    Args:
        losses (dict): Dictionary of loss tensors
    """
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())

class TrainLoop:
    """
    Training loop for VAE + Diffusion model (BsplineVarianceDiT).

    Responsibilities:
    - Handle data batching and preparation
    - Forward/backward passes
    - Optimizer updates and learning rate scheduling
    - EMA updates of model parameters
    - Checkpoint saving / auto-resume
    - Logging to console and TensorBoard
    """

    def __init__(
        self,
        model_vae,
        model_dit,
        data,
        batch_size,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        use_fp16=False,
        weight_decay=0.0,
        lr_anneal_steps=0,
        active_sh_degree=0,
        white_background=True,
        use_tensorboard=True,
        has_pretrain_weight=False,
        gradient_accumulation_steps=1,
        mem_ratio=1.0,
        args=None,
        deformation_mean_file="dataset/data_statistics/motion_mean.pt",
        deformation_std_file="dataset/data_statistics/motion_std.pt",
        static_mean_file="dataset/data_statistics/static_mean.pt",
        static_std_file="dataset/data_statistics/static_std.pt"
    ):
        """
        Initialize training loop with models, data loader, optimizer, EMA, and logger.
        """

        # ---------------------------------
        # Accelerator setup for multi-GPU / FP16 / DDP
        # ---------------------------------
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        self.accelerator = Accelerator(
            mixed_precision='fp16' if use_fp16 else 'no',
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=gradient_accumulation_steps,
            project_dir=logger.get_dir(),
        )

        # Save experiment args
        options = logger.args_to_dict(args)
        if self.accelerator.is_main_process:
            logger.save_args(options)

        # -------------------------------
        # Model, data, and training hyperparameters
        # -------------------------------
        self.model_vae = model_vae
        self.model_dit = model_dit
        self.data = data
        self.batch_size = batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate] if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.use_fp16 = use_fp16
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * self.accelerator.num_processes
        self.active_sh_degree = active_sh_degree
        self.has_pretrain_weight = has_pretrain_weight
        self.mem_ratio = mem_ratio

        # -------------------------------
        # Optimizer
        # -------------------------------
        self.model_params = list(self.model_dit.parameters())
        self.master_params = self.model_params
        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)

        # -------------------------------
        # Load deformation / static scaling stats
        # -------------------------------
        device = self.accelerator.device
        self.deformation_mean = torch.mean(
            torch.load(deformation_mean_file), dim=0, keepdim=True
        ).to(device=device, dtype=torch.float32)
        self.deformation_std = torch.mean(
            torch.load(deformation_std_file), dim=0, keepdim=True
        ).to(device=device, dtype=torch.float32)

        self.static_mean = torch.mean(
            torch.load(static_mean_file), dim=0, keepdim=True
        ).to(device=device, dtype=torch.float32)
        self.static_std = torch.mean(
            torch.load(static_std_file), dim=0, keepdim=True
        ).to(device=device, dtype=torch.float32)

        # -------------------------------
        # Auto-resume from latest checkpoint
        # -------------------------------
        if args.auto_resume:
            self.auto_resume()

        # -------------------------------
        # Warmup learning rate scheduler
        # -------------------------------
        num_warmup_steps = 2000
        def warmup_lr_schedule(steps):
            if steps < num_warmup_steps:
                return float(steps) / float(max(1, num_warmup_steps))
            return 1.0
        self.warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda=warmup_lr_schedule)

        # -------------------------------
        # Prepare models, optimizer, scheduler, and data for Accelerator
        # -------------------------------
        self.model_vae, self.model_dit, self.opt, self.warmup_scheduler, self.data = \
            self.accelerator.prepare(
                self.model_vae, self.model_dit, self.opt, self.warmup_scheduler, self.data
            )

        # -------------------------------
        # EMA parameters
        # -------------------------------
        self.ema_params = [
            copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
        ]

        # -------------------------------
        # TensorBoard visualization
        # -------------------------------
        self.use_tensorboard = use_tensorboard
        if self.use_tensorboard and self.accelerator.is_main_process:
            self.writer = logger.Visualizer(os.path.join(logger.get_dir(), 'tf_events'))

    # ---------------------------------
    # Auto-resume helpers
    # ---------------------------------
    def auto_resume(self):
        """
        Auto-resume from latest checkpoint if available.
        Loads model, optimizer, and EMA parameters.
        """
        exp_dir = os.path.join(logger.get_dir(), "checkpoints")
        if os.path.exists(exp_dir):
            checkpoints = []
            for f in os.listdir(exp_dir):
                if f.startswith("diffusion_") and f.endswith(".pt"):
                    try:
                        step = int(f.replace("diffusion_", "").replace(".pt", ""))
                        checkpoints.append((step, os.path.join(exp_dir, f)))
                    except ValueError:
                        continue
            if len(checkpoints) > 0:
                latest_step, latest_checkpoint = max(checkpoints, key=lambda x: x[0])
                logger.log(f"Auto-resuming from step {latest_step}...")
                self.resume_step = latest_step
                self._load_state_dict(latest_checkpoint, latest_step)

    def _load_state_dict(self, checkpoint_path, step):
        """
        Load diffusion model checkpoint, optimizer, and EMA states.
        """
        opt_path = os.path.join(logger.get_dir(), "checkpoints", f"opt{step:06d}.pt")
        ema_path = os.path.join(logger.get_dir(), "checkpoints", f"ema_diffusion_0.9999_{step:06d}.pt")

        opt_checkpoint = torch.load(opt_path, map_location=self.accelerator.device)
        ema_checkpoint = torch.load(ema_path, map_location=self.accelerator.device)

        # Remove 'module.' prefix if present
        model_checkpoint = {k.replace('module.', ''): v for k, v in ema_checkpoint.items()}
        opt_checkpoint = {k.replace('module.', ''): v for k, v in opt_checkpoint.items()}

        self.model_dit.load_state_dict(model_checkpoint)
        self.model_params = list(self.model_dit.parameters())
        self.master_params = self.model_params
        self.opt.load_state_dict(opt_checkpoint)

    # ---------------------------------
    # Training loop
    # ---------------------------------
    def run_loop(self):
        """
        Main training loop.
        """
        while not self.lr_anneal_steps or self.step <= self.lr_anneal_steps:
            start_time = time.time()
            batch = next(self.data)
            self.run_step(batch)
            step_time = time.time() - start_time
            logger.logkv_mean("step_time", step_time)

            # Logging and checkpointing
            if self.step % self.log_interval == 0 and self.accelerator.is_main_process:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
            self.step += 1

        # Save final step if not already
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch):
        """
        Single training step: forward, backward, optimize, and log.
        """
        self.forward_backward(batch)
        self.optimize()
        self.log_step()

    def forward_backward(self, batch):
        """
        Forward and backward pass for diffusion model.
        Normalizes latent variables using precomputed deformation/static mean/std.
        Logs losses to console and TensorBoard.
        """
        micro_static_pc = batch[0].to(self.accelerator.device)
        micro_static_nms = batch[1].to(self.accelerator.device)
        micro_delta_pc = batch[2].to(self.accelerator.device)
        micro_text_prmpts = batch[-1]

        with self.accelerator.accumulate(self.model_dit):
            # Encode VAE without gradient
            with torch.no_grad():
                kl, x, idx, pc0_embed_ori, _, _ = self.model_vae.module.encode(
                    micro_static_pc, micro_static_nms, micro_delta_pc
                )
                x0_latent, xt_latent = x.chunk(2, dim=-1)
                micro_pc_nm_samples = (x0_latent - self.static_mean) / self.static_std
                micro_latent_samples = (xt_latent - self.deformation_mean) / self.deformation_std

            # Compute diffusion losses
            losses = rf_training_losses(
                self.model_dit,
                torch.cat([micro_pc_nm_samples, micro_latent_samples], dim=-1),
                micro_text_prmpts,
                f0_channels=32
            )
            loss = losses["loss"].mean()
            log_loss_dict(losses)

            if self.use_tensorboard and self.step % self.log_interval == 0 and self.accelerator.is_main_process:
                self.writer.write_dict({k: v.mean().item() for k, v in losses.items()}, self.step)

            self.accelerator.backward(loss)

    def optimize(self):
        """
        Optimizer step: gradient clipping, step update, scheduler step, EMA update.
        """
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.model_params, 1.0)

        self.opt.step()
        self.warmup_scheduler.step()
        self.opt.zero_grad()

        logger.logkv_mean("lr", self.opt.param_groups[0]["lr"])
        current_scale = self.accelerator.scaler.get_scale() if self.accelerator.scaler else 1.0
        logger.logkv("lg_grad_scale", np.log(current_scale))

        # EMA update
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def log_step(self):
        """
        Log current step and total samples processed.
        """
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        """
        Save model checkpoint, EMA checkpoints, and optimizer state.
        Only executed on main process.
        """
        if not self.accelerator.is_main_process:
            return

        def save_checkpoint(rate, params, name="diffusion"):
            state_dict = self._master_params_to_state_dict(params)
            logger.log(f"saving model {name} {rate}...")
            if not rate:
                filename = f"{name}_{(self.step+self.resume_step):06d}.pt"
            else:
                filename = f"ema_{name}_{rate}_{(self.step+self.resume_step):06d}.pt"
            with open(os.path.join(get_logdir(), filename), "wb") as f:
                torch.save(state_dict, f)

        # Save main model + EMA models
        save_checkpoint(0, self.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        # Save optimizer state
        with open(os.path.join(get_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"), "wb") as f:
            torch.save(self.opt.state_dict(), f)

    def _master_params_to_state_dict(self, master_params):
        """
        Convert master parameters (used for optimizer) back to state_dict for saving.
        """
        state_dict = self.model_dit.state_dict()
        for i, (name, _value) in enumerate(self.model_dit.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

def main():
    """
    Main entry point for training the VAE + BsplineVarianceDiT model.
    """

    # -------------------------------
    # Parse command-line arguments
    # -------------------------------
    args = create_argparser().parse_args()

    # -------------------------------
    # Load model configurations
    # -------------------------------
    model_and_DIT_config = OmegaConf.load(args.config_dit)
    model_and_VAE_config = OmegaConf.load(args.config_vae)

    # Initialize models
    model_vae = VariationalAutoEncoder(**model_and_VAE_config['model'])
    model_dit = BsplineVarianceDiT(**model_and_DIT_config['model'])

    # -------------------------------
    # Configure logger
    # -------------------------------
    logger.configure(args.exp_name)
    has_pretrain_weight = False

    # -------------------------------
    # Load pretrained VAE checkpoint if specified
    # -------------------------------
    if args.vae_ckpt is not None:
        print("Loading pretrained VAE weights from:", args.vae_ckpt)
        new_state_dict = OrderedDict()
        state_dict = torch.load(args.vae_ckpt, map_location="cpu")
        for k, v in state_dict.items():
            new_state_dict[k[7:] if k.startswith("module.") else k] = v
        msg = model_vae.load_state_dict(new_state_dict, strict=False)
        logger.log("VAE pretrain load msg:", msg)
        has_pretrain_weight = True

    # -------------------------------
    # Load pretrained Diffusion checkpoint if specified
    # -------------------------------
    if args.dit_ckpt is not None:
        print("Loading pretrained Diffusion weights from:", args.dit_ckpt)
        new_state_dict = OrderedDict()
        state_dict = torch.load(args.dit_ckpt, map_location="cpu")
        for k, v in state_dict.items():
            new_state_dict[k[7:] if k.startswith("module.") else k] = v
        msg = model_dit.load_state_dict(new_state_dict, strict=False)
        logger.log("Diffusion pretrain load msg:", msg)
        has_pretrain_weight = True

    # -------------------------------
    # Log model configs and parameter count
    # -------------------------------
    logger.log("VAE and Diffusion configs:", model_and_VAE_config, model_and_DIT_config)
    num_dit_params = sum(p.numel() for p in model_dit.parameters() if p.requires_grad) / 1e6
    logger.log("Number of trainable Diffusion parameters: {:.2f} M".format(num_dit_params))

    # -------------------------------
    # Prepare dataset loader
    # -------------------------------
    logger.log("Creating data loader...")
    print("Dataset root path:", args.data_dir)
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        train=True,
        uncond_p=args.uncond_p,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        txt_file=args.txt_file,
        text_condition=True  # Use text prompts for conditional training
    )

    # -------------------------------
    # Launch training loop
    # -------------------------------
    logger.log("Starting training loop...")
    TrainLoop(
        model_vae=model_vae,
        model_dit=model_dit,
        data=data,
        batch_size=args.batch_size,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        use_fp16=args.use_fp16,
        weight_decay=args.weight_decay,
        use_tensorboard=args.use_tensorboard,
        has_pretrain_weight=has_pretrain_weight,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mem_ratio=args.mem_ratio,
        args=args,
    ).run_loop()


# -------------------------------
# Entry point
# -------------------------------
if __name__ == "__main__":
    main()


