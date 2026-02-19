# =========================
# Standard Library Imports
# =========================
import os
import copy
import argparse
import time
from datetime import timedelta
from collections import OrderedDict

# =========================
# Third-Party Imports
# =========================
import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from omegaconf import OmegaConf

import torch.distributed as dist
from torch.optim import AdamW
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from pytorch3d.ops import knn_points

# =========================
# Project Imports
# =========================
from utils import logger
from dataset.dataset_vae import load_data
from models.VAEs.vae_bspline import VariationalAutoEncoder

from models.b_spline_utils import (
    make_clamped_uniform_knots,
    bspline_basis_matrix_cox,
    B_spline_compute_batch_sharedB,
)


# ============================================================
# Logging Utilities
# ============================================================

def get_logdir():
    """
    Create (if necessary) and return the checkpoint directory.

    Returns
    -------
    str
        Path to the checkpoint directory inside the current logger dir.
    """
    checkpoint_dir = os.path.join(logger.get_dir(), "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir


# ============================================================
# Argument Parser
# ============================================================

def create_argparser():
    """
    Create command line argument parser for training or evaluation.

    Returns
    -------
    argparse.ArgumentParser
    """

    def none_or_str(value):
        """
        Convert 'none' (case insensitive) to None.
        Useful for optional CLI string arguments.
        """
        if value.lower() == "none":
            return None
        return value

    parser = argparse.ArgumentParser()

    # ---------------------
    # Experiment settings
    # ---------------------
    parser.add_argument("--exp_name", type=str, default="outputs/")
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--ckpt", type=str, default=None)

    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--use_tensorboard", action="store_true")

    # ---------------------
    # Model config
    # ---------------------
    parser.add_argument("--config", type=str, default="configs/vae.yml")

    # ---------------------
    # Training hyperparameters
    # ---------------------
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--ema_rate", type=float, default=0.9999)

    parser.add_argument("--render_l1_weight", type=float, default=1.0)
    parser.add_argument("--kl_weight", type=float, default=1e-5)

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    # Geometry / correspondence losses
    parser.add_argument("--knn_k", type=int, default=8)
    parser.add_argument("--beta", type=float, default=7.0)
    parser.add_argument("--xyz_loss_weight", type=float, default=0.1)
    parser.add_argument("--corres_loss_weight", type=float, default=0.2)

    parser.add_argument("--sample_timesteps", type=int, default=1)

    parser.add_argument("--auto_resume", action="store_true")

    # ---------------------
    # Dataset options
    # ---------------------
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=4)
    parser.add_argument("--txt_file", type=str, default="")

    return parser


# ============================================================
# Loss Logging
# ============================================================

def log_loss_dict(losses):
    """
    Log a dictionary of loss values.

    Parameters
    ----------
    losses : dict
        Dictionary where keys are loss names and values are either
        scalars or tensors.

    Behavior
    --------
    - Scalars are logged directly.
    - Tensors are averaged before logging.
    """

    for key, values in losses.items():
        if isinstance(values, (int, float)):
            logger.logkv_mean(key, values)
        else:
            # If tensor, log its mean value
            logger.logkv_mean(key, values.mean().item())

class TrainLoop:
    """
    Main training loop for the VAE deformation model.

    Handles:
    - Distributed training via 🤗 Accelerate
    - Checkpoint saving / auto resume
    - Forward + backward passes
    - Loss computation (KL, fitting, correspondence, ARAP)
    """

    def __init__(
        self,
        model,
        data,
        batch_size,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=True,
        weight_decay=0.0,
        lr_anneal_steps=0,
        use_tensorboard=True,
        render_l1_weight=1.0,
        has_pretrain_weight=False,
        xyz_loss_weight=0.1,
        corres_loss_weight=0.1,
        knn_k=8,
        beta=7.0,
        kl_weight=1e-5,
        gradient_accumulation_steps=1,
        args=None,
        auto_resume=False,
    ):

        # =====================================================
        # Accelerator setup (distributed + mixed precision)
        # =====================================================
        ddp_kwargs = DistributedDataParallelKwargs(
            find_unused_parameters=False
        )

        self.accelerator = Accelerator(
            mixed_precision="fp16" if use_fp16 else "no",
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=gradient_accumulation_steps,
            project_dir=logger.get_dir(),
        )

        # Save experiment args
        options = logger.args_to_dict(args)
        if self.accelerator.is_main_process:
            logger.save_args(options)

        # =====================================================
        # Basic attributes
        # =====================================================
        self.model = model
        self.data = data
        self.batch_size = batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
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

        # Loss weights
        self.knn_k = knn_k
        self.beta = beta
        self.xyz_loss_weight = xyz_loss_weight
        self.corres_loss_weight = corres_loss_weight
        self.kl_weight = kl_weight
        self.render_l1_weight = render_l1_weight

        self.has_pretrain_weight = has_pretrain_weight
        self.L1loss = torch.nn.L1Loss()

        # =====================================================
        # Optimizer & LR scheduler
        # =====================================================
        self.opt = AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Precompute B-spline basis (CPU → moved later)
        self.knots = make_clamped_uniform_knots(16, 3)
        sample_timesteps = torch.linspace(0.0, 1.0, steps=16)
        self.B_s = bspline_basis_matrix_cox(3, self.knots, sample_timesteps)
        self.B_s.requires_grad_(False)

        # Warmup scheduler
        num_warmup_steps = 1000

        def warmup_lr_schedule(step):
            if step < num_warmup_steps:
                return float(step) / float(max(1, num_warmup_steps))
            return 1.0

        self.warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lr_lambda=warmup_lr_schedule
        )

        # =====================================================
        # Prepare with Accelerate (wrap model, optimizer, data)
        # =====================================================
        self.model, self.opt, self.warmup_scheduler, self.data = (
            self.accelerator.prepare(
                self.model,
                self.opt,
                self.warmup_scheduler,
                self.data,
            )
        )

        # Move precomputed tensors to correct device
        self.B_s = self.B_s.to(self.accelerator.device)
        self.knots = self.knots.to(self.accelerator.device)

        self.model_params = list(self.model.parameters())

        # =====================================================
        # Auto resume
        # =====================================================
        if auto_resume:
            self.auto_resume()

        # =====================================================
        # TensorBoard logger
        # =====================================================
        self.use_tensorboard = use_tensorboard
        if self.use_tensorboard and self.accelerator.is_main_process:
            self.writer = logger.Visualizer(
                os.path.join(logger.get_dir(), "tf_events")
            )

    # =========================================================
    # Saving / Loading
    # =========================================================

    def save(self):
        """Save model and optimizer checkpoint."""
        if not self.accelerator.is_main_process:
            return

        step_id = self.step + self.resume_step
        model_path = os.path.join(get_logdir(), f"deformation_{step_id:06d}.pt")
        opt_path = os.path.join(get_logdir(), f"opt{step_id:06d}.pt")

        logger.log(f"Saving checkpoint to {model_path}")

        try:
            state = self.accelerator.unwrap_model(self.model).state_dict()
        except Exception:
            state = self.model.state_dict()

        torch.save(state, model_path)
        torch.save(self.opt.state_dict(), opt_path)
        
    def _load_state_dict(self, model_path, opt_path=None):
        """Load model weights and optionally optimizer state (paths must exist)."""
        model_checkpoint = torch.load(model_path, map_location=self.accelerator.device)

        # target to load into (unwrapped model if possible)
        try:
            target_model = self.accelerator.unwrap_model(self.model)
        except Exception:
            target_model = self.model

        # adapt 'module.' prefix if needed
        adapted_ck = {}
        for k, v in model_checkpoint.items():
            new_k = k
            if k.startswith("module.") and new_k not in target_model.state_dict():
                new_k = k.replace("module.", "", 1)
            adapted_ck[new_k] = v

        target_model.load_state_dict(adapted_ck)

        # optionally load optimizer state
        if opt_path is not None and os.path.exists(opt_path):
            try:
                opt_checkpoint = torch.load(opt_path, map_location=self.accelerator.device)
                self.opt.load_state_dict(opt_checkpoint)
            except Exception as e:
                logger.log(f"Warning: failed to load optimizer state_dict: {e}. Continuing without optimizer state.")

    def auto_resume(self):
        """Resume from the latest checkpoint in the experiment folder."""
        exp_dir = os.path.join(logger.get_dir(), "checkpoints")
        if not os.path.exists(exp_dir):
            return

        checkpoints = []
        for f in os.listdir(exp_dir):
            if f.startswith("deformation_") and f.endswith(".pt"):
                try:
                    step = int(f.replace("deformation_", "").replace(".pt", ""))
                    checkpoints.append((step, os.path.join(exp_dir, f)))
                except ValueError:
                    continue

        if not checkpoints:
            return

        latest_step, latest_model_path = max(checkpoints, key=lambda x: x[0])
        logger.log(f"Auto-resuming from step {latest_step}")

        self.resume_step = latest_step
        opt_path = os.path.join(exp_dir, f"opt{latest_step:06d}.pt")
        self._load_state_dict(latest_model_path, opt_path)

    # =========================================================
    # Training step helpers
    # =========================================================

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv(
            "samples",
            (self.step + self.resume_step + 1) * self.global_batch,
        )

    def run_step(self, batch):
        """Run forward + backward for one batch."""
        self.accelerator.wait_for_everyone()
        self.forward_backward(batch)
        self.log_step()

    # =========================================================
    # Optimization
    # =========================================================

    def optimize(self):
        """Optimizer step + gradient clipping + scheduler."""
        try:
            self.accelerator.clip_grad_norm_(self.model.parameters(), 0.5)
        except Exception:
            pass

        self.opt.step()
        self.warmup_scheduler.step()
        self.opt.zero_grad()

        logger.logkv_mean("lr", self.opt.param_groups[0]["lr"])

        if self.accelerator.scaler is not None:
            scale = self.accelerator.scaler.get_scale()
        else:
            scale = 1.0

        logger.logkv("lg_grad_scale", np.log(scale))

    # =========================================================
    # Losses
    # =========================================================

    def correspondeces_loss(self, traj_pred, gt_traj):
        """Robust L2 (Charbonnier) correspondence loss."""
        return torch.mean(torch.sqrt((traj_pred - gt_traj) ** 2 + 1e-6))

    def arap_knn_loss(self, static_pc, traj_pred, k=8, eps=1e-6):
        """
        ARAP (As-Rigid-As-Possible) regularization using fixed KNN graph.
        Encourages local rigidity across time.
        """

        traj_pred = torch.cat(
            [static_pc.unsqueeze(1), traj_pred + static_pc.unsqueeze(1)],
            dim=1,
        )

        B, T, N, _ = traj_pred.shape
        knn_out = knn_points(static_pc, static_pc, K=k)
        idx = knn_out.idx

        idx_expand = idx.unsqueeze(1).unsqueeze(-1).expand(-1, T, -1, -1, 3)

        neighbors = torch.gather(
            traj_pred.unsqueeze(3).expand(-1, -1, -1, k, -1),
            dim=2,
            index=idx_expand,
        )

        center = traj_pred.unsqueeze(3)
        local_dist = torch.norm(neighbors - center, dim=-1)

        diff = local_dist[:, 1:] - local_dist[:, :1]
        return torch.mean(torch.sqrt(diff**2 + eps**2))

    # =========================================================
    # Forward + backward
    # =========================================================

    def forward_backward(self, batch):
        """Compute losses and run backward pass."""

        static_pc = batch[0].to(self.accelerator.device)
        static_nms = batch[1].to(self.accelerator.device)
        delta_pc = batch[2].to(self.accelerator.device)
        traj_sampled = batch[3].to(self.accelerator.device)

        with self.accelerator.accumulate(self.model):

            losses = {}
            output = self.model(static_pc, static_nms, delta_pc)

            kl_loss = output.get("kl", torch.zeros(1, device=self.accelerator.device)).mean()
            loss = kl_loss * self.kl_weight
            losses["delta_kl"] = kl_loss

            pred_traj = output["logits"]

            fitting_loss = torch.mean(
                torch.sqrt((pred_traj - delta_pc) ** 2 + 1e-6)
            )

            traj_pred = B_spline_compute_batch_sharedB(self.B_s, pred_traj)
            corres_loss = self.correspondeces_loss(traj_pred, traj_sampled)
            arap_loss = self.arap_knn_loss(static_pc, traj_pred, k=self.knn_k)

            losses.update(
                {
                    "fitting_loss": fitting_loss,
                    "corres_loss": corres_loss,
                    "knn_arap_loss": arap_loss,
                }
            )

            loss = (
                loss
                + fitting_loss * self.xyz_loss_weight
                + corres_loss * self.corres_loss_weight
                + arap_loss * 0.001
            )

            if self.use_tensorboard and self.step % self.log_interval == 0:
                if self.accelerator.is_main_process:
                    self.writer.write_dict(
                        {k: v.item() for k, v in losses.items()},
                        self.step,
                    )

            log_loss_dict(losses)
            self.accelerator.backward(loss)
            self.optimize()

    # =========================================================
    # Main training loop
    # =========================================================

    def run_loop(self):
        """Run the full training loop."""
        while not self.lr_anneal_steps or self.step <= self.lr_anneal_steps:
            start_time = time.time()

            batch = next(self.data)
            self.run_step(batch)

            step_time = time.time() - start_time
            logger.logkv_mean("step_time", step_time)

            if self.step % self.log_interval == 0 and self.accelerator.is_main_process:
                logger.dumpkvs()

            if self.step % self.save_interval == 0:
                self.save()

            self.step += 1

        if (self.step - 1) % self.save_interval != 0:
            self.save()


def main():
    """
    Entry point for training the deformation VAE.

    Responsibilities
    ----------------
    1. Configure distributed / NCCL environment
    2. Parse CLI arguments
    3. Build model from config
    4. Optionally load pretrained weights
    5. Create dataloader
    6. Launch training loop
    """

    import os

    # =========================================================
    # NCCL / Distributed environment settings
    # =========================================================

    # Increase NCCL timeout (seconds) to avoid timeout on long steps
    os.environ["NCCL_TIMEOUT"] = "1800"  # 30 minutes

    # Disable NCCL watchdog monitoring (fixes heartbeat stalls)
    os.environ["TORCH_NCCL_ENABLE_MONITORING"] = "0"

    # Optional: heartbeat timeout if monitoring is enabled
    os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "600"

    # =========================================================
    # Parse command line arguments
    # =========================================================
    args = create_argparser().parse_args()

    # =========================================================
    # Load model configuration
    # =========================================================
    model_config = OmegaConf.load(args.config)

    model = VariationalAutoEncoder(**model_config["model"])

    # =========================================================
    # Optional: load pretrained weights
    # =========================================================
    has_pretrain_weight = False

    if args.ckpt is not None:
        print("Loading pretrained weights from:", args.ckpt)

        state_dict = torch.load(args.ckpt, map_location="cpu")
        new_state_dict = OrderedDict()

        # Remove "module." prefix if checkpoint was saved with DDP
        for k, v in state_dict.items():
            new_key = k[7:] if k.startswith("module.") else k
            new_state_dict[new_key] = v

        msg = model.load_state_dict(new_state_dict, strict=False)
        print("Loaded VAE weights:", msg)

        has_pretrain_weight = True

    # =========================================================
    # Logger setup
    # =========================================================
    logger.configure(args.exp_name)

    logger.log("Model config:", model_config)
    logger.log(
        "Trainable params: {:.2f} M".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        )
    )

    # =========================================================
    # Data loader
    # =========================================================
    logger.log("Creating data loader...")

    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        train=True,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        txt_file=args.txt_file,
        num_pts=model_config["model"]["num_inputs"],
        sample_timesteps=args.sample_timesteps,
        text_condition=False,
    )

    # =========================================================
    # Start training
    # =========================================================
    logger.log("Starting training...")

    TrainLoop(
        model,
        data=data,
        batch_size=args.batch_size,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        weight_decay=args.weight_decay,
        use_tensorboard=args.use_tensorboard,
        render_l1_weight=args.render_l1_weight,
        has_pretrain_weight=has_pretrain_weight,
        knn_k=args.knn_k,
        beta=args.beta,
        xyz_loss_weight=args.xyz_loss_weight,
        corres_loss_weight=args.corres_loss_weight,
        kl_weight=args.kl_weight,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        args=args,
        auto_resume=args.auto_resume,
    ).run_loop()


# =============================================================
# Script entry
# =============================================================
if __name__ == "__main__":
    # Enables faster convolution benchmarking (good for fixed input sizes)
    # torch.backends.cudnn.benchmark = True
    main()
