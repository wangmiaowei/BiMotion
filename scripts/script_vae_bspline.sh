#!/bin/bash
#
# ========================= SLURM SETTINGS =========================
#SBATCH --job-name=VAE_Train
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:8
#SBATCH --mem=400G
#SBATCH --time=7-02:00:00

#SBATCH --account=YOUR_ACCOUNT            # 🔴 TODO: set your SLURM account
#SBATCH --partition=YOUR_PARTITION        # 🔴 TODO: set your GPU partition

#SBATCH --output=PATH_TO_LOGS/VAE_%j.out  # 🔴 TODO: set log directory
#SBATCH --error=PATH_TO_LOGS/VAE_%j.err   # 🔴 TODO: set log directory


# ========================= ENVIRONMENT =========================
source /path/to/your/.bashrc              # 🔴 TODO: update path and set your CUDA_HOME, CUDA_PATH... 
conda activate BiMotion                   # 🔴 TODO: ensure env exists


# ========================= TRAINING =========================
accelerate launch --num_processes 8 \
  train_vae_bspline.py \
  --log_interval 32 \
  --batch_size 32 \
  --lr 2e-5 \
  --use_fp16 \
  --weight_decay 0.01 \
  --exp_name YOUR_EXPERIMENT_PATH/Bspline_motion_generation/VAE_Training \
  # 🔴 TODO: set experiment output directory "YOUR_EXPERIMENT_PATH"
  --save_interval 8000 \
  --config configs/vae_animate.yml \
  --use_tensorboard \
  --start_idx 0 \
  --end_idx 90000 \
  --txt_file dataset/full_paths_V2.lst \
  --data_dir YOUR_DATASET_PATH/bspline_motion_dataset/ \
  # 🔴 TODO: set dataset root path "YOUR_DATASET_PATH"
  --kl_weight 1e-5 \
  --render_l1_weight 1.0 \
  --xyz_loss_weight 1.0 \
  --gradient_accumulation_steps 1

  # --auto_resume   # Optional: resume training
