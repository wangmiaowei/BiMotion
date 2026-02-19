#!/bin/bash
#
# ========================= SLURM SETTINGS =========================
#SBATCH --job-name=DIT_Train
#SBATCH --nodes=1
#SBATCH --cpus-per-task=36
#SBATCH --gres=gpu:8
#SBATCH --mem=320G
#SBATCH --time=7-02:00:00
#SBATCH --account=YOUR_ACCOUNT           # 🔴 TODO: set your SLURM account
#SBATCH --partition=YOUR_PARTITION       # 🔴 TODO: set your GPU partition
#SBATCH --output=PATH_TO_LOGS/DIT_%j.out # 🔴 TODO: set log directory
#SBATCH --error=PATH_TO_LOGS/DIT_%j.err  # 🔴 TODO: set log directory

# ========================= ENVIRONMENT =========================
# 1) Load conda environment
source /path/to/your/.bashrc             # 🔴 TODO: update path and set your CUDA_HOME, CUDA_PATH... 
conda activate BiMotion                  # 🔴 TODO: ensure environment exists


# ========================= TRAINING =========================
# For multi-GPU training on a single node using accelerate
accelerate launch --num_processes 8 train_dit_bspline.py \
  --log_interval 32 \
  --batch_size 32 \
  --lr 1e-4 \
  --use_fp16 \
  --weight_decay 0.0 \
  
  # 🔴 TODO: set path to pretrained VAE checkpoint consistent in your global statistic compute
  --vae_ckpt YOUR_EXPERIMENT_PATH/Bspline_motion_generation/VAE_Training/checkpoints/Your_Latest.pt \

  # 🔴 TODO: set experiment output directory
  --exp_name YOUR_EXPERIMENT_PATH/Bspline_motion_generation/DIT_training \

  --save_interval 10000 \
  --config_dit configs/diffusion.yml \
  --config_vae configs/vae_animate.yml \
  --use_tensorboard \
  --start_idx 0 \
  --end_idx 99999 \
  --txt_file dataset/full_paths_V2.lst \
  
  # 🔴 TODO: set dataset root path
  --data_dir YOUR_DATASET_PATH/bspline_motion_dataset/ \
  
  --uncond_p 0.1 \
  --sample_timesteps 16 \
  --gradient_accumulation_steps 1
  # --auto_resume   # Optional: resume from latest checkpoint