# BiMotion: B-spline Motion for Text-guided Dynamic 3D Character Generation [TODO: Replace authors,arXiv,Project Page]

<div align="center">

[![Project Page](https://img.shields.io/badge/Project-Page-green?style=for-the-badge&logo=googlechrome&logoColor=white)](https://wangmiaowei.github.io/DecoupledGaussian.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2503.05484-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2503.05484v1)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-ffcc00?style=for-the-badge)](https://huggingface.co/datasets/miaoweiwang/BiMotion)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

**[Miaowei Wang](https://wangmiaowei.github.io/)<sup>1</sup> · [Yibo Zhang](https://github.com/YiboZhang)<sup>2</sup> · Rui Ma<sup>2</sup> · Weiwei Xu<sup>3</sup> · Changqing Zou<sup>3</sup> · Daniel Morris<sup>4</sup>**

<sup>1</sup>University of Edinburgh | <sup>2</sup>Jilin University | <sup>3</sup>Zhejiang University | <sup>4</sup>Michigan State University

</div>

![BiMotion Teaser](assets/teaser.png)


## 📑 0. Open-Source Plan

We plan to release all components of our project according to the following schedule:

- ⬜ Paper release – before 25/02/2026
- ⬜ Project page setup – before 25/02/2026
- ✅ Inference code & model weights – released
- ✅ BIMO dataset – released
- ✅ Training scripts – released
- ⬜ Dataset preprocessing scripts – before 27/02/2026

## 📝 1. Abstract

Text-guided dynamic 3D character generation has advanced rapidly, yet producing high-quality motion that faithfully reflects rich textual descriptions remains challenging. **BiMotion** addresses these limitations by representing motion with **continuous differentiable B-spline curves**, enabling more effective motion generation without modifying the capabilities of the underlying generative model.
### Key Contributions:
* **B-spline Representation**: A closed-form, Laplacian-regularized B-spline solver that efficiently compresses variable-length motion sequences into compact fixed-length control points.
* **Enhanced Geometry Adherence**: Introduces a normal-fusion strategy for input shape adherence along with correspondence-aware and local-rigidity losses.
* **BIMO Dataset**: A new dataset featuring diverse variable-length 3D motion sequences with rich, high-quality text annotations.


## 🛠️ 2. Environment Setup
> **Tested Environment**: CUDA 12.4 / 12.8, Ubuntu 22.04, Python 3.10

### 2.1. Basic Installation
```bash
# Clone Repository 
git clone --recursive https://github.com/wangmiaowei/BiMotion.git
cd BiMotion

# Create Environment
conda env create -f environment.yml
conda activate BiMotion
pip install -r requirements.txt
```

### 2.2. Specialized Dependencies

Installation of `pytorch3d` and `flash-attention` can be sensitive to environment versions. We recommend:

```bash
# PyTorch3D
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@V0.7.8"

# Flash Attention (Recommended pre-built wheel)
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```
If you encounter **GLIBC errors**, build from source:

```bash
pip uninstall flash_attn -y
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
pip install ninja
pip install --no-build-isolation .
```



### 2.3. Blender Integration

```bash
pip install bpy==3.6.0 --extra-index-url https://download.blender.org/pypi/

wget https://download.blender.org/release/Blender3.6/blender-3.6.0-linux-x64.tar.xz
tar -xf blender-3.6.0-linux-x64.tar.xz
rm blender-3.6.0-linux-x64.tar.xz
```


## 🚀 3. Evaluation & Inference

### 📂 3.1. Download Weights

Download the pre-trained checkpoints from Hugging Face:

```bash
WEIGHT_DIR="/path/to/your_model_weights"

hf download miaoweiwang/BiMotion \
  --repo-type dataset \
  --local-dir "$WEIGHT_DIR" \
  --include "model_weights/*.pt"
```

### 🏃 3.2. Running Inference

We provide two modes for motion generation depending on your input format:

#### Option A: 3D textured Object (.glb)

Generate animated results from a static GLB file.

```bash
python demo_evaluate_glb.py \
  --glb_name man \
  --prompt "He pivots, rotates his torso, sweeps one leg in a circular kick, then plants his foot and returns to guard." \
  --guidance_scale 3 \
  --export_format fbx \
  --seed 0 \
  --input_glbs_folder "visualize_experiments/input_glbs" \
  --save_path "visualize_experiments/results" \
  --vae_path "$WEIGHT_DIR/model_weights/vae_weight.pt" \
  --dit_path "$WEIGHT_DIR/model_weights/diffusion_weight.pt" \
  --azi 90 --ele 45 --generated_seq_length 20

```

> **Note:** `glb_name` is the filename in your input folder (exclude the `.glb` extension).

> **Result:** You will find `man_azi90_ele45.mp4` in your results folder as follows.

<p align="center">
  <img src="assets/demo.gif" alt="Demo GIF" width="600"/>
</p>

#### Option B: Untextured raw Mesh (.obj / .ply)

Suitable for meshes without textures.

```bash
python demo_evaluate_mesh.py \
  --mesh_name "hand" \
  --prompt "The fingers bend toward the palm and then straighten out." \
  --guidance_scale 3 \
  --seed 0 \
  --input_meshes_folder "visualize_experiments/input_meshes" \
  --save_path "visualize_experiments/results" \
  --vae_path "$WEIGHT_DIR/model_weights/vae_weight.pt" \
  --dit_path "$WEIGHT_DIR/model_weights/diffusion_weight.pt" \
  --generated_seq_length 20

```

> **Note:** The system automatically detects the mesh extension.

> **Result:** You will find 20 sequential mesh files in the output path.

### 💡 3.3. Tips for Better Results

* 🎲 **Try Multiple Seeds** — Outputs vary; experiment with different seeds.
* 🎚️ **Adjust Guidance Scale** — Increase to **5.0** if motion is too subtle.
* ✍️ **Refine Prompts** — Rephrase or use keywords similar to BIMO annotations.
* 🧭 **Check Mesh Orientation** — Ensure **Z-up** and **facing +Y**.




## 📊 4. Model Training

### 4.1. Download the BIMO Dataset

```bash
# Set the root directory where the dataset will be stored
# (Ensure you have ~3.5 TB of free space)

DATASET_PATH="YOUR_DATASET_PATH"

python dataset/download_dataset.py $DATASET_PATH
```

* `DATASET_PATH` should be the **root directory** where the BIMO dataset will be saved.
* Each motion subdirectory retains its original dataset name:

  * **DeformingThings4D** ([DT4D](https://github.com/rabbityl/DeformingThings4D))
  * **UID** from ObjaverseV1 ([paper](https://arxiv.org/abs/2212.08051))
  * **SHA256** from ObjaverseXL ([website](https://objaverse.allenai.org/))

**Note:**
For the training steps below, we provide a **SLURM script for single-node, multi-GPU training**.
For **multi-node, multi-GPU setups**, we recommend using 👉 [idr_accelerate](https://github.com/idriscnrs/idr_accelerate).


### 4.2. Train the B-spline VAE Model

1. Open the training script and modify the parameters that are already clearly marked:

```bash
scripts/script_vae_bspline.sh
```

2. Launch training (example: 1 node with 8 GPUs):

```bash
sbatch scripts/script_vae_bspline.sh
```


### 4.3. Compute Global Latent Statistics

We can compute the **global latent statistics (mean and std)** using **1 GPU**.
You may also run it via `srun` on a SLURM system if preferred.

```bash
#!/bin/bash

# Launch encoding of latent statistics
accelerate launch --num_processes 1 encode_latent_statistic.py \
  --batch_size 32 \
  --use_fp16 \
  --weight_decay 0.01 \
  # 🔴 TODO: set the checkpoint path of your trained VAE model
  --ckpt YOUR_EXPERIMENT_PATH/Bspline_motion_generation/VAE_Training/checkpoints/Your_Latest.pt \
  --config configs/vae_animate.yml \
  --start_idx 0 \
  --end_idx 38727 \
  --txt_file dataset/full_paths_V2.lst \
  # 🔴 TODO: set the root path to your dataset
  --data_dir YOUR_DATASET_PATH/bspline_motion_dataset/
```

**Notes:**

* Ensure the `--ckpt` points to the same checkpoint used in VAE training.
* Ensure the `--data_dir` matches the one used during VAE training.
* Batch size can be adjusted based on your GPU memory.
* This script saves **motion and static latent mean/std** in `dataset/data_statistics/`.



### 4.4. Train the Diffusion Model (DIT)

1. Open the training script and update the marked parameters:

```bash
scripts/script_dit_bspline.sh
```

2. Launch training (example: 1 node with 8 GPUs):

```bash
sbatch scripts/script_dit_bspline.sh
```

**Notes:** Keep the trained VAE checkpoints **the same as step 4.3** when providing input to the diffusion model.


## 📜 Citation [TODO: Replace]

If you find BiMotion useful in your research, please cite:

```bibtex
@article{wang2025bimotion,
  title={BiMotion: B-spline Motion for Text-guided Dynamic 3D Character Generation},
  author={Wang, Miaowei and Zhang, Yibo and Ma, Rui and Xu, Weiwei and Zou, Changqing and Morris, Daniel},
  journal={arXiv preprint arXiv:2503.05484},
  year={2026}
}

```