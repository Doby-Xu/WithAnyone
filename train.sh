#!/bin/bash
    
# --- 1. Model Paths Configuration ---
# FLUX Base Model
export REPO_ID="/path/to/FLUX"
export REPO_ID_AE="/path/to/FLUX"
export REPO_FLOW="/path/to/FLUX/flux1-dev.safetensors"
export REPO_AE="/path/to/FLUX/vae/ae.safetensors"
export CKPT_PATH="/path/to/FLUX/flux1-dev.safetensors"
export AE_PATH="/path/to/FLUX/vae/ae.safetensors"

# Text Encoders
export T5_PATH="/path/to/ckpts/xflux"
export CLIP_PATH="/path/to/ckpts/clip"

# --- 2. Dataset Paths Configuration ---
export REF_DIR="/path/to/data/ref/untar/"
export REF_CLUSTER_DIR="/path/to/data/ref/npy/"
export REF_DICT_PATH="/path/to/data/ref/ref_dict.pth"

# --- 3. Launch Training ---
accelerate launch train.py \
    --project_dir ./output_dir \
    --checkpointing_steps 1000 \
    --batch_size 1 \
    --id_loss_weight 0.1 \
    --nce_loss_weight 0.1 \
    --cp_ratio 0.0 \
    --ipa_choice "arcface" \
    --use_GT_aligned_id_loss "True" \
    --dataset_path /path/to/your/dataset