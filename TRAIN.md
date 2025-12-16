# Training Instructions for WithAnyone

Welcome to the training guide for **WithAnyone**. If you encounter any issues during the process, please feel free to raise an issue in the repository.

## Quick Start

Follow these steps to set up the environment and launch training quickly.

1.  **Convert ArcFace Model:**
    Using [onnx2torch](https://github.com/ENOT-AutoDL/onnx2torch), convert the ArcFace model `antelopev2/glintr100.onnx` to a PyTorch compatible format: `glintr100.pth`.

2.  **Prepare Dataset and Base Model:**
    Download the **Multi-ID 2M** dataset. Untar the archives into a reference folder and a cluster center folder.
    *Note: You no longer need to modify source code for paths. All paths are now configured via environment variables in the training script.*

3.  **Construct Negative Pool Index:**
    Run the script `withanyone/utils/0815_neg_pool_construct.py` to generate `ref_dict.pth`. This file is required for the extended negative pool.
    
    ```bash
    python withanyone/utils/0815_neg_pool_construct.py \
        --ref_dir /path/to/ref/untar/ \
        --save_path /path/to/ref/ref_dict.pth
    ```

4.  **[Optional] Custom Dataloader:**
    You may replace the provided minimal dataloader with your own implementation if desired.

5.  **Launch Training:**
    Create a `train.sh` script to set up environment variables and launch the training.
    
    **Example `train.sh`:**
    ```bash
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
    ```

    Run the script:
    ```bash
    chmod +x train.sh
    ./train.sh
    ```

6. **Merge IPA:**
    SigLIP IPA and ArcFace IPA are trained separately. After training, you can merge the two IPAs using the provided script `withanyone/utils/merge_ipa.py`.
   

---

## Detailed Instructions

### 1. Dataset, Base Model, and ArcFace Model Preparation

**Reference Preparation & Indexing**
To efficiently draw references during training, the dataset references must be untarred in advance. Additionally, an index is required for efficient negative pool construction.

1.  Run the construction script with your paths:
    ```bash
    python withanyone/utils/0815_neg_pool_construct.py \
        --ref_dir /mnt/data/ref/untar/ \
        --save_path /mnt/data/ref/ref_dict.pth \
        --num_workers 16
    ```
2.  Ensure the `REF_DIR` and `REF_DICT_PATH` environment variables in your `train.sh` point to these locations.

**ArcFace Model Conversion**
We require the PyTorch version of ArcFace for ID loss and gradient computation. Please follow the instructions in [onnx2torch](https://github.com/ENOT-AutoDL/onnx2torch) to convert the ONNX model to PyTorch format.

**FLUX, CLIP, and T5 Models Paths**
Instead of modifying code, set the following environment variables in your `train.sh`:
*   `REPO_ID`, `REPO_FLOW`, `REPO_AE`: Paths to FLUX model components.
*   `T5_PATH`: Path to the T5 encoder.
*   `CLIP_PATH`: Path to the CLIP encoder.

If these variables are not set, the code will attempt to download models from HuggingFace by default.

### 2. [Optional] Dataloader Preparation

A minimal, runnable dataloader is provided in `withanyone/negativeloader_full.py`. It reads paths from environment variables (`REF_DIR`, `REF_CLUSTER_DIR`, `REF_DICT_PATH`).

**Note:** This dataloader is **NOT** the original version used in our experiments (which remains closed-source due to confidentiality).
*   **Limitations:** It resizes all multi-resolution images to 512x512 and only samples 2-person data records.
*   **Recommendation:** We strongly recommend implementing your own dataloader that supports mixed-resolution images and mixed-number-of-persons buffer collation for optimal results.

### 3. Training Arguments

Below are the key arguments for `train.py`. For a complete list of arguments, please refer to the source code.

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--id_loss_weight` | `0.1` | Weight for the ID loss. |
| `--nce_loss_weight` | `0.1` | Weight for the NCE loss. |
| `--exneg_pool` | `"True"` | Whether to use the extended negative pool. |
| `--cp_ratio` | `0.0` | Ratio for paired data. Default is `0.0` for the reconstruction-only pre-training stage. We recommend setting this to `0.5` for the subsequent paired tuning stage. |
| `--ipa_choice` | `"arcface"` | Choice of IPA embedder. We train ArcFace IPA and SigLIP IPA separately. |
| `--use_GT_aligned_id_loss` | `"False"` | Whether to use GT aligned images for ID loss. Recommended: `"False"` for reconstruction-only pre-training, and `"True"` for subsequent stages. |
| `--text_dropout` | `0.1` | Probability to drop the text condition. Recommended: Set to `1` during the first 20k steps. |
| `--drop_content` | `""` | Content used to replace the dropped text condition. In our experiments, we used `"n person(s)"` for the first 20k steps (where *n* is the number of persons in the image). |

### 4. Merging SigLIP and ArcFace IPAs

In our experiments, we trained SigLIP IPA and ArcFace IPA separately. After training, we merged the two IPAs using the script `withanyone/utils/merge_ipa.py`.