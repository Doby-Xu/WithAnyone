<div align="center">
  <h2>WithAnyone: Towards Controllable and ID-Consistent Image Generation</h2>
  <!-- <p>
    Hengyuan Xu, Wei Cheng, Peng Xing, Yixiao Fang, Shuhan Wu, Rui Wang,
  </p>
  <p>
     Xianfang Zeng, Daxin Jiang, Gang Yu, Xingjun Ma, Yu-Gang Jiang
  </p> -->
  <!-- <p><em>(† Project lead, ‡ Corresponding authors)</em></p> -->
  <!-- <p>Fudan University, StepFun</p> -->
  <p>
    <a href="https://arxiv.org/abs/2510.14975"><img src="https://img.shields.io/badge/arXiv-2510.14975-b31b1b.svg" alt="arXiv"/></a>
    <a href="https://doby-xu.github.io/WithAnyone/"><img src="https://img.shields.io/badge/Project-Page-blue.svg" alt="Project Page"/></a>
    <a href="https://huggingface.co/WithAnyone/WithAnyone"><img src="https://img.shields.io/badge/HuggingFace-Model-yellow.svg" alt="HuggingFace"/></a>
    <a href="https://huggingface.co/datasets/WithAnyone/MultiID-Bench"><img src="https://img.shields.io/badge/MultiID-Bench-Green.svg" alt="MultiID-Bench"/></a>
    <a href="https://huggingface.co/datasets/WithAnyone/MultiID-2M"><img src="https://img.shields.io/badge/MultiID_2M-Dataset-Green.svg" alt="MultiID-2M"/></a>
    <a href="https://huggingface.co/spaces/WithAnyone/WithAnyone_demo"><img src="https://img.shields.io/badge/Huggingface-Demo-blue.svg" alt="MultiID-2M"/></a>
  </p>

  
  
</div>

<!-- <p align="center">
  <a href="assets/teaser.pdf">
    <img src="assets/teaser.png" alt="Teaser" width="800"/>
  </a>
</p> -->

<p align="center">
  <a href="assets/withanyone.gif">
    <img src="assets/withanyone.gif" alt="Teaser" width="800"/>
  </a>
</p>
<!-- <p align="center"><em>(† Project lead, ‡ Corresponding authors)</em></p>  -->

Star us if you find this project useful! ⭐

## 🎉 Updates
- [10/2025] 🔥 [HuggingFace Space Demo](https://huggingface.co/spaces/WithAnyone/WithAnyone_demo) is online! Try it out!
- [10/2025] 🔥 [Model checkpoints](https://huggingface.co/WithAnyone/WithAnyone), [MultiID-Bench](https://huggingface.co/datasets/WithAnyone/MultiID-Bench), and [MultiID-2M](https://huggingface.co/datasets/WithAnyone/MultiID-2M) are released!
- [10/2025] 🔥 [Codebase](https://github.com/Doby-Xu/WithAnyone) and [Project Page](https://doby-xu.github.io/WithAnyone/) are relased!

## 🕒 Action Items
- [x] Inference scripts
- [x] WithAnyone - FLUX.1
- [x] WithAnyone.K.preview - FLUX.1 Kontext
- [x] WithAnyone.Ke.preview - FLUX.1 Kontext
- [ ] WithAnyone - FLUX.1 Kontext
- [x] MultiID-Bench
- [x] MultiID-2M Part 1
- [ ] MultiID-2M Part 2
- [ ] WithAnyone - FLUX.1 Krea
- [ ] Training codebase (As soon as the repo reaches 1k stars)

  

## 📑Introduction

**Highlight of WithAnyone**
- **Controllable**: WithAnyone aims to mitigate the "copy-paste" artifacts in face generation. Previous methods have a tendency to directly copy and paste the reference face onto the generated image, leading poor controllability of expressions, hairstyles, accessories, and even poses. They falls into a clear trade-off between similarity and copy-paste. The more similar the generated face is to the reference, the more copy-paste artifacts it has. WithAnyone is an attampt to break this trade-off. 
- **Multi-ID Generation**: WithAnyone can generate multiple given identities in a single image. With the help of controllable face generation, all generated faces can fit harmoniously in one group photo.


<div style="text-align:center; margin-top:12px;">
  <img src="assets/fidelity_vs_copypaste_v200_single.png" alt="Copy-Paste" style="width:70%; max-width:900px; height:auto; display:inline-block;">
</div>

<!-- <div style="display:flex; gap:10px; align-items:center;">
  <img src="assets/001.webp" alt="001" style="width:35%; height:auto;">
  <img src="assets/005.webp" alt="005" style="width:24%; height:auto;">
  <img src="assets/009.webp" alt="009" style="width:32%; height:auto;">
</div> -->




## ⚡️ Quick Start

### 🏰 Model Zoo
| Model | Description | Download |
|-|-|-|
| WithAnyone 1.0 - FLUX.1 | Main model with FLUX.1  | [HuggingFace](https://huggingface.co/WithAnyone/WithAnyone) |
| WithAnyone.K.preview - FLUX.1 Kontext | For t2i generation with FLUX.1 Kontext | [HuggingFace](https://huggingface.co/WithAnyone/WithAnyone) |
| WithAnyone.Ke.preview - FLUX.1 Kontext  | For face-editing with FLUX.1 Kontext | [HuggingFace](https://huggingface.co/WithAnyone/WithAnyone) |

If you just want to try it out, please use the base model WithAnyone - FLUX.1. The other models are for the following use cases:


<details>
<summary>WithAnyone.K</summary>
This is a preliminary version of WithAnyone with FLUX.1 Kontext. It can be used for text-to-image generation with multiple given identities. However, stability and quality are not as good as the base model. Please use it with caution. We are working on improving it.
</details>

<details>
<summary>WithAnyone.Ke</summary>
This is a face editing version of WithAnyone with FLUX.1 Kontext, leveraging the editing capabilities of FLUX.1 Kontext. Please use it with `gradio_edit.py` instead of `gradio_app.py`. It is still a preliminary version, and we are working on improving it.
</details>



### 🔧 Requirements

Use `pip install -r requirements.txt` to install the necessary packages.

### 🔧 Model Checkpoints

You can download the necessary model checkpoints in one of the two ways:

1. Directly run the inference scripts. The checkpoints will be downloaded automatically by the `hf_hub_download` function in the code to your `$HF_HOME` (default: `~/.cache/huggingface`).
2. Use `huggingface-cli download <repo name>` to download:
   - `black-forest-labs/FLUX.1-dev`
   - `xlabs-ai/xflux_text_encoders`
   - `openai/clip-vit-large-patch14`
   - `google/siglip-base-patch16-256-i18n`
   - `withanyone/withanyone`  
   Then run the inference scripts. You can download only the checkpoints you need to speed up setup and save disk space.  
   Example for `black-forest-labs/FLUX.1-dev`:
   - `huggingface-cli download black-forest-labs/FLUX.1-dev flux1-dev.safetensors`
   - `huggingface-cli download black-forest-labs/FLUX.1-dev ae.safetensors`  
   Ignore the text encoder in the `black-forest-labs/FLUX.1-dev` model repo (it is there for `diffusers` calls). All checkpoints together require about 51 GB of disk space (~40 in hub and 10 in xet).

After downloading, set the following arguments in the inference script to the local paths of the downloaded checkpoints:

```
--flux_path <path to flux1-dev.safetensors>
--clip_path <path to clip-vit-large-patch14>
--t5_path <path to xflux_text_encoders>
--siglip_path <path to siglip-base-patch16-256-i18n>
--ipa_path <path to withanyone>
```

<div style="color:#999; font-size:0.95em; margin-top:8px;">
We need to use the ArcFace model for face embedding. It will automatically be downloaded to `./models/`. However, there is an original bug. If you see an error like `assert 'detection' in self.models`, please manually move the model directory:
</div>
<pre style="color:#888; background:transparent; border:0; padding:0; margin-top:8px;">
mv models/antelopev2/ models/antelopev2_
mv models/antelopev2_/antelopev2/ models/antelopev2/
rm -rf models/antelopev2_, antelopev2.zip
</pre>

### ⚡️ Gradio Demo

The Gradio GUI demo is a good starting point to experiment with WithAnyone. Run it with:

```
python gradio_app.py --flux_path <path to flux1-dev directory> --ipa_path <path to withanyone directory> \
    --clip_path <path to clip-vit-large-patch14> \
    --t5_path <path to xflux_text_encoders> \
    --siglip_path <path to siglip-base-patch16-256-i18n> \
    --model_type "flux-dev" # or "flux-kontext" for WithAnyone.K
```



❗ WithAnyone requires face bounding boxes (bboxes). You should provide them to indicate where faces are. You can provide face bboxes in two ways:
1. Upload an example image with desired face locations in `Mask Configuration (Option 1: Automatic)`. The face bboxes will be extracted automatically, and faces will be generated in the same locations. Do not worry if the given image has a different resolution or aspect ratio; the face bboxes will be resized accordingly.
2. Input face bboxes directly in `Mask Configuration (Option 2: Manual)`. The format is `x1,y1,x2,y2` for each face, one per line.
3. <span style="color: #999;">(NOT recommended) leave both options empty, and the face bboxes will be randomly chosen from a pre-defined set. </span>

⭕ WithAnyone works well with LoRA. If you have any stylized LoRA checkpoints, use `--additional_lora_ckpt <path to lora checkpoint>` when launching the demo. The LoRA will be merged into the diffusion model. 
```
python gradio_app.py --flux_path <path to flux1-dev directory> --ipa_path <path to withanyone directory> \
    --additional_lora_ckpt <path to lora checkpoint> \
    --lora_scale 0.8 # adjust the weight as needed 
```

⭕ In `Advanced Options`, there is a slider controlling whether outputs are more "similar in spirit" or "similar in form" to the reference faces.  
- Move the slider to the right to preserve more details in the reference image (expression, makeup, accessories, hairstyle, etc.). Identity will also be better preserved.
- Move it to the left for more freedom and creativity. Stylization can be stronger, hair style and makeup can be changed.

<details>
<summary>How the slider works and some tips</summary>
The slider actually controlls the weight of SigLIP embedding and ArcFace embedding. The former preserves more mid-level semantic details, while the latter preserves more high-level identity information. 

SigLIP is a general image embedding model, capturing more than just faces, while ArcFace is a face-specific embedding model, capturing only identity information. 

When using high arcface weight (slider to the left), please add more description of the identity in the prompt, since arcface embedding may lose information like hairstyle, skin color, body build, age, etc. 
</details>

### 💡 Tips for Better Results
Be prepared for the first few runs as it may not be very satisfying. 

- Provide detailed prompts describing the identity. WithAnyone is "controllable", so it needs more information to be controlled. Here are something that might go wrong if not specified:
  - Skin color (generally the race is fine, but for asain descent, if not specified, it may generate darker skin tone);
  - Age (e.g., intead of "a man", try "a young man". If not specified, it may generate an older figure);
  - Body build;
  - Hairstyle;
  - Accessories (glasses, hats, earrings, etc.);
  - Makeup
- Use the slider to balance between "Resemblance in Spirit" and "Resemblance in Form" according to your needs. If you want to preserve more details in the reference image, move the slider to the right; if you want more freedom and creativity, move it to the left.
- Try it with LoRAs from community. They are usually fantastic.


## ⚙️ Batch Inference

You can use `infer_withanyone.py` for batch inference. The script supports generating multiple images with MultiID-Bench.

### Download MultiID-Bench

Download from [HuggingFace](https://huggingface.co/datasets/WithAnyone/MultiID-Bench).

```
huggingface-cli download WithAnyone/MultiID-Bench --repo-type dataset --local-dir <path to MultiID-Bench directory>
```

And convert the arrow file to a folder of images and a json file using `MultiID_Bench/hf2bench.py`:

```
python MultiID_Bench/parquet2bench.py --parquet <path to local dir> --output_dir <path to output directory>
``` 

You will get a folder with the following structure:

```
<output_dir>/
  ├── p1/untar
  ├── p2/untar
  ├── p3/
  ├── p1.json
  ├── p2.json
  └── p3.json
```

### Run Batch Inference

```
python infer_withanyone.py \
  --eval_json_path <path to MultiID-Bench subset json> \
  --data_root <path to MultiID-Bench subset images> \
  --save_path <path to save results>  \
  --use_matting True \ # set to True when siglip_weight > 0.0
  --siglip_weight 0.0 \ # Resemblance in Spirit vs Resemblance in Form, higher means more similar to reference
  --id_weight 1.0 \ # usually, set it to 1 - id_weight, higher means more controllable
  --t5_path <path to xflux_text_encoders> \
  --clip_path <path to clip-vit-large-patch14> \
  --ipa_path <path to withanyone> \
  --flux_path <path to flux1-dev> 

```
Where the data_root should be p1/untar, p2/untar, or p3/ depending on which subset you want to evaluate. The eval_json_path should be the corresponding json file converted from the parquet file.

## ⚙️ Face Edit with FLUX.1 Kontext
You can use `gradio_edit.py` for face editing with FLUX.1 Kontext and WithAnyone.Ke. 
<p align="center">
  <a href="assets/kontext.jpg">
    <img src="assets/kontext.jpg" alt="Face Edit" width="800"/>
  </a>
</p>
Run it with:

```
python gradio_edit.py --flux_path <path to flux1-dev directory> --ipa_path <path to withanyone directory> \
    --clip_path <path to clip-vit-large-patch14> \
    --t5_path <path to xflux_text_encoders> \
    --siglip_path <path to siglip-base-patch16-256-i18n> \
    --model_type "flux-kontext"
```


## 📜 License and Disclaimer

The **code** of WithAnyone is released under the [**Apache License 2.0**](https://www.apache.org/licenses/LICENSE-2.0), while the WithAnyone **model and associated datasets** are made available **solely for non-commercial academic research purposes**.

- **License Terms:**  
  The WithAnyone model is distributed under the [**FLUX.1 [dev] Non-Commercial License v1.1.1**](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md). All underlying base models remain governed by their respective original licenses and terms, which shall continue to apply in full. Users must comply with all such applicable licenses when using this project.

- **Permitted Use:**  
  This project may be used for lawful academic research, analysis, and non-commercial experimentation only. Any form of commercial use, redistribution for profit, or application that violates applicable laws, regulations, or ethical standards is strictly prohibited.

- **User Obligations:**  
  Users are solely responsible for ensuring that their use of the model and dataset complies with all relevant laws, regulations, institutional review policies, and third-party license terms.

- **Disclaimer of Liability:**  
  The authors, developers, and contributors make no warranties, express or implied, regarding the accuracy, reliability, or fitness of this project for any particular purpose. They shall not be held liable for any damages, losses, or legal claims arising from the use or misuse of this project, including but not limited to violations of law or ethical standards by end users.

- **Acceptance of Terms:**  
  By downloading, accessing, or using this project, you acknowledge and agree to be bound by the applicable license terms and legal requirements, and you assume full responsibility for all consequences resulting from your use.


## 🌹 Acknowledgement
We thank the following prior art for their excellent open source work: 
- [PuLID](https://github.com/ToTheBeginning/PuLID)
- [UNO](https://github.com/bytedance/UNO)
- [UniPortrait](https://github.com/junjiehe96/UniPortrait)
- [InfiniteYou](https://github.com/bytedance/InfiniteYou)
- [DreamO](https://github.com/bytedance/DreamO)
- [UMO](https://github.com/bytedance/UMO)

## 📑 Citation

If you find this project useful in your research, please consider citing:

```bibtex
@article{xu2025withanyone,
  title={WithAnyone: Towards Controllable and ID-Consistent Image Generation}, 
  author={Hengyuan Xu and Wei Cheng and Peng Xing and Yixiao Fang and Shuhan Wu and Rui Wang and Xianfang Zeng and Gang Yu and Xinjun Ma and Yu-Gang Jiang},
  journal={arXiv preprint arxiv:2510.14975},
  year={2025}
}
```


