# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates. All rights reserved.
# Copyright (c) 2024 Black Forest Labs and The XLabs-AI Team. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Literal

import torch
from einops import rearrange
from PIL import ExifTags, Image
import torchvision.transforms.functional as TVF


from withanyone.flux.modules.layers import (
    DoubleStreamBlockLoraProcessor,
    DoubleStreamBlockProcessor,
    SingleStreamBlockLoraProcessor,
    SingleStreamBlockProcessor,
)
from withanyone.flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from withanyone.flux.util import (
    load_ae,
    load_clip,
    load_flow_model_no_lora,
    load_flow_model_diffusers,
    load_t5,
)

from withanyone.flux.model import SiglipEmbedding, create_person_cross_attention_mask_varlen


def preprocess_ref(raw_image: Image.Image, long_size: int = 512):

    image_w, image_h = raw_image.size

    if image_w >= image_h:
        new_w = long_size
        new_h = int((long_size / image_w) * image_h)
    else:
        new_h = long_size
        new_w = int((long_size / image_h) * image_w)

    raw_image = raw_image.resize((new_w, new_h), resample=Image.LANCZOS)
    target_w = new_w // 16 * 16
    target_h = new_h // 16 * 16

    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    right = left + target_w
    bottom = top + target_h


    raw_image = raw_image.crop((left, top, right, bottom))


    raw_image = raw_image.convert("RGB")
    return raw_image


from io import BytesIO
import insightface
import numpy as np
class FaceExtractor:
    def __init__(self, model_path = "/data/MIBM/BenCon"):
        self.model = insightface.app.FaceAnalysis(name = "antelopev2", root=model_path, providers=['CUDAExecutionProvider'])
        self.model.prepare(ctx_id=0, det_thresh=0.45)
    
    def extract_moref(self, img, bboxes, face_size_restriction=1):
        """
        Extract faces from an image based on bounding boxes in JSON data.
        Makes each face square and resizes to 512x512.
        
        Args:
            img: PIL Image or image data
            json_data: JSON object with 'bboxes' and 'crop' information
            
        Returns:
            List of PIL Images, each 512x512, containing extracted faces
        """
        # Ensure img is a PIL Image
        try:
            if not isinstance(img, Image.Image) and not isinstance(img, torch.Tensor):
                img = Image.open(BytesIO(img))
            
            # bboxes = json_data['bboxes']
            # crop = json_data['crop']
            # print("len of bboxes:", len(bboxes))
            # Recalculate bounding boxes based on crop info
            # new_bboxes = [recalculate_bbox(bbox, crop) for bbox in bboxes]
            new_bboxes = bboxes
            # any of the face is less than 100 * 100, we ignore this image
            for bbox in new_bboxes:
                x1, y1, x2, y2 = bbox
                if x2 - x1 < face_size_restriction or y2 - y1 < face_size_restriction:
                    return []
            # print("len of new_bboxes:", len(new_bboxes))
            faces = []
            for bbox in new_bboxes:
                # print("processing bbox")
                # Convert coordinates to integers
                x1, y1, x2, y2 = map(int, bbox)
                
                # Calculate width and height
                width = x2 - x1
                height = y2 - y1
                
                # Make the bounding box square by expanding the shorter dimension
                if width > height:
                    # Height is shorter, expand it
                    diff = width - height
                    y1 -= diff // 2
                    y2 += diff - (diff // 2)  # Handle odd differences
                elif height > width:
                    # Width is shorter, expand it
                    diff = height - width
                    x1 -= diff // 2
                    x2 += diff - (diff // 2)  # Handle odd differences
                
                # Ensure coordinates are within image boundaries
                img_width, img_height = img.size
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img_width, x2)
                y2 = min(img_height, y2)
                
                # Extract face region
                face_region = img.crop((x1, y1, x2, y2))
                
                # Resize to 512x512
                face_region = face_region.resize((512, 512), Image.LANCZOS)
                
                faces.append(face_region)
            # print("len of faces:", len(faces))
            return faces
        except Exception as e:
            print(f"Error processing image: {e}")
            return []

    def __call__(self, img):
        # if np, get PIL, else, get np
        if isinstance(img, torch.Tensor):
            img_np = img.cpu().numpy()
            img_pil = Image.fromarray(img_np)
        elif isinstance(img, Image.Image):
            img_pil = img
            img_np = np.array(img)
        elif isinstance(img, np.ndarray):
            img_np = img
            img_pil = Image.fromarray(img)

        else:
            raise ValueError("Unsupported image format. Please provide a PIL Image or numpy array.")
        # Detect faces in the image
        faces = self.model.get(img_np)
        # use one 
        if len(faces) > 0:
            bboxes = []
            face = faces[0]
            bbox = face.bbox.astype(int)
            bboxes.append(bbox)
            return self.extract_moref(img_pil, bboxes)[0]
        else:
            print("Warning: No faces detected in the image.")
            return img_pil
            
    
class WithAnyonePipeline:
    def __init__(
        self,
        model_type: str,
        ipa_path: str,
        device: torch.device,
        offload: bool = False,
        only_lora: bool = False,
        no_lora: bool = False,
        lora_rank: int = 16,
        face_extractor = None,
        additional_lora_ckpt: str = None,
        lora_weight: float = 1.0,
        clip_path: str = "openai/clip-vit-large-patch14",
        t5_path: str = "xlabs-ai/xflux_text_encoders",
        flux_path: str = "black-forest-labs/FLUX.1-dev",
        siglip_path: str = "google/siglip-base-patch16-256-i18n",
        
    ):
        self.device = device
        self.offload = offload
        self.model_type = model_type

        self.clip = load_clip(clip_path, self.device)
        self.t5 = load_t5(t5_path, self.device, max_length=512)
        self.ae = load_ae(flux_path, model_type, device="cpu" if offload else self.device)
        self.use_fp8 = "fp8" in model_type


        if additional_lora_ckpt is not None:
            self.model = load_flow_model_diffusers(
                model_type,
                flux_path,
                ipa_path,
                device="cpu" if offload else self.device,
                lora_rank=lora_rank,
                use_fp8=self.use_fp8,
                additional_lora_ckpt=additional_lora_ckpt,
                lora_weight=lora_weight,

            ).to("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.model = load_flow_model_no_lora(
                model_type,
                flux_path,
                ipa_path,
                device="cpu" if offload else self.device,
                use_fp8=self.use_fp8
            )

        if face_extractor is not None:
            self.face_extractor = face_extractor
        else:
            self.face_extractor = FaceExtractor()

        self.siglip = SiglipEmbedding(siglip_path=siglip_path)


    def load_ckpt(self, ckpt_path):
        if ckpt_path is not None:
            from safetensors.torch import load_file as load_sft
            print("Loading checkpoint to replace old keys")
            # load_sft doesn't support torch.device
            if ckpt_path.endswith('safetensors'):
                sd = load_sft(ckpt_path, device='cpu')
                missing, unexpected = self.model.load_state_dict(sd, strict=False, assign=True)
            else:
                dit_state = torch.load(ckpt_path, map_location='cpu')
                sd = {}
                for k in dit_state.keys():
                    sd[k.replace('module.','')] = dit_state[k]
                missing, unexpected = self.model.load_state_dict(sd, strict=False, assign=True)
                self.model.to(str(self.device))
            print(f"missing keys: {missing}\n\n\n\n\nunexpected keys: {unexpected}")



    def __call__(
        self,
        prompt: str,
        width: int = 512,
        height: int = 512,
        guidance: float = 4,
        num_steps: int = 50,
        seed: int = 123456789,
        **kwargs
    ):
        width = 16 * (width // 16)
        height = 16 * (height // 16)

        device_type = self.device if isinstance(self.device, str) else self.device.type
        if device_type == "mps":
            device_type = "cpu"  # for support macos mps
        with torch.autocast(enabled=self.use_fp8, device_type=device_type, dtype=torch.bfloat16):
            return self.forward(
                prompt,
                width,
                height,
                guidance,
                num_steps,
                seed,       
                **kwargs
            )



    @torch.inference_mode
    def forward(
        self,
        prompt: str,
        width: int,
        height: int,
        guidance: float,
        num_steps: int,
        seed: int,
        ref_imgs: list[Image.Image] | None = None,
        arcface_embeddings: list[torch.Tensor] = None,
        bboxes = None,
        id_weight: float = 1.0,
        siglip_weight: float = 1.0,
    ):
        x = get_noise(
            1, height, width, device=self.device,
            dtype=torch.bfloat16, seed=seed
        )
        timesteps = get_schedule(
            num_steps,
            (width // 8) * (height // 8) // (16 * 16),
            shift=True,
        )
        if self.offload:
            self.ae.encoder = self.ae.encoder.to(self.device)

        if ref_imgs is None:
            siglip_embeddings = None
        else:
            siglip_embeddings = self.siglip(ref_imgs).to(self.device, torch.bfloat16).permute(1,0,2,3)
            # num_ref, (1), n, d


        if arcface_embeddings is not None:
            arcface_embeddings =  arcface_embeddings.unsqueeze(1)
            # num_ref, 1, 512
            arcface_embeddings = arcface_embeddings.to(self.device, torch.bfloat16)


        if self.offload:
            self.offload_model_to_cpu(self.ae.encoder)
            self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)


        inp_cond = prepare(t5=self.t5, clip=self.clip,img=x,prompt=prompt
                           )
        if self.offload:
            self.offload_model_to_cpu(self.t5, self.clip)
            self.model = self.model.to(self.device)



        img = inp_cond["img"]
        img_length = img.shape[1]
        ##### create mask for siglip and arcface #####
        if bboxes is not None:
            arc_mask = create_person_cross_attention_mask_varlen(
                    batch_size=img.shape[0],
                    # num_heads=self.params.num_heads,
                    # txt_len=text_length,
                    img_len=img_length,
                    id_len=8,  
                    bbox_lists=bboxes,
                    max_num_ids=len(bboxes[0]),
                    original_width=width,
                    original_height= height,
                ).to(img.device)
            siglip_mask = create_person_cross_attention_mask_varlen(
                batch_size=img.shape[0],
                # num_heads=self.params.num_heads,
                # txt_len=text_length,
                img_len=img_length,
                id_len=256+8,  
                bbox_lists=bboxes,
                max_num_ids=len(bboxes[0]),
                original_width=width,
                original_height= height,
            ).to(img.device)





        results = denoise(
            self.model,
            **inp_cond,
            timesteps=timesteps,
            guidance=guidance,
            arcface_embeddings=arcface_embeddings,
            siglip_embeddings=siglip_embeddings,
            bboxes=bboxes,
            id_weight=id_weight,
            siglip_weight=siglip_weight,
            img_height=height,
            img_width=width,
            arc_mask=arc_mask if bboxes is not None else None,
            siglip_mask=siglip_mask if bboxes is not None else None,
        )

        x = results


        if self.offload:
            self.offload_model_to_cpu(self.model)
            self.ae.decoder.to(x.device)
        x = unpack(x.float(), height, width)
        x = self.ae.decode(x)
        self.offload_model_to_cpu(self.ae.decoder)

        x1 = x.clamp(-1, 1)
        x1 = rearrange(x1[-1], "c h w -> h w c")
        output_img = Image.fromarray((127.5 * (x1 + 1.0)).cpu().byte().numpy())


        return output_img

    def offload_model_to_cpu(self, *models):
        if not self.offload: return
        for model in models:
            model.cpu()
            torch.cuda.empty_cache()
