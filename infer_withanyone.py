# Copyright (c) 2025 Fudan University and StepFun Co., Ltd. All rights reserved.
# Modifications made under the Apache License, Version 2.0.

# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates. All rights reserved.

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
import dataclasses
from typing import Literal

from accelerate import Accelerator
from transformers import HfArgumentParser
from PIL import Image
import json
import itertools

from withanyone.flux.pipeline import WithAnyonePipeline

from util import extract_moref, general_face_preserving_resize, horizontal_concat, extract_object, FaceExtractor

import numpy as np

import random
import torch



from transformers import AutoModelForImageSegmentation
from torch.cuda.amp import autocast


BACK_UP_BBOXES_DOUBLE = [

    [[100,100,200,200], [300,100,400,200]], # 2 faces
    [[150,100,250,200], [300,100,400,200]]
]

BACK_UP_BBOXES = [ # for single face
    [[150,100,250,200]],
    [[100,100,200,200]],
    [[200,100,300,200]],
    [[250,100,350,200]],
    [[300,100,400,200]],
]






@dataclasses.dataclass
class InferenceArgs:
    prompt: str | None = None
    image_paths: list[str] | None = None
    eval_json_path: str | None = None
    offload: bool = False
    num_images_per_prompt: int = 1
    model_type: Literal["flux-dev", "flux-dev-fp8", "flux-schnell"] = "flux-dev"
    width: int = 512
    height: int = 512
    ref_size: int = -1
    num_steps: int = 25
    guidance: float = 4
    seed: int = 1234
    save_path: str = "output/inference"
    only_lora: bool = True
    concat_refs: bool = False
    lora_rank: int = 64
    data_resolution: int = 512
    save_iter: str = "500"
    use_rec: bool = False
    drop_text: bool = False
    use_matting: bool = False
    id_weight: float = 1.0
    siglip_weight: float = 1.0
    bbox_from_json: bool = True
    data_root: str = "./"
    # for lora
    additional_lora: str | None = None
    trigger: str = ""
    lora_weight: float = 1.0

    # path to the ipa model
    ipa_path: str = "./ckpt/ipa.safetensors"
    clip_path: str = "openai/clip-vit-large-patch14"
    t5_path: str = "xlabs-ai/xflux_text_encoders"
    flux_path: str = "black-forest-labs/FLUX.1-dev"
    siglip_path: str = "google/siglip-base-patch16-256-i18n"



def main(args: InferenceArgs):
    accelerator = Accelerator()

    face_extractor = FaceExtractor()

    pipeline = WithAnyonePipeline(
        args.model_type,
        args.ipa_path,
        accelerator.device,
        args.offload,
        only_lora=args.only_lora,
        face_extractor=face_extractor,
        additional_lora_ckpt=args.additional_lora,
        lora_weight=args.lora_weight,
        clip_path=args.clip_path,
        t5_path=args.t5_path,
        flux_path=args.flux_path,
        siglip_path=args.siglip_path,
    )

    
    
    if args.use_matting:
        birefnet = AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0', trust_remote_code=True).to('cuda', dtype=torch.bfloat16)


    assert args.prompt is not None or args.eval_json_path is not None, \
        "Please provide either prompt or eval_json_path"
    
    # if args.eval_json_path is not None:
    assert args.eval_json_path is not None, "Please provide eval_json_path. This script only supports batch inference."
    with open(args.eval_json_path, "rt") as f:
        data_dicts = json.load(f)
    data_root = args.data_root
    

    
    metadata = {}
    for (i, data_dict), j in itertools.product(enumerate(data_dicts), range(args.num_images_per_prompt)):


        if (i * args.num_images_per_prompt + j) % accelerator.num_processes != accelerator.process_index:
            continue
        # check if exist, if this image is already generated, skip it
        


        # if any of the images are None, skip this image
        if not os.path.exists(os.path.join(data_root, data_dict["image_paths"][0])):
            print(f"Image {i} does not exist, skipping...")
            print("path:", os.path.join(data_root, data_dict["image_paths"][0]))
            continue


        # extract bbox

        ori_img_path = data_dict.get("ori_img_path", None)
        # ori_img = Image.open(os.path.join(data_root, data_dict["ori_img_path"]))

        # basename = data_dict["ori_img_path"].split(".")[0].replace("/", "_") 
        if ori_img_path is None:
            basename = f"{i}_{j}"
        else:
            basename = data_dict["ori_img_path"].split(".")[0].replace("/", "_") 
            ori_img = Image.open(os.path.join(data_root, ori_img_path))
        bboxes = None
        print("Processing image", i, basename)
        if not args.bbox_from_json:
            if ori_img_path is None:
                print(f"Image {i} has no ori_img_path, cannot extract bbox, skipping...")
                continue
            ori_img = Image.open(os.path.join(data_root, ori_img_path))
            bboxes = face_extractor.locate_bboxes(ori_img)
            # cut bbox length to num of imgae_paths 
            if bboxes is not None and len(bboxes) > len(data_dict["image_paths"]):
                bboxes = bboxes[:len(data_dict["image_paths"])]
            elif bboxes is not None and len(bboxes) < len(data_dict["image_paths"]):
                print(f"Image {i} has less faces than image_paths, continuing...")
                continue
        else:
            json_file_path = os.path.join(data_root, basename + ".json")
            if os.path.exists(json_file_path):
                with open(json_file_path, "r") as f:
                    json_data = json.load(f)
                old_bboxes = json_data.get("bboxes", None)

                if old_bboxes is None:
                    print(f"Image {i} has no bboxes in json file, using backup bboxes...")
                    # v202 -> 2 faces v200_single -> 1 face
                    if "v202" in args.eval_json_path:
                        old_bboxes = random.choice(BACK_UP_BBOXES_DOUBLE)
                    elif "v200_single" in args.eval_json_path:
                        old_bboxes = random.choice(BACK_UP_BBOXES)
                    
                
                def recalculate_bbox( bbox, crop):
                    """
                    The image is cropped, so we need to recalculate the bbox.
                    bbox: [x1, y1, x2, y2]
                    crop: [x1c, y1c, x2c, y2c]
                    we just need to minus x1c and y1c from x1, y1,
                    """
                    x1, y1, x2, y2 = bbox
                    x1c, y1c, x2c, y2c = crop
                    return [x1-x1c, y1-y1c, x2-x1c, y2-y1c]
                crop = json_data.get("crop", None)
                rec_bboxes = [
                    recalculate_bbox(bbox, crop) if crop is not None else bbox for bbox in old_bboxes]
                # face_preserving_resize(image, bboxes, 512)
                if ori_img_path is not None:                    
                    _, bboxes = general_face_preserving_resize(ori_img, rec_bboxes, 512)
                # else we consider the provided bbox is already in target size
                else:
                    bboxes = rec_bboxes
            
        if bboxes is None:

            print(f"Image {i} has no face, bboxes are None, using backup bboxes..., basename: {basename}")

            bboxes = random.choice(BACK_UP_BBOXES_DOUBLE)
            print(f"Use backup bboxes: {bboxes}")


        ref_imgs = []
        arcface_embeddings = []
        if not args.use_rec:
            break_flag = False
            for img_path in data_dict["image_paths"]:
                img = Image.open(os.path.join(data_root, img_path))


                ref_img, arcface_embedding = face_extractor.extract(img)
                
                if ref_img is not None and arcface_embedding is not None:
                    if args.use_matting:
                        ref_img, _ = extract_object(birefnet, ref_img)
                    ref_imgs.append(ref_img)
                    arcface_embeddings.append(arcface_embedding)
                else:
                    print(f"Image {i} has no face, skipping...")
                    break_flag = True
                    break
            if break_flag:
                continue
        else:
            ref_imgs, arcface_embeddings = face_extractor.extract_refs(ori_img)
            
            if ref_imgs is None or arcface_embeddings is None:
                print(f"Image {i} has no face, skipping...")
                continue

            if args.use_matting:
                ref_imgs = [extract_object(birefnet, ref_img)[0] for ref_img in ref_imgs]


        # arcface to tensor
        arcface_embeddings = [torch.tensor(arcface_embedding) for arcface_embedding in arcface_embeddings]
        arcface_embeddings = torch.stack(arcface_embeddings).to(accelerator.device)


        # check, if any of the images are None, if so, skip this image
        if any(ref_img is None for ref_img in ref_imgs):
            print(f"Image {i}: failed to extract face, skipping...")
            continue


        if args.ref_size==-1:
            args.ref_size = 512 if len(ref_imgs)==1 else 320


        if args.trigger != "" and args.trigger is not None:
            data_dict["prompt"] = args.trigger + " " + data_dict["prompt"]


        image_gen = pipeline(
            prompt=data_dict["prompt"] if not args.drop_text else "",
            width=args.width,
            height=args.height,
            guidance=args.guidance,
            num_steps=args.num_steps,
            seed=args.seed,
            ref_imgs=ref_imgs,
            arcface_embeddings=arcface_embeddings,
            bboxes=[bboxes],
            id_weight=args.id_weight,
            siglip_weight=args.siglip_weight,

        )


        if args.concat_refs:
            image_gen = horizontal_concat([image_gen, *ref_imgs])

        os.makedirs(args.save_path, exist_ok=True)

        
        save_path = os.path.join(args.save_path, basename)
        os.makedirs(os.path.join(args.save_path, basename), exist_ok=True)

        # save refs, image_gen and original image
        for k, ref_img in enumerate(ref_imgs):
            ref_img.save(os.path.join(save_path, f"ref_{k}.jpg"))
        image_gen.save(os.path.join(save_path, f"out.jpg"))
        # original image
        ori_img = Image.open(os.path.join(data_root, data_dict["ori_img_path"])) if "ori_img_path" in data_dict else None
        if ori_img is not None:
            ori_img.save(os.path.join(save_path, f"ori.jpg"))
        # save config
        args_dict = vars(args)
        args_dict['prompt'] = data_dict["prompt"]
        args_dict["name"] = data_dict["name"] if "name" in data_dict else None
        json.dump(args_dict, open(os.path.join(save_path, f"meta.json"), 'w'), indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = HfArgumentParser([InferenceArgs])
    args = parser.parse_args_into_dataclasses()[0]
    main(args)


