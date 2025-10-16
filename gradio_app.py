# Copyright (c) 2025 Fudan University. All rights reserved.



import dataclasses
import json
import os
from pathlib import Path
from typing import List, Literal, Optional

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw

from withanyone.flux.pipeline import WithAnyonePipeline
from util import extract_moref, face_preserving_resize
import insightface


def captioner(prompt: str, num_person = 1) -> List[List[float]]:
    # use random choose for testing
    # within 512
    if num_person == 1:
        bbox_choices = [
            # expanded, centered and quadrant placements
            [96, 96, 288, 288],
            [128, 128, 320, 320],
            [160, 96, 352, 288],
            [96, 160, 288, 352],
            [208, 96, 400, 288],
            [96, 208, 288, 400],
            [192, 160, 368, 336],
            [64, 128, 224, 320],
            [288, 128, 448, 320],
            [128, 256, 320, 448],
            [80, 80, 240, 272],
            [196, 196, 380, 380],
            # originals
            [100, 100, 300, 300],
            [150, 50, 450, 350],
            [200, 100, 500, 400],
            [250, 150, 512, 450],
        ]
        return [bbox_choices[np.random.randint(0, len(bbox_choices))]]
    elif num_person == 2:
        # realistic side-by-side rows (no vertical stacks or diagonals)
        bbox_choices = [
            [[64, 112, 224, 304], [288, 112, 448, 304]],
            [[48, 128, 208, 320], [304, 128, 464, 320]],
            [[32, 144, 192, 336], [320, 144, 480, 336]],
            [[80, 96, 240, 288], [272, 96, 432, 288]],
            [[80, 160, 240, 352], [272, 160, 432, 352]],
            [[64, 128, 240, 336], [272, 144, 432, 320]],  # slight stagger, same row
            [[96, 160, 256, 352], [288, 160, 448, 352]],
            [[64, 192, 224, 384], [288, 192, 448, 384]],  # lower row
            [[16, 128, 176, 320], [336, 128, 496, 320]],  # near edges
            [[48, 120, 232, 328], [280, 120, 464, 328]],
            [[96, 160, 240, 336], [272, 160, 416, 336]],  # tighter faces
            [[72, 136, 232, 328], [280, 152, 440, 344]],  # small vertical offset
            [[48, 120, 224, 344], [288, 144, 448, 336]],  # asymmetric sizes
            [[80, 224, 240, 416], [272, 224, 432, 416]],  # bottom row
            [[80, 64, 240, 256], [272, 64, 432, 256]],    # top row
            [[96, 176, 256, 368], [288, 176, 448, 368]],
        ]
        return bbox_choices[np.random.randint(0, len(bbox_choices))]
    
    elif num_person == 3:
        # Non-overlapping 3-person layouts within 512x512
        bbox_choices = [
            [[20, 140, 150, 360], [180, 120, 330, 360], [360, 130, 500, 360]],
            [[30, 100, 160, 300], [190, 90, 320, 290], [350, 110, 480, 310]],
            [[40, 180, 150, 330], [200, 180, 310, 330], [360, 180, 470, 330]],
            [[60, 120, 170, 300], [210, 110, 320, 290], [350, 140, 480, 320]],
            [[50, 80, 170, 250], [200, 130, 320, 300], [350, 80, 480, 250]],
            [[40, 260, 170, 480], [190, 60, 320, 240], [350, 260, 490, 480]],
            [[30, 120, 150, 320], [200, 140, 320, 340], [360, 160, 500, 360]],
            [[80, 140, 200, 300], [220, 80, 350, 260], [370, 160, 500, 320]],
        ]
        return bbox_choices[np.random.randint(0, len(bbox_choices))]
    elif num_person == 4:
        # Non-overlapping 4-person layouts within 512x512
        bbox_choices = [
            [[20, 100, 120, 240], [140, 100, 240, 240], [260, 100, 360, 240], [380, 100, 480, 240]],
            [[40, 60, 200, 260], [220, 60, 380, 260], [40, 280, 200, 480], [220, 280, 380, 480]],
            [[180, 30, 330, 170], [30, 220, 150, 380], [200, 220, 320, 380], [360, 220, 490, 380]],
            [[30, 60, 140, 200], [370, 60, 480, 200], [30, 320, 140, 460], [370, 320, 480, 460]],
            [[20, 120, 120, 380], [140, 100, 240, 360], [260, 120, 360, 380], [380, 100, 480, 360]],
            [[30, 80, 150, 240], [180, 120, 300, 280], [330, 80, 450, 240], [200, 300, 320, 460]],
            [[30, 140, 110, 330], [140, 140, 220, 330], [250, 140, 330, 330], [370, 140, 450, 330]],
            [[40, 80, 150, 240], [40, 260, 150, 420], [200, 80, 310, 240], [370, 80, 480, 240]],
        ]
        return bbox_choices[np.random.randint(0, len(bbox_choices))]

        


class FaceExtractor:
    def __init__(self, model_path="./"):
        self.model = insightface.app.FaceAnalysis(name="antelopev2", root="./")
        self.model.prepare(ctx_id=0)

    def extract(self, image: Image.Image):
        """Extract single face and embedding from an image"""
        image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        res = self.model.get(image_np)
        if len(res) == 0:
            return None, None
        res = res[0]
        bbox = res["bbox"]
        moref = extract_moref(image, {"bboxes": [bbox]}, 1)
        return moref[0], res["embedding"]

    def extract_refs(self, image: Image.Image):
        """Extract multiple faces and embeddings from an image"""
        image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        res = self.model.get(image_np)
        if len(res) == 0:
            return None, None, None
        ref_imgs = []
        arcface_embeddings = []
        bboxes = []
        for r in res:
            bbox = r["bbox"]
            bboxes.append(bbox)
            moref = extract_moref(image, {"bboxes": [bbox]}, 1)
            ref_imgs.append(moref[0])
            arcface_embeddings.append(r["embedding"])
        
        # Convert bboxes to the correct format
        new_img, new_bboxes = face_preserving_resize(image, bboxes, 512)
        return ref_imgs, arcface_embeddings, new_bboxes, new_img


def resize_bbox(bbox, ori_width, ori_height, new_width, new_height):
    """Resize bounding box coordinates while preserving aspect ratio"""
    x1, y1, x2, y2 = bbox
    
    # Calculate scaling factors
    width_scale = new_width / ori_width
    height_scale = new_height / ori_height
    
    # Use minimum scaling factor to preserve aspect ratio
    min_scale = min(width_scale, height_scale)
    
    # Calculate offsets for centering the scaled box
    width_offset = (new_width - ori_width * min_scale) / 2
    height_offset = (new_height - ori_height * min_scale) / 2
    
    # Scale and adjust coordinates
    new_x1 = int(x1 * min_scale + width_offset)
    new_y1 = int(y1 * min_scale + height_offset)
    new_x2 = int(x2 * min_scale + width_offset)
    new_y2 = int(y2 * min_scale + height_offset)
    
    return [new_x1, new_y1, new_x2, new_y2]


def draw_bboxes_on_image(image, bboxes):
    """Draw bounding boxes on image for visualization"""
    if bboxes is None:
        return image
    
    # Create a copy to draw on
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    
    # Draw each bbox with a different color
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        # Draw label
        draw.text((x1, y1-15), f"Face {i+1}", fill=color)
    
    return img_draw


def create_demo(
    model_type: str = "flux-dev",
    ipa_path: str = "./ckpt/ipa.safetensors",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    offload: bool = False,
    lora_rank: int = 64,
    additional_lora_ckpt: Optional[str] = None,
    lora_scale: float = 1.0,
    clip_path: str = "openai/clip-vit-large-patch14",
    t5_path: str = "xlabs-ai/xflux_text_encoders",
    flux_path: str = "black-forest-labs/FLUX.1-dev",
):
    
    face_extractor = FaceExtractor()
    # Initialize pipeline and face extractor
    pipeline = WithAnyonePipeline(
        model_type,
        ipa_path,
        device,
        offload,
        only_lora=True,
        no_lora=True,
        lora_rank=lora_rank,
        additional_lora_ckpt=additional_lora_ckpt,
        lora_weight=lora_scale,
        face_extractor=face_extractor,
        clip_path=clip_path,
        t5_path=t5_path,
        flux_path=flux_path,
    )
    
    
    




    def parse_bboxes(bbox_text):
        """Parse bounding box text input"""
        if not bbox_text or bbox_text.strip() == "":
            return None
        
        try:
            bboxes = []
            lines = bbox_text.strip().split("\n")
            for line in lines:
                if not line.strip():
                    continue
                coords = [float(x) for x in line.strip().split(",")]
                if len(coords) != 4:
                    raise ValueError(f"Each bbox must have 4 coordinates (x1,y1,x2,y2), got: {line}")
                bboxes.append(coords)
            # print(f"\nParsed bboxes: {bboxes}\n")
            return bboxes
        except Exception as e:
            raise gr.Error(f"Invalid bbox format: {e}")
    
    def extract_from_multi_person(multi_person_image):
        """Extract references and bboxes from a multi-person image"""
        if multi_person_image is None:
            return None, None, None, None
        
        # Convert from numpy to PIL if needed
        if isinstance(multi_person_image, np.ndarray):
            multi_person_image = Image.fromarray(multi_person_image)
        
        ref_imgs, arcface_embeddings, bboxes, new_img = face_extractor.extract_refs(multi_person_image)
        
        if ref_imgs is None or len(ref_imgs) == 0:
            raise gr.Error("No faces detected in the multi-person image")
        
        # Limit to max 4 faces
        ref_imgs = ref_imgs[:4]
        arcface_embeddings = arcface_embeddings[:4]
        bboxes = bboxes[:4]
        
        # Create visualization with bboxes
        viz_image = draw_bboxes_on_image(new_img, bboxes)
        
        # Format bboxes as string for display
        bbox_text = "\n".join([f"{bbox[0]:.1f},{bbox[1]:.1f},{bbox[2]:.1f},{bbox[3]:.1f}" for bbox in bboxes])
        
        return ref_imgs, arcface_embeddings, bboxes, viz_image
    
    def process_and_generate(
        prompt, 
        width, height, 
        guidance, num_steps, seed,
        ref_img1, ref_img2, ref_img3, ref_img4,
        manual_bboxes_text, 
        multi_person_image,
        siglip_weight
    ):
        # Collect and validate reference images
        ref_images = [img for img in [ref_img1, ref_img2, ref_img3, ref_img4] if img is not None]
        
        if not ref_images:
            raise gr.Error("At least one reference image is required")
        
        # Process reference images to extract face and embeddings
        ref_imgs = []
        arcface_embeddings = []
        
        # Modified bbox handling logic
        if multi_person_image is not None:
            # Extract from multi-person image mode
            extracted_refs, extracted_embeddings, bboxes_, _ = extract_from_multi_person(multi_person_image)
            if extracted_refs is None:
                raise gr.Error("Failed to extract faces from the multi-person image")
            
            print("bboxes from multi-person image:", bboxes_)
            # need to resize bboxes from 512 512 to width height
            bboxes_ = [resize_bbox(bbox, 512, 512, width, height) for bbox in bboxes_]
            
        else:
            # Parse manual bboxes
            bboxes_ = parse_bboxes(manual_bboxes_text)
            
            # If no manual bboxes provided, use automatic captioner
            if bboxes_ is None:
                print("No multi-person image or manual bboxes provided. Using automatic captioner.")
                # Generate automatic bboxes based on image dimensions
                bboxes__ = captioner(prompt, num_person=len(ref_images))
                # resize to width height
                bboxes_ = [resize_bbox(bbox, 512, 512, width, height) for bbox in bboxes__]
                print("Automatically generated bboxes:", bboxes_)
        
        bboxes = [bboxes_] 

        for img in ref_images:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            
            ref_img, embedding = face_extractor.extract(img)
            if ref_img is None or embedding is None:
                raise gr.Error("Failed to extract face from one of the reference images")
            
            ref_imgs.append(ref_img)
            arcface_embeddings.append(embedding)
        

            
            
        if bboxes is None:
            raise gr.Error("Either provide manual bboxes or a multi-person image for bbox extraction")
        
        if len(bboxes[0]) != len(ref_imgs):
            raise gr.Error(f"Number of bboxes ({len(bboxes[0])}) must match number of reference images ({len(ref_imgs)})")
        
        # Convert arcface embeddings to tensor
        arcface_embeddings = [torch.tensor(embedding) for embedding in arcface_embeddings]
        arcface_embeddings = torch.stack(arcface_embeddings).to(device)
        
        # Generate image
        final_prompt = prompt 

        print(f"Generating image of size {width}x{height} with bboxes: {bboxes} ")

        if seed < 0:
            seed = np.random.randint(0, 1000000)
        
        image_gen = pipeline(
            prompt=final_prompt,
            width=width,
            height=height,
            guidance=guidance,
            num_steps=num_steps,
            seed=seed if seed > 0 else None,
            ref_imgs=ref_imgs,
            arcface_embeddings=arcface_embeddings,
            bboxes=bboxes,
            id_weight = 1 - siglip_weight,
            siglip_weight=siglip_weight,
        )
        
        # Save temp file for download
        temp_path = "temp_generated.png"
        image_gen.save(temp_path)

        # draw bboxes on the generated image for debug
        debug_face = draw_bboxes_on_image(image_gen, bboxes[0])
        
        return image_gen, debug_face, temp_path
    
    def update_bbox_display(multi_person_image):
        if multi_person_image is None:
            return None, gr.update(visible=True), gr.update(visible=False)
        
        try:
            _, _, _, viz_image = extract_from_multi_person(multi_person_image)
            return viz_image, gr.update(visible=False), gr.update(visible=True)
        except Exception as e:
            return None, gr.update(visible=True), gr.update(visible=False)

    # Create Gradio interface
    with gr.Blocks() as demo:
        # gr.Markdown("# WithAnyone Demo")
        # # gr.Markdown(badges_text)
        gr.HTML("""
            <div style="text-align: center; max-width: 900px; margin: 0 auto;">
                <h1 style="font-size: 2.0rem; font-weight: 700; display: block;">WithAnyone</h1>
                <h2 style="font-size: 1.5rem; font-weight: 300; margin-bottom: 1rem; display: block;">Official Gradio Demo for <a href="https://github.com/Doby-Xu/WithAnyone/tree/main">WithAnyone: Towards Controllable and ID-Consistent Image Generation</a></h2>
                <a href="https://doby-xu.github.io/WithAnyone/">[Project Page]</a>&ensp;
                <a href="https://github.com/Doby-Xu/WithAnyone/tree/main">[Code]</a>&ensp;
                <a href="https://huggingface.co/WithAnyone/WithAnyone">[Model]</a>&ensp;
                <a href="https://huggingface.co/datasets/WithAnyone/MultiID-Bench">[MultiID-Bench]</a>&ensp;
                <a href="https://huggingface.co/datasets/WithAnyone/MultiID-2M">[MultiID-2M]</a>
            </div>
            """)

        gr.Markdown("""
            ### 💡 How to Use This Demo:
            1. **Upload identity (ID) refernces.** The model will generate images containing the same identities. You can upload 1-4 reference images, each containing one clear human face. 
            2. **Provide detailed prompts describing the identity.** Please refer to **Tips** under the Generated Image field. WithAnyone is "controllable", so it needs more information to be controlled. 
            3. **[Recommended] Specify bounding boxes for each identity.** You can either: 
                - Upload a multi-person image containing all the desired identities, and the system will automatically detect faces and extract bounding boxes. (Recommended for most users)
                - Manually input bounding boxes in the text box (format: x1,y1,x2,y2, one per line). If left empty, the system will automatically generate bounding boxes based on the number of reference images. 
            4. **[Recommended] Adjust the "Spiritual Resemblance <--> Formal Resemblance" slider** to balance between preserving identity details and allowing creative freedom. Move it to the right for more resemblance to the reference images, or to the left for more creativity.
            5. **Click the "Generate" button to generate an image.** Enjoy!
            """)
        
        with gr.Row():
            
            with gr.Column():
                # Input controls
                generate_btn = gr.Button("Generate", variant="primary")
                with gr.Row():
                    with gr.Column():
                        siglip_weight = gr.Slider(0.0, 1.0, 1.0, step=0.05, label="Spiritual Resemblance <--> Formal Resemblance")
                with gr.Row():
                    prompt = gr.Textbox(label="Prompt", value="a person in a beautiful garden. High resolution, extremely detailed")
                    # use_text_prompt = gr.Checkbox(label="Use text prompt", value=True)
                    

                with gr.Row():
                    # Image generation settings
                    with gr.Column():
                        width = gr.Slider(512, 1024, 768, step=64, label="Generation Width")
                        height = gr.Slider(512, 1024, 768, step=64, label="Generation Height")
                    
                with gr.Accordion("Advanced Options", open=False):
                    with gr.Row():
                        num_steps = gr.Slider(1, 50, 25, step=1, label="Number of steps")
                        guidance = gr.Slider(1.0, 10.0, 4.0, step=0.1, label="Guidance")
                        seed = gr.Number(-1, label="Seed (-1 for random)")
                    
                        # start_at = gr.Slider(0, 50, 0, step=1, label="Start Identity at Step")
                        # end_at = gr.Number(-1, label="End Identity at Step (-1 for last)")
                        
                    # with gr.Row():
                    #     # skip_every = gr.Number(-1, label="Skip Identity Every N Steps (-1 for no skip)")
                        
                    #     siglip_weight = gr.Slider(0.0, 1.0, 1.0, step=0.05, label="Siglip Weight")
                    
                
                with gr.Row():
                    with gr.Column():
                        # Reference image inputs
                        gr.Markdown("### Face References (1-4 required)")
                        ref_img1 = gr.Image(label="Reference 1", type="pil")
                        ref_img2 = gr.Image(label="Reference 2", type="pil", visible=True)
                        ref_img3 = gr.Image(label="Reference 3", type="pil", visible=True) 
                        ref_img4 = gr.Image(label="Reference 4", type="pil", visible=True)
                    
                    with gr.Column():
                        # Bounding box inputs
                        gr.Markdown("### Mask Configuration (Option 1: Automatic)")
                        multi_person_image = gr.Image(label="Multi-person image (for automatic bbox extraction)", type="pil")
                        bbox_preview = gr.Image(label="Detected Faces", type="pil")
                        
                        
                        gr.Markdown("### Mask Configuration (Option 2: Manual)")
                        manual_bbox_input = gr.Textbox(
                            label="Manual Bounding Boxes (one per line, format: x1,y1,x2,y2)",
                            lines=4,
                            placeholder="100,100,200,200\n300,100,400,200"
                        )
                
                
                
                
                
                # generate_btn = gr.Button("Generate", variant="primary")
            
            with gr.Column():
                # Output display
                output_image = gr.Image(label="Generated Image")
                debug_face = gr.Image(label="Debug. Faces are expected to be generated in these boxes")
                download_btn = gr.File(label="Download full-resolution", type="filepath", interactive=False)

        # Examples section
        with gr.Row():
            
            gr.Markdown("""
            # Example Configurations
                        
            ### Tips for Better Results
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
            """)
        with gr.Row():
            examples = gr.Examples(
                examples=[
                    [
                        "a highly detailed portrait of a woman shown in profile. Her long, dark hair flows elegantly, intricately decorated with an abundant array of colorful flowers—ranging from soft light pinks and vibrant light oranges to delicate greyish blues—and lush green leaves, giving a sense of natural beauty and charm. Her bright blue eyes are striking, and her lips are painted a vivid red, adding to her alluring appearance. She is clad in an ornate garment with intricate floral patterns in warm hues like pink and orange, featuring exquisite detailing that speaks of fine craftsmanship. Around her neck, she wears a decorative choker with intricate designs, and dangling from her ears are beautiful blue teardrop earrings that catch the light. The background is filled with a profusion of flowers in various shades, creating a rich, vibrant, and romantic atmosphere that complements the woman's elegant and enchanting look.",  # prompt
                        1024, 1024,  # width, height
                        4.0, 25, 42,  # guidance, num_steps, seed
                        "assets/ref1.jpg", None, None, None,  # ref images
                        "240,180,540,500", None,  # manual_bbox_input, multi_person_image
                        # True,  # use_text_prompt
                        0.0,  # siglip_weight
                    ],
                    [
                        "High resolution anfd extremely detailed image of two elegant ladies enjoying a serene afternoon in a quaint Parisian café. They both wear fashionable trench coats and stylish berets, exuding an air of sophistication. One lady gently sips on a cappuccino, while her companion reads an intriguing novel with a subtle smile. The café is framed by charming antique furniture and vintage posters adorning the walls. Soft, warm light filters through a window, casting delicate shadows and creating a cozy, inviting atmosphere. Captured from a slightly elevated angle, the composition highlights the warmth of the scene in a gentle watercolor illustrative style. ",  # prompt
                        1024, 1024,  # width, height
                        4.0, 25, 42,  # guidance, num_steps, seed
                        "assets/ref1.jpg", "assets/ref2.jpg", None, None,  # ref images
                        "248,172,428,498\n554,128,728,464", None,  # manual_bbox_input, multi_person_image
                        # True,  # use_text_prompt
                        0.0,  # siglip_weight
                    ]
                ],
                inputs=[
                    prompt, width, height, guidance, num_steps, seed,
                    ref_img1, ref_img2, ref_img3, ref_img4,
                    manual_bbox_input, multi_person_image, 
                    siglip_weight
                ],
                label="Click to load example configurations"
            )
        # Set up event handlers
        multi_person_image.change(
            fn=update_bbox_display,
            inputs=[multi_person_image],
            outputs=[bbox_preview, manual_bbox_input, bbox_preview]
        )
        
        generate_btn.click(
            fn=process_and_generate,
            inputs=[
                prompt, width, height, guidance, num_steps, seed,
                ref_img1, ref_img2, ref_img3, ref_img4,
                manual_bbox_input, multi_person_image, 
                siglip_weight
            ],
            outputs=[output_image,debug_face, download_btn]
        )
    
    return demo


if __name__ == "__main__":
    from transformers import HfArgumentParser

    @dataclasses.dataclass
    class AppArgs:
        model_type: Literal["flux-dev", "flux-dev-fp8", "flux-schnell"] = "flux-dev"
        device: Literal["cuda", "cpu"] = (
            "cuda" if torch.cuda.is_available() 
            else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() 
            else "cpu"
        )
        offload: bool = False
        lora_rank: int = 64
        port: int = 7860
        additional_lora: str = None
        lora_scale: float = 1.0
        ipa_path: str = "WithAnyone/WithAnyone"
        clip_path: str = "openai/clip-vit-large-patch14"
        t5_path: str = "xlabs-ai/xflux_text_encoders"
        flux_path: str = "black-forest-labs/FLUX.1-dev"

    parser = HfArgumentParser([AppArgs])
    args = parser.parse_args_into_dataclasses()[0]

    demo = create_demo(
        args.model_type, 
        args.ipa_path,
        args.device, 
        args.offload,
        args.lora_rank,
        args.additional_lora,
        args.lora_scale,
        args.clip_path,
        args.t5_path,
        args.flux_path,
    )
    demo.launch(server_port=args.port)

