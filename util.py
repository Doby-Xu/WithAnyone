# Copyright (c) 2025 Fudan University. All rights reserved.


from io import BytesIO
import random
from PIL import Image
import numpy as np
import cv2
import insightface
import torch
from torchvision import transforms
from torch.cuda.amp import autocast

def face_preserving_resize(img, face_bboxes, target_size=512):
    """
    Resize image while ensuring all faces are preserved in the output.
    
    Args:
        img: PIL Image
        face_bboxes: List of [x1, y1, x2, y2] face coordinates
        target_size: Maximum dimension for resizing
        
    Returns:
        Tuple of (resized image, new_bboxes) or (None, None) if faces can't fit
    """
    
    x1_1, y1_1, x2_1, y2_1 = map(int, face_bboxes[0])
    x1_2, y1_2, x2_2, y2_2 = map(int, face_bboxes[1])
    min_x1 = min(x1_1, x1_2)
    min_y1 = min(y1_1, y1_2)
    max_x2 = max(x2_1, x2_2)
    max_y2 = max(y2_1, y2_2)
    # print("min_x1:", min_x1, "min_y1:", min_y1, "max_x2:", max_x2, "max_y2:", max_y2)
    # if any of them is negative, we cannot resize (Idk why this happens)
    if min_x1 < 0 or min_y1 < 0 or max_x2 < 0 or max_y2 < 0:
        return None, None

    # if face width is longer than the image height, or the face height is longer than the image width, we cannot resize
    face_width = max_x2 - min_x1
    face_height = max_y2 - min_y1
    if face_width > img.height or face_height > img.width:
        return None, None
        
    # Create a copy of face_bboxes for transformation
    new_bboxes = []
    for bbox in face_bboxes:
        new_bboxes.append(list(map(int, bbox)))
    
    # Choose cropping strategy based on image aspect ratio
    if img.width > img.height:
        # We need to crop width to make a square
        square_size = img.height
        
        # Calculate valid horizontal crop range that preserves all faces
        left_max = min_x1  # Leftmost position that includes leftmost face
        right_min = max_x2 - square_size  # Rightmost position that includes rightmost face
        
        if right_min <= left_max:
            # We can find a valid crop window
            start = random.randint(int(right_min), int(left_max)) if right_min < left_max else int(right_min)
            start = max(0, min(start, img.width - square_size))  # Ensure within image bounds
        else:
            # Faces are too far apart for square crop - use center of faces
            face_center = (min_x1 + max_x2) // 2
            start = max(0, min(face_center - (square_size // 2), img.width - square_size))
        
        cropped_img = img.crop((start, 0, start + square_size, square_size))
        
        # Adjust bounding box coordinates based on crop
        for bbox in new_bboxes:
            bbox[0] -= start  # x1 adjustment
            bbox[2] -= start  # x2 adjustment
            # y coordinates remain unchanged
    else:
        # We need to crop height to make a square
        square_size = img.width
        
        # Calculate valid vertical crop range that preserves all faces
        top_max = min_y1  # Topmost position that includes topmost face
        bottom_min = max_y2 - square_size  # Bottommost position that includes bottommost face
        
        if bottom_min <= top_max:
            # We can find a valid crop window
            start = random.randint(int(bottom_min), int(top_max)) if bottom_min < top_max else int(bottom_min)
            start = max(0, min(start, img.height - square_size))  # Ensure within image bounds
        else:
            # Faces are too far apart for square crop - use center of faces
            face_center = (min_y1 + max_y2) // 2
            start = max(0, min(face_center - (square_size // 2), img.height - square_size))
        
        cropped_img = img.crop((0, start, square_size, start + square_size))
        
        # Adjust bounding box coordinates based on crop
        for bbox in new_bboxes:
            bbox[1] -= start  # y1 adjustment
            bbox[3] -= start  # y2 adjustment
            # x coordinates remain unchanged
    
    # Calculate scale factor for resizing from square_size to target_size
    scale_factor = target_size / square_size
    
    # Adjust bounding boxes for the resize operation
    for bbox in new_bboxes:
        bbox[0] = int(bbox[0] * scale_factor)
        bbox[1] = int(bbox[1] * scale_factor)
        bbox[2] = int(bbox[2] * scale_factor)
        bbox[3] = int(bbox[3] * scale_factor)
    
    # Final resize to target size
    resized_img = cropped_img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    # Make sure all coordinates are within bounds (0 to target_size)
    # for bbox in new_bboxes:
    #     bbox[0] = max(0, min(bbox[0], target_size - 1))
    #     bbox[1] = max(0, min(bbox[1], target_size - 1))
    #     bbox[2] = max(1, min(bbox[2], target_size))
    #     bbox[3] = max(1, min(bbox[3], target_size))
    
    return resized_img, new_bboxes

def extract_moref(img, json_data, face_size_restriction=100):
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
        if not isinstance(img, Image.Image) and not isinstance(img, torch.Tensor) and not isinstance(img, JpegImageFile):
            img = Image.open(BytesIO(img))
        
        bboxes = json_data['bboxes']
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

def general_face_preserving_resize(img, face_bboxes, target_size=512):
    """
    Resize image while ensuring all faces are preserved in the output.
    Handles any number of faces (1-5).
    
    Args:
        img: PIL Image
        face_bboxes: List of [x1, y1, x2, y2] face coordinates
        target_size: Maximum dimension for resizing
        
    Returns:
        Tuple of (resized image, new_bboxes) or (None, None) if faces can't fit
    """
    # Find bounding region containing all faces
    if not face_bboxes:
        print("Warning: No face bounding boxes provided.")
        return None, None
        
    min_x1 = min(bbox[0] for bbox in face_bboxes)
    min_y1 = min(bbox[1] for bbox in face_bboxes)
    max_x2 = max(bbox[2] for bbox in face_bboxes)
    max_y2 = max(bbox[3] for bbox in face_bboxes)

    # Check for negative coordinates
    if min_x1 < 0 or min_y1 < 0 or max_x2 < 0 or max_y2 < 0:
        # print("Warning: Negative coordinates found in face bounding boxes.")
        # return None, None
        min_x1 = max(min_x1, 0)
        min_y1 = max(min_y1, 0)

    # Check if faces fit within image
    face_width = max_x2 - min_x1
    face_height = max_y2 - min_y1
    if face_width > img.height or face_height > img.width:
        # print("Warning: Faces are too large for the image dimensions.")
        # return None, None
        # Instead of returning None, we will crop the image to fit the faces
        max_x2 = min(max_x2, img.width)
        max_y2 = min(max_y2, img.height)
        min_x1 = max(min_x1, 0)
        min_y1 = max(min_y1, 0)
    # Create a copy of face_bboxes for transformation
    new_bboxes = []
    for bbox in face_bboxes:
        new_bboxes.append(list(map(int, bbox)))
    
    # Choose cropping strategy based on image aspect ratio
    if img.width > img.height:
        # Crop width to make a square
        square_size = img.height
        
        # Calculate valid horizontal crop range
        left_max = min_x1
        right_min = max_x2 - square_size
        
        if right_min <= left_max:
            # We can find a valid crop window
            start = random.randint(int(right_min), int(left_max)) if right_min < left_max else int(right_min)
            start = max(0, min(start, img.width - square_size))
        else:
            # Faces are too far apart - use center of faces
            face_center = (min_x1 + max_x2) // 2
            start = max(0, min(face_center - (square_size // 2), img.width - square_size))
        
        cropped_img = img.crop((start, 0, start + square_size, square_size))
        
        # Adjust bounding box coordinates
        for bbox in new_bboxes:
            bbox[0] -= start
            bbox[2] -= start
    else:
        # Crop height to make a square
        square_size = img.width
        
        # Calculate valid vertical crop range
        top_max = min_y1
        bottom_min = max_y2 - square_size
        
        if bottom_min <= top_max:
            start = random.randint(int(bottom_min), int(top_max)) if bottom_min < top_max else int(bottom_min)
            start = max(0, min(start, img.height - square_size))
        else:
            face_center = (min_y1 + max_y2) // 2
            start = max(0, min(face_center - (square_size // 2), img.height - square_size))
        
        cropped_img = img.crop((0, start, square_size, start + square_size))
        
        # Adjust bounding box coordinates
        for bbox in new_bboxes:
            bbox[1] -= start
            bbox[3] -= start
    
    # Calculate scale factor and adjust bounding boxes
    scale_factor = target_size / square_size
    
    for bbox in new_bboxes:
        bbox[0] = int(bbox[0] * scale_factor)
        bbox[1] = int(bbox[1] * scale_factor)
        bbox[2] = int(bbox[2] * scale_factor)
        bbox[3] = int(bbox[3] * scale_factor)
    
    # Final resize to target size
    resized_img = cropped_img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    # Make sure all coordinates are within bounds
    for bbox in new_bboxes:
        bbox[0] = max(0, min(bbox[0], target_size - 1))
        bbox[1] = max(0, min(bbox[1], target_size - 1))
        bbox[2] = max(1, min(bbox[2], target_size))
        bbox[3] = max(1, min(bbox[3], target_size))
    
    return resized_img, new_bboxes



def horizontal_concat(images):
    widths, heights = zip(*(img.size for img in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for img in images:
        new_im.paste(img, (x_offset, 0))
        x_offset += img.size[0]

    return new_im

def extract_object(birefnet, image):


    if image.mode != 'RGB':
        image = image.convert('RGB')
    input_images = transforms.ToTensor()(image).unsqueeze(0).to('cuda', dtype=torch.bfloat16)

    # Prediction
    with torch.no_grad(), autocast(dtype=torch.bfloat16):
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze().float()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image.size)
    
    # Create a binary mask (0 or 255)
    binary_mask = mask.convert("L")
    
    # Create a new image with black background
    result = Image.new("RGB", image.size, (0, 0, 0))
    
    # Paste the original image onto the black background using the mask
    result.paste(image, (0, 0), binary_mask)
    
    return result, mask

class FaceExtractor:
    def __init__(self):
        self.model = insightface.app.FaceAnalysis(name = "antelopev2", root="./")
        self.model.prepare(ctx_id=0, det_thresh=0.4)

    def extract(self, image: Image.Image):
        image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        res = self.model.get(image_np)
        if len(res) == 0:
            return None, None
        res = res[0]
        # print(res.keys())
        bbox = res["bbox"]
        # print("len(bbox)", len(bbox))

        moref = extract_moref(image, {"bboxes": [bbox]}, 1)
        # print("len(moref)", len(moref))
        return moref[0], res["embedding"]

    def locate_bboxes(self, image: Image.Image):
        image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        res = self.model.get(image_np)
        if len(res) == 0:
            return None
        bboxes = []
        for r in res:
            bbox = r["bbox"]
            bboxes.append(bbox)

        _, new_bboxes_ = general_face_preserving_resize(image, bboxes, 512)

        # ensure the bbox is square
        new_bboxes = []
        for bbox in new_bboxes_:
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            if w > h:
                diff = w - h
                y1 = max(0, y1 - diff // 2)
                y2 = min(512, y2 + diff // 2 + diff % 2)
            else:
                diff = h - w
                x1 = max(0, x1 - diff // 2)
                x2 = min(512, x2 + diff // 2 + diff % 2)
            new_bboxes.append([x1, y1, x2, y2])

        return new_bboxes
    def extract_refs(self, image: Image.Image):
        """
        Extracts reference faces from the image.
        Returns a list of reference images and their arcface embeddings.
        """
        image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        res = self.model.get(image_np)
        if len(res) == 0:
            return None, None
        ref_imgs = []
        arcface_embeddings = []
        for r in res:
            bbox = r["bbox"]
            moref = extract_moref(image, {"bboxes": [bbox]}, 1)
            ref_imgs.append(moref[0])
            arcface_embeddings.append(r["embedding"])
        return ref_imgs, arcface_embeddings
