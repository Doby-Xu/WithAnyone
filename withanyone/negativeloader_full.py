# --coding:utf-8--

import os
import sys
import json
import random
import time
import concurrent.futures
from io import BytesIO

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as F
import webdataset as wds
from PIL import Image, ImageFilter
from PIL.JpegImagePlugin import JpegImageFile
from tqdm import tqdm

# Add current directory to path
sys.path.insert(0, os.getcwd())

# --- Global Configuration ---
REF_DIR = os.getenv("REF_DIR", "/data/MultiID-2M//ref/untar/")
REF_CLUSTER_DIR = os.getenv("REF_CLUSTER_DIR", "/data/MultiID-2M//ref/npy/")
REF_DICT_PATH = os.getenv("REF_DICT_PATH", "/data/MultiID-2M//ref/ref_dict.pth")
NUM_WORKERS_BUILD = 8  # Workers for building ref dict

# Cosine similarity for reference selection
cos = nn.CosineSimilarity(dim=0)

# --- Helper Functions: Initialization ---

def load_or_build_ref_dict(save_path, ref_dir):
    """
    Loads the reference dictionary from disk. 
    If it doesn't exist, builds it from the reference directory.
    """
    if os.path.exists(save_path):
        print(f"Loading REF_DICT from {save_path}...")
        return torch.load(save_path, map_location="cpu")
    
    print(f"REF_DICT not found at {save_path}. Building from {ref_dir}...")
    
    def process_person(name):
        """Process embeddings for a single person directory"""
        try:
            person_dir = os.path.join(ref_dir, name)
            if not os.path.isdir(person_dir):
                return None
            
            ref_files = [f for f in os.listdir(person_dir) if f.endswith(".npy")]
            if not ref_files:
                return None
                
            # Limit to 200 files to save memory/time
            if len(ref_files) > 200:
                ref_files = np.random.choice(ref_files, 200, replace=False)
                
            emb_list = []
            for f in ref_files:
                arr = np.load(os.path.join(person_dir, f), allow_pickle=True).item()
                # Handle different key names in npy files
                emb = arr.get("embeddings", arr.get("embedding"))[0]
                emb_list.append(emb)
                
            np_arr = np.stack(emb_list)  # (N, 512)
            # Use bfloat16 for memory efficiency
            tensor = torch.from_numpy(np_arr).to(torch.bfloat16)  
            
            return (name, tensor)
        except Exception as e:
            print(f"Error processing {name}: {str(e)}")
            return None

    # Get all person directories
    if not os.path.exists(ref_dir):
        print(f"Error: Reference directory {ref_dir} does not exist.")
        return {}

    person_dirs = [name for name in sorted(os.listdir(ref_dir)) 
                   if os.path.isdir(os.path.join(ref_dir, name))]

    ref_dict = {}
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS_BUILD) as executor:
        future_to_person = {executor.submit(process_person, name): name for name in person_dirs}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_person), 
                          total=len(future_to_person), 
                          desc="Processing embeddings"):
            result = future.result()
            if result:
                name, tensor = result
                ref_dict[name] = tensor

    print(f"Processing completed in {time.time() - start_time:.2f} seconds")
    print(f"Processed {len(ref_dict)} people")
    
    print(f"Saving to {save_path} ...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(ref_dict, save_path)
    
    return ref_dict

# Initialize Globals
REF_DICT = load_or_build_ref_dict(REF_DICT_PATH, REF_DIR)
# Ensure REF_NAME_LIST only contains valid keys from the dictionary
REF_NAME_LIST = list(REF_DICT.keys())

# --- Helper Functions: Image Processing ---

def recalculate_bbox(bbox, crop):
    """
    Adjusts bounding box coordinates after an image crop.
    bbox: [x1, y1, x2, y2], crop: [x1c, y1c, x2c, y2c]
    """
    x1, y1, x2, y2 = bbox
    x1c, y1c, x2c, y2c = crop
    return [x1-x1c, y1-y1c, x2-x1c, y2-y1c]



def extract_moref(img, json_data, face_size_restriction=100):
    """
    Extract faces from a reference image based on JSON bboxes.
    Resizes extracted faces to 512x512.
    """
    try:
        if not isinstance(img, (Image.Image, torch.Tensor, JpegImageFile)):
            img = Image.open(BytesIO(img))
        
        bboxes = json_data['bboxes']
        
        # Filter small faces
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            if x2 - x1 < face_size_restriction or y2 - y1 < face_size_restriction:
                return []

        faces = []
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            width = x2 - x1
            height = y2 - y1
            
            # Make square
            if width > height:
                diff = width - height
                y1 -= diff // 2
                y2 += diff - (diff // 2)
            elif height > width:
                diff = height - width
                x1 -= diff // 2
                x2 += diff - (diff // 2)
            
            # Boundary checks
            img_width, img_height = img.size
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_width, x2)
            y2 = min(img_height, y2)
            
            face_region = img.crop((x1, y1, x2, y2))
            face_region = face_region.resize((512, 512), Image.LANCZOS)
            faces.append(face_region)
            
        return faces
    except Exception as e:
        print(f"Error processing image in extract_moref: {e}")
        return []

def general_face_preserving_resize(img, face_bboxes, target_size=512):
    """
    Resize image ensuring all faces are preserved.
    Returns (resized_img, new_bboxes) or (None, None).
    """
    if not face_bboxes:
        return None, None
        
    min_x1 = min(bbox[0] for bbox in face_bboxes)
    min_y1 = min(bbox[1] for bbox in face_bboxes)
    max_x2 = max(bbox[2] for bbox in face_bboxes)
    max_y2 = max(bbox[3] for bbox in face_bboxes)

    if min_x1 < 0 or min_y1 < 0 or max_x2 < 0 or max_y2 < 0:
        return None, None

    face_width = max_x2 - min_x1
    face_height = max_y2 - min_y1
    if face_width > img.height or face_height > img.width:
        return None, None
        
    new_bboxes = [list(map(int, bbox)) for bbox in face_bboxes]
    
    # Crop strategy
    if img.width > img.height:
        square_size = img.height
        left_max = min_x1
        right_min = max_x2 - square_size
        
        if right_min <= left_max:
            start = random.randint(int(right_min), int(left_max)) if right_min < left_max else int(right_min)
            start = max(0, min(start, img.width - square_size))
        else:
            face_center = (min_x1 + max_x2) // 2
            start = max(0, min(face_center - (square_size // 2), img.width - square_size))
        
        cropped_img = img.crop((start, 0, start + square_size, square_size))
        for bbox in new_bboxes:
            bbox[0] -= start
            bbox[2] -= start
    else:
        square_size = img.width
        top_max = min_y1
        bottom_min = max_y2 - square_size
        
        if bottom_min <= top_max:
            start = random.randint(int(bottom_min), int(top_max)) if bottom_min < top_max else int(bottom_min)
            start = max(0, min(start, img.height - square_size))
        else:
            face_center = (min_y1 + max_y2) // 2
            start = max(0, min(face_center - (square_size // 2), img.height - square_size))
        
        cropped_img = img.crop((0, start, square_size, start + square_size))
        for bbox in new_bboxes:
            bbox[1] -= start
            bbox[3] -= start
    
    # Resize
    scale_factor = target_size / square_size
    for bbox in new_bboxes:
        bbox[0] = int(bbox[0] * scale_factor)
        bbox[1] = int(bbox[1] * scale_factor)
        bbox[2] = int(bbox[2] * scale_factor)
        bbox[3] = int(bbox[3] * scale_factor)
    
    resized_img = cropped_img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    # Clamp coordinates
    for bbox in new_bboxes:
        bbox[0] = max(0, min(bbox[0], target_size - 1))
        bbox[1] = max(0, min(bbox[1], target_size - 1))
        bbox[2] = max(1, min(bbox[2], target_size))
        bbox[3] = max(1, min(bbox[3], target_size))
    
    return resized_img, new_bboxes

def extract_faces_variable(img, json_data, num_faces=None):
    """
    Extract faces from current image (Self-Reference).
    """
    if not isinstance(img, (Image.Image, torch.Tensor, JpegImageFile)):
        img = Image.open(BytesIO(img))
    
    bboxes = json_data['bboxes']
    crop = json_data.get('crop', [0, 0, img.width, img.height])
    
    if num_faces is not None:
        bboxes = bboxes[:num_faces]
    
    new_bboxes = [recalculate_bbox(bbox, crop) for bbox in bboxes]
    
    for bbox in new_bboxes:
        x1, y1, x2, y2 = bbox
        if x2 - x1 < 100 or y2 - y1 < 100:
            return []
    
    faces = []
    for bbox in new_bboxes:
        x1, y1, x2, y2 = map(int, bbox)
        width = x2 - x1
        height = y2 - y1
        
        if width > height:
            diff = width - height
            y1 -= diff // 2
            y2 += diff - (diff // 2)
        elif height > width:
            diff = height - width
            x1 -= diff // 2
            x2 += diff - (diff // 2)
        
        img_width, img_height = img.size
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_width, x2)
        y2 = min(img_height, y2)
        
        face_region = img.crop((x1, y1, x2, y2))
        face_region = face_region.resize((512, 512), Image.LANCZOS)
        faces.append(face_region)
    
    return faces

# --- Main Processing Logic ---
def _decode_item_data(item):
    """Decodes raw bytes in the item dictionary."""
    if "json" in item and isinstance(item["json"], bytes):
        item["json"] = json.loads(item["json"].decode("utf-8"))
    if "npy" in item and isinstance(item["npy"], bytes):
        item["npy"] = np.load(BytesIO(item["npy"]), allow_pickle=True).item()
    if "jpg" in item and isinstance(item["jpg"], bytes):
        item["jpg"] = Image.open(BytesIO(item["jpg"]))
    return item

def _get_person_names(json_data, num_faces):
    """Determines person names from JSON data."""
    if "name" in json_data:
        names = json_data["name"][:num_faces]
    elif "names" in json_data:
        names = json_data["names"][:num_faces]
    else:
        keyword_parts = json_data["keyword"].split(" ")
        names = keyword_parts[:min(len(keyword_parts), num_faces)]
    
    # Pad with generic names if insufficient
    while len(names) < num_faces:
        names.append(f"person_{len(names)+1}")
    return names

def _find_best_reference(ori_embeddings, person_name):
    """Finds the best reference file for a person based on cosine similarity."""
    ref_dir = os.path.join(REF_DIR, person_name)
    if not os.path.exists(ref_dir):
        return None
    
    ref_files = [f for f in os.listdir(ref_dir) if f.endswith('.npy')]
    if not ref_files:
        return None
    
    best_ref = None
    best_score = 0
    
    # Try up to 20 random references
    candidates = random.sample(ref_files, min(20, len(ref_files)))
    
    for ref_filename in candidates:
        ref_path = os.path.join(ref_dir, ref_filename)
        try:
            ref_data = np.load(ref_path, allow_pickle=True).item()
            ref_emb = torch.tensor(ref_data.get("embeddings", ref_data.get("embedding"))[0])
            
            # Calculate max similarity against all original embeddings
            scores = [cos(ref_emb, torch.tensor(emb)) for emb in ori_embeddings]
            current_score = max(scores)
            
            if current_score > best_score:
                best_score = current_score
                best_ref = ref_path
            
            if current_score > 0.8:  # Early exit threshold
                break
        except Exception:
            continue
            
    if best_ref is None or best_score < 0.6:
        return None
    return best_ref

def _load_external_references(person_names, ori_embeddings):
    """Loads external reference images and embeddings."""
    num_faces = len(person_names)
    ref_images = []
    ref_arcfaces = torch.zeros((num_faces, 512), dtype=torch.bfloat16)
    cluster_embeddings = torch.zeros((num_faces, 512), dtype=torch.bfloat16)
    
    for idx, name in enumerate(person_names):
        ref_file = _find_best_reference(ori_embeddings, name)
        if ref_file is None:
            return None
        
        # Load reference embedding
        ref_data = np.load(ref_file, allow_pickle=True).item()
        ref_emb_vec = ref_data.get("embeddings", ref_data.get("embedding"))[0]
        
        # Load cluster embedding
        cluster_file = os.path.join(REF_CLUSTER_DIR, name + ".npy")
        if not os.path.exists(cluster_file):
            print(f"No cluster embedding file found for {name}, skipping")
            return None
        cluster_emb_vec = np.load(cluster_file, allow_pickle=True)
        
        # Load and process reference image
        try:
            ref_img_path = ref_file.replace('.npy', '.jpg')
            ref_json_path = ref_file.replace('.npy', '.json')
            
            ref_img = Image.open(ref_img_path)
            with open(ref_json_path, 'r') as f:
                ref_json = json.load(f)
                
            extracted_faces = extract_moref(ref_img, ref_json)
            if not extracted_faces:
                return None
            
            ref_images.append(extracted_faces[0])
            ref_arcfaces[idx] = torch.tensor(ref_emb_vec, dtype=torch.bfloat16)
            cluster_embeddings[idx] = torch.tensor(cluster_emb_vec, dtype=torch.bfloat16)
            
        except Exception as e:
            print(f"Error loading reference for {name}: {e}")
            return None
            
    return ref_images, ref_arcfaces, cluster_embeddings

def _build_negative_pool(person_names, num_faces):
    """Constructs the negative embedding pool."""
    pool_size = 4033
    negative_pool = torch.empty((num_faces, pool_size, 512), dtype=torch.bfloat16)

    for i, name in enumerate(person_names):
        # Filter names to avoid self-negatives
        suitable_names = [n for n in REF_NAME_LIST if n != name]
        if not suitable_names:
            suitable_names = REF_NAME_LIST 
            
        rand_names = np.random.choice(suitable_names, size=pool_size, replace=True)

        selected_embs = torch.empty((pool_size, 512), dtype=torch.bfloat16)
        unique_names = set(rand_names)
        
        for uniq_name in unique_names:
            mask = (rand_names == uniq_name)
            pool = REF_DICT[uniq_name]
            # Randomly sample from the person's embedding pool
            count = mask.sum()
            idx = torch.randint(0, pool.shape[0], (count,))
            selected_embs[mask] = pool[idx]

        negative_pool[i] = selected_embs
    return negative_pool

def process_item(item, cp_ratio=1.0):
    """
    Processes a single data item for the dataloader.
    Handles decoding, resizing, reference selection, and negative pool construction.
    """

    # Hardcoded for minimal collate function
    target_size = 512
    
    try:
        # 1. Decode Inputs
        item = _decode_item_data(item)
        img = item["jpg"]
        json_data = item["json"]
        npy_data = item["npy"]
        
        # 2. BBox Processing & Filtering
        bboxes = npy_data['bboxes']
        crop = json_data.get('crop', [0, 0, img.width, img.height])
        new_bboxes = [recalculate_bbox(bbox, crop) for bbox in bboxes]

        # Strict face count filtering 
        num_faces = len(new_bboxes)
        if not (1 <= num_faces <= 5):
            return None

        # Hardcoded for minimal collate function
        if num_faces != 2:
            return None
            
        # 3. Image Resizing
        resized_img, rec_bboxes = general_face_preserving_resize(img, new_bboxes, target_size=target_size)
        if resized_img is None:
            return None

        # 4. Prepare Main Image Tensor
        # Normalize to [-1, 1] and CHW format
        img_tensor = torch.from_numpy((np.array(resized_img) / 127.5) - 1).permute(2, 0, 1)

        # 5. Determine Person Names
        person_names = _get_person_names(json_data, num_faces)
        
        # 6. Reference Selection Strategy
        ori_embeddings = npy_data.get("embeddings", npy_data.get("embedding"))
        gt_embeddings = torch.tensor(ori_embeddings)
        
        use_external_ref = random.random() < cp_ratio
        
        if use_external_ref:
            result = _load_external_references(person_names, ori_embeddings)
            if result is None:
                return None
            ref_images, ref_arcfaces, cluster_embeddings = result
        else:
            # Self-Reference Strategy
            ref_images = extract_faces_variable(item["jpg"], json_data, num_faces)
            if not ref_images or len(ref_images) != num_faces:
                return None
            
            ref_arcfaces = gt_embeddings.clone()
            # Dummy cluster embedding (sync with GT)
            cluster_embeddings = gt_embeddings.clone()

        if len(ref_images) != num_faces:
            return None

        # 7. Negative Pool Construction
        negative_pool = _build_negative_pool(person_names, num_faces)

        # 8. Final Assembly
        return {
            "img": img_tensor,
            "ref_imgs": ref_images,
            "txt": json_data["caption_en"],
            "arcface_embedding": ref_arcfaces,
            "num_faces": num_faces,
            "bboxes": rec_bboxes,
            "gt_embeddings": gt_embeddings,
            "cluster_embeddings": cluster_embeddings,
            "negative_pool": negative_pool,
        }

    except Exception as e:
        print(f"Unexpected error while processing item: {e}")
        return None

# --- Collate Functions ---

def collate_fn_withanyone(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    
    MAX_FACES = 2  # Maximum faces supported
    
    result = {
        "img": [],
        "ref_imgs": [],
        "ref_mask": [],
        "txt": [],
        "arcface_embedding": [],
        "num_faces": [],
        "bboxes": [],
        "gt_embeddings": [],
        "cluster_embeddings": [],
        "negative_pool": [],
    }
    
    for item in batch:
        result["img"].append(item["img"])
        result["txt"].append(item["txt"])
        result["num_faces"].append(item["num_faces"])
        result["bboxes"].append(item["bboxes"])
        
        num_faces = item["num_faces"]
        ref_imgs = item["ref_imgs"]
        arcface_embedding = item["arcface_embedding"]
        gt_embeddings = item["gt_embeddings"]

        # Create mask (1 for valid faces, 0 for padding)
        mask = torch.ones(MAX_FACES, dtype=torch.bool)
        mask[num_faces:] = False
        result["ref_mask"].append(mask)
        
        # Pad if needed
        if num_faces < MAX_FACES:
            padding_arcface = torch.zeros(MAX_FACES - num_faces, 512)
            result["arcface_embedding"].append(torch.cat([arcface_embedding, padding_arcface], dim=0))
            result["gt_embeddings"].append(torch.cat([gt_embeddings, padding_arcface], dim=0))
            result["cluster_embeddings"].append(torch.cat([item["cluster_embeddings"], padding_arcface], dim=0))
            result["negative_pool"].append(item["negative_pool"])
            padding = torch.zeros(MAX_FACES - num_faces, 3, 512, 512)
            padded_refs = torch.cat([ref_imgs, padding], dim=0)
            result["ref_imgs"].append(padded_refs)
        else:
            result["arcface_embedding"].append(arcface_embedding[:MAX_FACES])
            result["gt_embeddings"].append(gt_embeddings[:MAX_FACES])
            result["ref_imgs"].append(ref_imgs[:MAX_FACES])
            if item["cluster_embeddings"] is not None:
                result["cluster_embeddings"].append(item["cluster_embeddings"][:MAX_FACES])
            else:
                # Should not happen given process_item logic
                result["cluster_embeddings"].append(None)
            result["negative_pool"].append(item["negative_pool"])

    # Stack tensors
    result["img"] = torch.stack(result["img"])
    result["ref_mask"] = torch.stack(result["ref_mask"])
    result["arcface_embedding"] = torch.stack(result["arcface_embedding"])
    result["gt_embeddings"] = torch.stack(result["gt_embeddings"])
    result["cluster_embeddings"] = torch.stack(result["cluster_embeddings"])
    result["negative_pool"] = torch.stack(result["negative_pool"])
    
    return result

def efficient_loader(tar_dir_list, batch_size=1, is_distributed=False, rank=0, world_size=1, cp_ratio=1.0, shardshuffle=1000):
    tar_list = []
    for tar_dir in tar_dir_list:
        tar_files = [os.path.join(tar_dir, f) for f in os.listdir(tar_dir) if f.endswith('.tar')]
        tar_list.extend(tar_files)
        print(f"Found {len(tar_files)} tar files in {tar_dir}, cumulative total: {len(tar_list)}")
    
    random.shuffle(tar_list)
    
    def process_item_with_cp_option(item):
        return process_item(item, cp_ratio=cp_ratio)
    
    if is_distributed:
        dataset = wds.WebDataset(
            tar_list, 
            nodesplitter=wds.split_by_node, 
            handler=wds.handlers.warn_and_continue,
            empty_check=True,
            shardshuffle=shardshuffle,
            resampled=True,
        ).map(process_item_with_cp_option).select(lambda x: x is not None)
    else:
        dataset = wds.WebDataset(
            tar_list,
            nodesplitter=wds.split_by_worker, 
            handler=wds.handlers.warn_and_continue,
            empty_check=True, 
            shardshuffle=shardshuffle,
        ).map(process_item_with_cp_option).select(lambda x: x is not None)
    
    loader = wds.WebLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_withanyone,
    )
    
    return loader

if __name__ == "__main__":
    # Example usage
    dataloader = efficient_loader(
        ["/data/"], # wherever you put your tars
        4,
        is_distributed=False,
        rank=0, 
        world_size=1,
        cp_ratio=0.5
    )
    print("Dataloader created")

    record_time = time.time()
    for i, data in enumerate(dataloader):
        print("data keys:", data.keys())
        print("data img shape:", data["img"].shape)
        print("data num_faces:", data["num_faces"])
        print("len of bboxes[0]:", len(data["bboxes"][0]))
        print("data ref_mask:", data["ref_mask"])
        print("shape of negative_pool:", data["negative_pool"].shape)
        
        print("--" * 20)
        print(f"Processed {i+1} batches in {time.time() - record_time:.2f} seconds" )
        record_time = time.time()
        # break