"""
Convert Hugging Face dataset back to benchmark format.
"""

import os
import json
import argparse
from tqdm import tqdm
import re
from datasets import load_from_disk, load_dataset
from PIL import Image
import io

def save_pil_image(img_bytes, output_path):
    """Save binary image data to a file."""
    if img_bytes is None:
        return False
    if isinstance(img_bytes, Image.Image):
        # just save directly and return
        # if RGBA, convert to RGB
        if img_bytes.mode == 'RGBA':
            img_bytes = img_bytes.convert('RGB')
        img_bytes.save(output_path)
        return True
    try:
        with open(output_path, 'wb') as f:
            f.write(img_bytes)
        return True
    except Exception as e:
        print(f"Error saving image to {output_path}: {str(e)}")
        return False

def ensure_dir(directory):
    """Ensure that a directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_hf_dataset(dataset_path_or_name, from_hub=False):
    """Load dataset from local path or Hugging Face Hub."""
    try:
        if from_hub:
            dataset = load_dataset(dataset_path_or_name)
            # If the dataset has multiple splits, use the 'train' split by default
            if isinstance(dataset, dict) and 'train' in dataset:
                return dataset['train']
            return dataset
        else:
            return load_from_disk(dataset_path_or_name)
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None

def hf_to_benchmark(dataset_path, output_dir, from_hub=False):
    """Convert Hugging Face dataset back to benchmark format."""
    # Load the dataset
    dataset = load_hf_dataset(dataset_path, from_hub)
    if dataset is None:
        return False
    
    print(f"Loaded dataset with {len(dataset)} examples")
    print(f"Keys in dataset: {dataset.column_names}")
    
    # Create output directories
    v202_dir = os.path.join(output_dir, 'p2', 'untar')
    v200_single_dir = os.path.join(output_dir, 'p1', 'untar')
    v200_m_num3_dir = os.path.join(output_dir, 'p3', 'num_3')
    v200_m_num4_dir = os.path.join(output_dir, 'p3', 'num_4')
    v200_m_refs_dir = os.path.join(output_dir, 'p3', 'refs')
    
    ensure_dir(v202_dir)
    ensure_dir(v200_single_dir)
    ensure_dir(v200_m_num3_dir)
    ensure_dir(v200_m_num4_dir)
    ensure_dir(v200_m_refs_dir)
    
    # Indexes for each subset
    v202_index = []
    v200_single_index = []
    v200_m_index = []
    
    # Process each example in the dataset
    for idx, example in enumerate(tqdm(dataset, desc="Converting to benchmark format")):
        if 1:
            # Generate a unique ID for this entry
            entry_id = f"{idx+1:04d}"
            
            # Extract data from the example
            subset = example['subset']
            
            # Use 'prompt' field instead of 'prompt'
            prompt = example['prompt'] 
            
            # Handle bboxes - might be string JSON or actual list
            bboxes = example['bboxes']
            if isinstance(bboxes, str):
                bboxes = json.loads(bboxes)
            
            num_persons = example.get('num_persons', 0)
            
            # Process based on subset
            if subset == 'p2':
                # Save GT image
                gt_path = os.path.join(v202_dir, f"{entry_id}.jpg")
                save_pil_image(example['GT'], gt_path)
                
                # Process input_images for reference images
                if 'input_images' in example and len(example['input_images']) >= 2:
                    save_pil_image(example['input_images'][0], os.path.join(v202_dir, f"{entry_id}_1.jpg"))
                    save_pil_image(example['input_images'][1], os.path.join(v202_dir, f"{entry_id}_2.jpg"))
                
                # Save JSON
                json_data = {
                    'caption_en': prompt,
                    'bboxes': bboxes
                }
                with open(os.path.join(v202_dir, f"{entry_id}.json"), 'w') as f:
                    json.dump(json_data, f, indent=2)
                
                # Add to v202 index
                v202_index.append({
                    'prompt': prompt,
                    'image_paths': [f"{entry_id}_1.jpg", f"{entry_id}_2.jpg"],
                    'ori_img_path': f"{entry_id}.jpg"
                })
                
            elif subset == 'p1':
                # Save GT image
                gt_path = os.path.join(v200_single_dir, f"{entry_id}.jpg")
                save_pil_image(example['GT'], gt_path)
                
                # Save reference image (from input_images)
                if 'input_images' in example and len(example['input_images']) >= 1:
                    save_pil_image(example['input_images'][0], os.path.join(v200_single_dir, f"{entry_id}_1.jpg"))
                
                # Save JSON
                json_data = {
                    'caption_en': prompt,
                    'bboxes': bboxes
                }
                with open(os.path.join(v200_single_dir, f"{entry_id}.json"), 'w') as f:
                    json.dump(json_data, f, indent=2)
                
                # Add to v200_single index
                v200_single_index.append({
                    'prompt': prompt,
                    'image_paths': [f"{entry_id}_1.jpg"],
                    'ori_img_path': f"{entry_id}.jpg"
                })
                
            elif subset == 'p3':
                # Determine output directory based on number of persons
                if num_persons == 3:
                    main_output_dir = v200_m_num3_dir
                    num_dir = 'num_3'
                else:
                    main_output_dir = v200_m_num4_dir
                    num_dir = 'num_4'
                
                # Save GT image
                gt_path = os.path.join(main_output_dir, f"{entry_id}.jpg")
                save_pil_image(example['GT'], gt_path)
                
                # Save JSON
                json_data = {
                    'caption_en': prompt,
                    'bboxes': bboxes
                }
                with open(os.path.join(main_output_dir, f"{entry_id}.json"), 'w') as f:
                    json.dump(json_data, f, indent=2)
                
                # Extract person IDs from prompt
                person_matches = re.findall(r'person_(\d+)', prompt)
                ref_paths = []
                names = []
                
                # Process and save reference images from input_images
                if 'input_images' in example:
                    for i, img_bytes in enumerate(example['input_images']):
                        person_id = f"person_{i+1}"
                        ref_dir = os.path.join(v200_m_refs_dir, person_id)
                        ensure_dir(ref_dir)
                        
                        ref_filename = f"{entry_id}_{i+1}.jpg"
                        ref_path = os.path.join(ref_dir, ref_filename)
                        
                        if save_pil_image(img_bytes, ref_path):
                            # Add to reference paths (relative to v200_m)
                            rel_path = f"refs/{person_id}/{ref_filename}"
                            ref_paths.append(rel_path)
                            names.append(person_id)
                
                # Add to v200_m index
                relative_gt_path = f"{num_dir}/{entry_id}.jpg"
                v200_m_index.append({
                    'prompt': prompt,
                    'image_paths': ref_paths,
                    'ori_img_path': relative_gt_path,
                    'name': names
                })
                
        # except Exception as e:
        #     print(f"Error processing example {idx}: {str(e)}")
        #     continue
    
    # Save all index files
    if v202_index:
        with open(os.path.join(output_dir, 'p2.json'), 'w') as f:
            json.dump(v202_index, f, indent=2)
        print(f"Generated p2.json index with {len(v202_index)} entries")
            
    if v200_single_index:
        with open(os.path.join(output_dir, 'p1.json'), 'w') as f:
            json.dump(v200_single_index, f, indent=2)
        print(f"Generated p1.json index with {len(v200_single_index)} entries")
    
    if v200_m_index:
        with open(os.path.join(output_dir, 'p3.json'), 'w') as f:
            json.dump(v200_m_index, f, indent=2)
        print(f"Generated p3.json index with {len(v200_m_index)} entries")
    
    print(f"Conversion complete. Benchmark data saved to {output_dir}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert Hugging Face dataset back to benchmark format')
    parser.add_argument('--dataset', type=str, required=True, help='Path to local dataset or Hugging Face Hub dataset ID')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory')
    parser.add_argument('--from_hub', action='store_true', help='Load dataset from Hugging Face Hub')
    
    args = parser.parse_args()
    
    hf_to_benchmark(args.dataset, args.output_dir, args.from_hub)

