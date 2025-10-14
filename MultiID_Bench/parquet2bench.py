import os
import json
import base64
import pandas as pd
import argparse
from tqdm import tqdm
import re

def base64_to_image(base64_string, output_path):
    """Convert a base64 string to an image file."""
    if base64_string is None:
        return False
    
    try:
        image_data = base64.b64decode(base64_string)
        with open(output_path, 'wb') as f:
            f.write(image_data)
        return True
    except Exception as e:
        print(f"Error decoding base64 to {output_path}: {str(e)}")
        return False

def ensure_dir(directory):
    """Ensure that a directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def parquet_to_benchmark(parquet_path, output_dir):
    """Convert parquet back to benchmark format."""
    # Read parquet file
    df = pd.read_parquet(parquet_path)
    
    # Create output directories
    v202_dir = os.path.join(output_dir, 'v202', 'untar')
    v200_single_dir = os.path.join(output_dir, 'v200_single', 'untar')
    v200_m_num3_dir = os.path.join(output_dir, 'v200_m', 'num_3')
    v200_m_num4_dir = os.path.join(output_dir, 'v200_m', 'num_4')
    v200_m_refs_dir = os.path.join(output_dir, 'v200_m', 'refs')
    
    ensure_dir(v202_dir)
    ensure_dir(v200_single_dir)
    ensure_dir(v200_m_num3_dir)
    ensure_dir(v200_m_num4_dir)
    ensure_dir(v200_m_refs_dir)
    
    # Indexes for each subset
    v202_index = []
    v200_single_index = []
    v200_m_index = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Converting parquet to benchmark"):
        try:
            # Generate a unique ID for this entry
            entry_id = f"{idx+1:04d}"
            
            # Parse prompt and bboxes
            prompt = row['prompt']
            bboxes = json.loads(row['bboxes']) if isinstance(row['bboxes'], str) else row['bboxes']
            
            # Count number of references (non-None)
            ref_count = sum(1 for i in range(1, 5) if row.get(f'ref_{i}') is not None)
            
            # Decide which subset based on reference count
            if ref_count == 2:  # v202 (two-person)
                output_subdir = v202_dir
                
                # Save GT image
                gt_path = os.path.join(output_subdir, f"{entry_id}.jpg")
                base64_to_image(row['GT'], gt_path)
                
                # Save reference images
                ref1_path = os.path.join(output_subdir, f"{entry_id}_1.jpg")
                ref2_path = os.path.join(output_subdir, f"{entry_id}_2.jpg")
                base64_to_image(row['ref_1'], ref1_path)
                base64_to_image(row['ref_2'], ref2_path)
                
                # Save JSON
                json_data = {
                    'caption_en': prompt,
                    'bboxes': bboxes
                }
                with open(os.path.join(output_subdir, f"{entry_id}.json"), 'w') as f:
                    json.dump(json_data, f, indent=2)
                
                # Add to v202 index using the desired format
                v202_index.append({
                    'prompt': prompt,
                    'image_paths': [f"{entry_id}_1.jpg", f"{entry_id}_2.jpg"],
                    'ori_img_path': f"{entry_id}.jpg"
                })
                    
            elif ref_count == 1:  # v200_single (single-person)
                output_subdir = v200_single_dir
                
                # Save GT image
                gt_path = os.path.join(output_subdir, f"{entry_id}.jpg")
                base64_to_image(row['GT'], gt_path)
                
                # Save reference image
                ref_path = os.path.join(output_subdir, f"{entry_id}_1.jpg")
                base64_to_image(row['ref_1'], ref_path)
                
                # Save JSON
                json_data = {
                    'caption_en': prompt,
                    'bboxes': bboxes
                }
                with open(os.path.join(output_subdir, f"{entry_id}.json"), 'w') as f:
                    json.dump(json_data, f, indent=2)
                
                # Add to v200_single index using the desired format
                v200_single_index.append({
                    'prompt': prompt,
                    'image_paths': [f"{entry_id}_1.jpg"],
                    'ori_img_path': f"{entry_id}.jpg"
                })
                
            else:  # v200_m (multi-person)
                # Extract person IDs from prompt
                person_matches = re.findall(r'person_(\d+)', prompt)
                person_ids = [int(pid) for pid in person_matches]
                
                # Determine number of persons
                num_persons = len(set(person_ids)) if person_ids else 3
                
                if num_persons == 3:
                    main_output_dir = v200_m_num3_dir
                    num_dir = 'num_3'
                else:
                    main_output_dir = v200_m_num4_dir
                    num_dir = 'num_4'
                
                # Save GT image
                gt_path = os.path.join(main_output_dir, f"{entry_id}.jpg")
                base64_to_image(row['GT'], gt_path)
                
                # Save JSON
                json_data = {
                    'caption_en': prompt,
                    'bboxes': bboxes
                }
                with open(os.path.join(main_output_dir, f"{entry_id}.json"), 'w') as f:
                    json.dump(json_data, f, indent=2)
                
                # Save reference images to refs directory and collect paths for index
                ref_paths = []
                names = []
                
                for i in range(1, 5):
                    ref_key = f'ref_{i}'
                    if ref_key in row and row[ref_key] is not None:
                        person_id = f"person_{i}"
                        ref_dir = os.path.join(v200_m_refs_dir, person_id)
                        ensure_dir(ref_dir)
                        
                        ref_filename = f"{entry_id}_{i}.jpg"
                        ref_path = os.path.join(ref_dir, ref_filename)
                        
                        if base64_to_image(row[ref_key], ref_path):
                            # Add to reference paths (relative to v200_m)
                            rel_path = f"refs/{person_id}/{ref_filename}"
                            ref_paths.append(rel_path)
                            names.append(person_id)
                
                # Create entry for v200_m index
                relative_gt_path = f"{num_dir}/{entry_id}.jpg"
                v200_m_index.append({
                    'prompt': prompt,
                    'image_paths': ref_paths,
                    'ori_img_path': relative_gt_path,
                    'name': names
                })
                
        except Exception as e:
            print(f"Error processing row {idx}: {str(e)}")
            continue
    
    # Save all index files
    if v202_index:
        with open(os.path.join(output_dir, 'v202.json'), 'w') as f:
            json.dump(v202_index, f, indent=2)
        print(f"Generated v202.json index with {len(v202_index)} entries")
            
    if v200_single_index:
        with open(os.path.join(output_dir, 'v200_single.json'), 'w') as f:
            json.dump(v200_single_index, f, indent=2)
        print(f"Generated v200_single.json index with {len(v200_single_index)} entries")
    
    if v200_m_index:
        with open(os.path.join(output_dir, 'v200_m.json'), 'w') as f:
            json.dump(v200_m_index, f, indent=2)
        print(f"Generated v200_m.json index with {len(v200_m_index)} entries")
    
    print(f"Conversion complete. Benchmark data saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert parquet back to benchmark format')
    parser.add_argument('--parquet', type=str, required=True, help='Path to parquet file')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory')
    
    args = parser.parse_args()
    
    parquet_to_benchmark(args.parquet, args.output_dir)

