import os
import torch
import numpy as np
from tqdm import tqdm
import time
import concurrent.futures
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Build reference embeddings dictionary from person directories")
    parser.add_argument(
        "--ref_dir",
        type=str,
        default="/mnt/xuhengyuan/data/2person/v5/ref/untar/",
        help="Directory containing person subdirectories with .npy embedding files"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="/mnt/xuhengyuan/data/2person/v5/ref/ref_dict.pth",
        help="Output path for the reference dictionary .pth file"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for storing embeddings (default: bfloat16)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of worker threads for parallel processing"
    )
    parser.add_argument(
        "--max_files_per_person",
        type=int,
        default=200,
        help="Maximum number of embedding files to load per person (default: 200)"
    )
    return parser.parse_args()

def process_person(name, ref_dir, max_files, dtype):
    """Process embeddings for a single person directory"""
    try:
        person_dir = os.path.join(ref_dir, name)
        if not os.path.isdir(person_dir):
            return None
        
        ref_files = [f for f in os.listdir(person_dir) if f.endswith(".npy")]
        if not ref_files:
            return None
            
        if len(ref_files) > max_files:
            ref_files = np.random.choice(ref_files, max_files, replace=False)
            
        emb_list = []
        for f in ref_files:
            arr = np.load(os.path.join(person_dir, f), allow_pickle=True).item()
            emb = arr.get("embeddings", arr.get("embedding"))[0]
            emb_list.append(emb)
            
        np_arr = np.stack(emb_list)  # (N, 512)
        tensor = torch.from_numpy(np_arr).to(dtype)  
        
        return (name, tensor)
    except Exception as e:
        print(f"Error processing {name}: {str(e)}")
        return None

def main():
    args = parse_args()
    
    # Convert dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }
    dtype = dtype_map[args.dtype]
    
    # Check if ref_dir exists
    if not os.path.exists(args.ref_dir):
        raise ValueError(f"Reference directory does not exist: {args.ref_dir}")
    
    print(f"Building REF_DICT from {args.ref_dir}...")
    print(f"Settings: dtype={args.dtype}, workers={args.num_workers}, max_files={args.max_files_per_person}")
    start_time = time.time()
    
    # Get all person directories
    person_dirs = [name for name in sorted(os.listdir(args.ref_dir)) 
                   if os.path.isdir(os.path.join(args.ref_dir, name))]
    
    print(f"Found {len(person_dirs)} person directories")
    
    ref_dict = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        # Submit all tasks
        future_to_person = {
            executor.submit(process_person, name, args.ref_dir, args.max_files_per_person, dtype): name 
            for name in person_dirs
        }
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_person), 
                          total=len(future_to_person), 
                          desc="Processing embeddings"):
            result = future.result()
            if result:
                name, tensor = result
                ref_dict[name] = tensor
    
    elapsed = time.time() - start_time
    print(f"Processing completed in {elapsed:.2f} seconds")
    print(f"Successfully processed {len(ref_dict)} people")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    print(f"Saving to {args.save_path} ...")
    torch.save(ref_dict, args.save_path)
    print("Done!")

if __name__ == "__main__":
    main()