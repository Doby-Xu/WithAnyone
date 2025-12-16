#!/usr/bin/env python
"""
Merge two safetensors files with renaming:
- Add '_arc' suffix to keys containing 'ipa', 'arcface', or 'siglip' from first file
- Add '_sig' suffix to keys containing 'ipa', 'arcface', or 'siglip' from second file
"""

import argparse
import re
from typing import Dict, List, Tuple

import torch
from safetensors.torch import load_file, save_file


def add_suffix_to_name(key: str, suffix: str) -> str:
    """
    Add suffix to the parameter name based on specific patterns.
    Examples:
    - ipa.0.norm1.weight -> ipa_arc.0.norm1.weight
    - arcface_in.0.weight -> arcface_in_arc.0.weight
    - siglip_in.3.bias -> siglip_in_arc.3.bias
    """
    # Handle ipa pattern
    if key.startswith("ipa."):
        return key.replace("ipa.", f"ipa{suffix}.", 1)
    
    # Handle arcface_in pattern
    if key.startswith("arcface_in."):
        return key.replace("arcface_in.", f"arcface_in{suffix}.", 1)
    
    # Handle siglip_in pattern
    if key.startswith("siglip_in."):
        return key.replace("siglip_in.", f"siglip_in{suffix}.", 1)
    
    # For other keys that contain our target patterns but don't match the above
    for pattern in ["arcface", "siglip"]:
        if pattern in key:
            parts = key.split(".")
            # Try to find the pattern in the parts and add suffix
            for i, part in enumerate(parts):
                if pattern in part:
                    parts[i] = f"{part}{suffix}"
                    return ".".join(parts)
    
    # Default case - just return the original key
    return key


def process(arc_path: str, sig_path: str, out_path: str, dry_run: bool = False) -> None:
    # Load both state dictionaries
    arc_state: Dict[str, torch.Tensor] = load_file(arc_path, device="cpu")
    sig_state: Dict[str, torch.Tensor] = load_file(sig_path, device="cpu")
    
    patterns = ["ipa", "arcface", "siglip"]
    out_state: Dict[str, torch.Tensor] = {}
    
    # Process arc state
    arc_keys_processed = 0
    arc_keys_unchanged = 0
    
    for k, v in arc_state.items():
        if any(pattern in k for pattern in patterns):
            new_key = add_suffix_to_name(k, "_arc")
            out_state[new_key] = v
            arc_keys_processed += 1
        else:
            out_state[k] = v
            arc_keys_unchanged += 1
    
    # Process sig state
    sig_keys_processed = 0
    sig_keys_unchanged = 0
    conflicts = 0
    
    for k, v in sig_state.items():
        if any(pattern in k for pattern in patterns):
            new_key = add_suffix_to_name(k, "_sig")
            
            # Check for conflicts
            if new_key in out_state:
                conflicts += 1
                print(f"Warning: Conflict for key {new_key}")
            else:
                out_state[new_key] = v
                sig_keys_processed += 1
        else:
            # For unchanged keys, only add if they don't already exist
            if k not in out_state:
                out_state[k] = v
                sig_keys_unchanged += 1
            else:
                # No need to count as conflict for regular parameters
                pass
    
    meta = {
        "note": "Merged safetensors with _arc and _sig suffixes",
        "arc_source": arc_path,
        "sig_source": sig_path,
        "arc_keys_processed": str(arc_keys_processed),
        "arc_keys_unchanged": str(arc_keys_unchanged),
        "sig_keys_processed": str(sig_keys_processed),
        "sig_keys_unchanged": str(sig_keys_unchanged),
        "conflicts": str(conflicts),
        "total_params": str(len(out_state)),
    }
    
    if dry_run:
        print("Dry run summary:", meta)
        # Print some example keys to verify renaming
        for pattern in patterns:
            arc_examples = [k for k in out_state.keys() if f"{pattern}_arc" in k][:3]
            sig_examples = [k for k in out_state.keys() if f"{pattern}_sig" in k][:3]
            if arc_examples:
                print(f"Example {pattern}_arc keys: {arc_examples}")
            if sig_examples:
                print(f"Example {pattern}_sig keys: {sig_examples}")
        return
    
    save_file(out_state, out_path, metadata=meta)
    print("Done.")
    print(meta)


def main():
    parser = argparse.ArgumentParser(description="Merge two safetensors files with renaming")
    parser.add_argument("--arc", required=True, help="Input .safetensors file for _arc suffix")
    parser.add_argument("--sig", required=True, help="Input .safetensors file for _sig suffix")
    parser.add_argument("--out", required=True, help="Output .safetensors file")
    parser.add_argument("--dry-run", action="store_true", help="Only print summary without writing")
    args = parser.parse_args()
    
    process(args.arc, args.sig, args.out, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
