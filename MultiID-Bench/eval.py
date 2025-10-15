# Copyright (c) 2025 Fudan University and StepFun Co., Ltd. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
CIVIL Benchmark Evaluation Tool
-------------------------------
Evaluates generated images against reference images using multiple metrics:
- ArcFace similarity for facial recognition
- CLIP image and text similarity
- Aesthetic scoring
- FID (Fréchet Inception Distance)
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from tqdm import tqdm
from colorama import init, Fore, Style

# Import metrics
from metrics.arcface_metric import ArcFace_Metrics
from metrics.clip_metric import CLIP_Metric
from metrics.aesthetic_metric import Aesthetic_Metric
# from metrics.anatomy_metric import Anatomy_Metric
from metrics.fid_metric import FID_Metric

from PIL import Image

# Initialize colorama for cross-platform colored terminal output
init()


def pad_image(img: Image.Image, ratio: float = 1.5) -> Image.Image:
    """
    Pad an image to a square with black background.
    
    Args:
        img: PIL Image to be padded
        ratio: Padding ratio relative to the longer side
        
    Returns:
        PIL Image: Padded square image
    """
    w, h = img.size
    
    # Find the longer side
    longer_side = max(h, w)
    
    # Calculate new size (ratio times the longer side)
    new_size = int(longer_side * ratio)
    
    # Create a new black square image
    padded_img = Image.new('RGB', (new_size, new_size), (0, 0, 0))
    
    # Calculate offsets to center the original image
    x_offset = (new_size - w) // 2
    y_offset = (new_size - h) // 2
    
    # Paste the original image onto the new padded image
    padded_img.paste(img, (x_offset, y_offset))
    
    return padded_img

class BenchEval_Geo:
    """
    Benchmark evaluation tool for celebrity portrait datasets.
    """
    
    def __init__(
        self,
        target_dir: str,
        output_dir: str = "./output/celeb_evaluation",
        ori_file_name: str = "ori.jpg",
        output_file_name: str = "output_0.jpg",
        ref_1_file_name: str = "ref_1.jpg",
        ref_2_file_name: str = "ref_2.jpg",
        ref_3_file_name: str = None,
        ref_4_file_name: str = None,
        meta_file_name: str = "meta.json",
        caption_keyword: str = "caption_en",
        names_keyword: str = "name",
        padding: bool = True,
        save_individual_results: bool = True,
    ):
        """
        Initialize the benchmark evaluation.
        
        Args:
            target_dir: Directory containing image folders to evaluate
            output_dir: Directory to save individual evaluation results
            ori_file_name: Filename for original images, set to None to skip original image comparison
            output_file_name: Filename for generated output images
            ref_1_file_name: Filename for first reference image
            ref_2_file_name: Filename for second reference image
            ref_3_file_name: Filename for third reference image, set to None to skip
            ref_4_file_name: Filename for fourth reference image, set to None to skip
            meta_file_name: Filename for metadata JSON
            caption_keyword: Key in metadata to extract caption text
            names_keyword: Key in metadata to extract celebrity names, set to None to skip name-based tests
            padding: Whether to pad images to squares
            save_individual_results: Whether to save per-image results
        """
        # Initialize metrics
        self._init_metrics()
        
        # Set file paths and config
        self._init_paths(target_dir, output_dir)
        self._init_config(
            ori_file_name, output_file_name, 
            ref_1_file_name, ref_2_file_name, ref_3_file_name, ref_4_file_name,
            meta_file_name, caption_keyword, names_keyword, padding, save_individual_results
        )
        
        # Initialize result storage and collectors
        self._init_results()
    
    def _init_metrics(self):
        """Initialize all metric calculation components"""
        self.metrics = ArcFace_Metrics()
        self.clip_metric = CLIP_Metric()
        self.aesthetic_metrics = Aesthetic_Metric()
        # self.anatomy_metrics = Anatomy_Metric()
        self.fid_metrics = FID_Metric()
    
    def _init_paths(self, target_dir, output_dir):
        """Initialize paths and create output directory if needed"""
        self.target_dir = target_dir
        self.output_dir = output_dir
        self.list_dir = [d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))]
    
    def _init_config(self, 
                   ori_file_name, output_file_name,
                   ref_1_file_name, ref_2_file_name, ref_3_file_name, ref_4_file_name,
                   meta_file_name, caption_keyword, names_keyword, padding, save_individual_results):
        """Initialize configuration settings"""
        # Create output directory if needed
        if save_individual_results and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Original and output image config
        self.ori_file_name = ori_file_name
        self.ori_flag = self._is_valid_filename(ori_file_name)
        self.output_file_name = output_file_name
        
        # Reference image config
        self.ref_1_file_name = ref_1_file_name
        self.ref_2_file_name = ref_2_file_name
        self.ref_3_file_name = ref_3_file_name
        self.ref_4_file_name = ref_4_file_name
        
        # Reference image flags
        self.ref_1_flag = self._is_valid_filename(ref_1_file_name)
        self.ref_2_flag = self._is_valid_filename(ref_2_file_name)
        self.ref_3_flag = self._is_valid_filename(ref_3_file_name)
        self.ref_4_flag = self._is_valid_filename(ref_4_file_name)
        
        self.meta_file_name = meta_file_name
        
        # Set options
        self.caption_keyword = caption_keyword
        self.names_keyword = names_keyword
        self.name_flag = self._is_valid_filename(names_keyword)
        self.padding = padding
        self.save_individual_results = save_individual_results
    
    def _is_valid_filename(self, filename):
        """Check if a filename is valid (not None, empty, or 'None')"""
        return filename is not None and filename != "" and filename != "None"
    
    def _init_results(self):
        """Initialize the results structure and metric collectors"""
        self.results = {
            "samples": [],
            "summary": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Initialize metric collectors based on flags
        self.collectors = MetricCollectors(
            use_original=self.ori_flag,
            use_names=self.name_flag
        )
    
    def _load_image(self, path: str, warn_if_fail=True) -> Optional[Image.Image]:
        """Safe image loading with error handling"""
        try:
            return Image.open(path)
        except Exception as e:
            if warn_if_fail:
                print(f"{Fore.RED}Error loading image {path}: {e}{Style.RESET_ALL}")
            return None
            
    def _load_metadata(self, path: str) -> Optional[Dict]:
        """Safe metadata loading with error handling"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"{Fore.RED}Error loading metadata {path}: {e}{Style.RESET_ALL}")
            return None
    
    def _save_individual_result(self, output_img: Image.Image, meta_id: str, metrics_data: Dict):
        """Save individual image result and metadata"""
        try:
            # Save image
            output_img.save(os.path.join(self.output_dir, f"{meta_id}.jpg"))
            
            # Save metrics as JSON
            with open(os.path.join(self.output_dir, f"{meta_id}.json"), "w") as f:
                json.dump(metrics_data, f, indent=4)
        except Exception as e:
            print(f"{Fore.RED}Error saving individual result for {meta_id}: {e}{Style.RESET_ALL}")
    
    def _load_reference_images(self, dir_path):
        """Load available reference images based on configuration flags"""
        refs = []
        ref_count = 0
        
        # Process each possible reference image
        ref_configs = [
            (self.ref_1_flag, self.ref_1_file_name, True),      # Required ref
            (self.ref_2_flag, self.ref_2_file_name, False),     # Optional refs
            (self.ref_3_flag, self.ref_3_file_name, False),
            (self.ref_4_flag, self.ref_4_file_name, False)
        ]
        
        for flag, filename, is_required in ref_configs:
            if not flag:
                continue
                
            ref_img = self._load_image(
                os.path.join(dir_path, filename),
                warn_if_fail=is_required
            )
            
            if ref_img:
                if self.padding:
                    ref_img = pad_image(ref_img)
                refs.append(ref_img)
                ref_count += 1
        
        return refs, ref_count
    
    def _calculate_metrics_for_sample(self, output_img, ori_img, refs, caption, names, dir_idx, directory):
        """Calculate all metrics for a sample and return the results"""
        face_models = ["arcface", "facenet", "adaface"]
        face_metrics = {}
        
        # Calculate metrics for each face model
        for model in face_models:
            face_metrics[model] = self._calculate_face_metrics(
                model, output_img, ori_img, refs, names
            )
        
        # Calculate other metrics
        clip_metrics = self._calculate_clip_metrics(output_img, ori_img, caption)
        aesthetic_score = self.aesthetic_metrics(output_img)
        # anatomical_score = self.anatomy_metrics(output_img, return_scores=True)[0]
        anatomical_score = 0.0  # Placeholder since anatomy metric is commented out
        
        # Validate metric calculation success
        if all(face_metrics[model]["sim_ref"] is None for model in face_models):
            print(f"{Fore.RED}Face similarity calculation failed for {directory}{Style.RESET_ALL}")
            return None
            
        # Log sample metrics periodically
        self._log_sample_metrics(
            dir_idx, directory, face_metrics, clip_metrics,
            aesthetic_score, anatomical_score, len(refs)
        )
        
        return {
            "face_metrics": face_metrics,
            "clip_i": clip_metrics["clip_i"],
            "clip_t": clip_metrics["clip_t"],
            "aesthetic_score": aesthetic_score,
            "anatomical_score": anatomical_score
        }
    
    def _calculate_face_metrics(self, model, output_img, ori_img, refs, names):
        """Calculate face metrics for a specific model"""
        metrics = {}
        
        # Original image similarity if available
        if self.ori_flag:
            _, sim_ori, non_matched_mean_ori = self.metrics.compare_faces_with_confusion_matrix(
                output_img, ori_img, model=model
            )
            metrics["sim_ori"] = sim_ori
            
            # Only test unmatched faces when ref_2 is available
            if self.ref_2_flag:
                metrics["non_matched_ori"] = non_matched_mean_ori
            else:
                metrics["non_matched_ori"] = 0.0
                
            # Calculate copy_geo metric when original image is available
            try:
                copy_geo_scores = self.metrics.calculate_copy_geo_multi_person(
                    output_img, ori_img, refs, model=model
                )
                metrics["copy_geo"] = copy_geo_scores
            except Exception as e:
                print(f"{Fore.YELLOW}Warning: Failed to calculate Copy-Geo using {model}: {str(e)}{Style.RESET_ALL}")
                metrics["copy_geo"] = None
        else:
            metrics["sim_ori"] = None
            metrics["non_matched_ori"] = None
            metrics["copy_geo"] = None
        
        # Reference similarity metrics
        _, sim_ref, non_matched_mean_ref = self.metrics.compare_faces_with_confusion_matrix_with_ref(
            output_img, refs, model=model
        )
        metrics["sim_ref"] = sim_ref
        
        if self.ref_2_flag:
            metrics["non_matched_ref"] = non_matched_mean_ref
        else:
            metrics["non_matched_ref"] = 0.0
        
        # Cluster similarity metrics if names are available
        if self.name_flag:
            # _, sim_cluster, non_matched_mean_cluster = self.metrics.compare_faces_with_confusion_matrix_with_ref_names(
            #     output_img, names, model=model
            # )
            sim_cluster = 0.0  # Placeholder since name-based metric is commented out
            metrics["sim_cluster"] = sim_cluster
        else:
            metrics["sim_cluster"] = None
        
        return metrics
    
    def _calculate_clip_metrics(self, output_img, ori_img, caption):
        """Calculate CLIP similarity metrics"""
        clip_i = None
        if self.ori_flag:
            clip_i = self.clip_metric.compute_clip_i(ori_img, output_img)
        
        clip_t = self.clip_metric.compute_clip_t(output_img, caption)
        
        return {
            "clip_i": clip_i,
            "clip_t": clip_t
        }
    
    def _log_sample_metrics(self, dir_idx, directory, face_metrics, clip_metrics, 
                           aesthetic_score, anatomical_score, ref_count):
        """Log metrics for a sample at periodic intervals"""
        # Only log occasionally to avoid console spam
        if dir_idx % max(1, len(self.list_dir) // 10) != 0:
            return
        
        print(f"\n{Fore.CYAN}Sample: {directory}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Using {ref_count} reference images for this sample{Style.RESET_ALL}")
        
        for model in face_metrics:
            model_metrics = face_metrics[model]
            
            # Print face similarity with original if available
            if self.ori_flag and model_metrics["sim_ori"] is not None:
                print(f"  {model.upper()} Face similarity (original): "
                      f"{model_metrics['sim_ori'].mean():.4f}")
            
            # Print face similarity with reference
            if model_metrics["sim_ref"] is not None:
                print(f"  {model.upper()} Face similarity (reference): "
                      f"{model_metrics['sim_ref'].mean():.4f}")
            
            # Print cluster similarity if available
            if self.name_flag and model_metrics["sim_cluster"] is not None:
                print(f"  {model.upper()} Cluster similarity: "
                      f"{model_metrics['sim_cluster'].mean():.4f}")
        
        # Print CLIP and other scores
        clip_i, clip_t = clip_metrics.get("clip_i"), clip_metrics.get("clip_t")
        if self.ori_flag:
            print(f"  CLIP image: {clip_i:.4f}", end="")
        print(f", CLIP text: {clip_t:.4f}")
        print(f"  Aesthetic score: {aesthetic_score:.4f}")
        print(f"  Anatomy score: {float(anatomical_score):.4f}\n")
    
    def _collect_sample_metrics(self, all_metrics, metadata, ref_count, output_img):
        """Process metrics from a sample for collection and result storage"""
        face_metrics = all_metrics["face_metrics"]
        
        # Collect metrics for each model
        for model in face_metrics:
            model_metrics = face_metrics[model]
            self.collectors.add_model_metrics(model, model_metrics, self.ori_flag, self.name_flag)
        
        # Collect model-independent metrics
        self.collectors.add_general_metrics(
            clip_i=all_metrics["clip_i"] if self.ori_flag else None,
            clip_t=all_metrics["clip_t"],
            aesthetic_score=all_metrics["aesthetic_score"],
            anatomy_score=all_metrics["anatomical_score"]
        )
        
        # Create sample data for results
        sample_data = self._create_sample_data(
            metadata, all_metrics, ref_count
        )
        
        # Save individual result if enabled
        if self.save_individual_results:
            self._save_individual_result(
                output_img,
                metadata.get("id", os.path.basename(os.path.normpath(metadata.get("path", "")))),
                sample_data
            )
        
        # Add to results
        self.results["samples"].append(sample_data)
        
        return sample_data
    
    def _create_sample_data(self, metadata, all_metrics, ref_count):
        """Create a dictionary of sample data for results"""
        face_metrics = all_metrics["face_metrics"]
        
        # Basic sample data
        sample_id = metadata.get("id", os.path.basename(os.path.normpath(metadata.get("path", ""))))
        sample_data = {
            "id": sample_id,
            "anatomy_score": float(all_metrics["anatomical_score"]),
            "caption": metadata.get(self.caption_keyword, ""),
            "clip_t": float(all_metrics["clip_t"]),
            "aes_score": float(all_metrics["aesthetic_score"]),
            "ref_count": ref_count
        }
        
        # Add conditional metrics
        if self.ori_flag:
            sample_data["clip_i"] = float(all_metrics["clip_i"])
        
        # Add model-specific metrics
        for model in face_metrics:
            model_metrics = face_metrics[model]
            
            # Reference similarity
            if model_metrics["sim_ref"] is not None:
                sample_data[f"{model}_sim_ref"] = float(model_metrics["sim_ref"].mean())
            
            # Original similarity if available
            if self.ori_flag and model_metrics["sim_ori"] is not None:
                sample_data[f"{model}_sim_ori"] = float(model_metrics["sim_ori"].mean())
            
            # Cluster similarity if available
            if self.name_flag and model_metrics["sim_cluster"] is not None:
                sample_data[f"{model}_sim_cluster"] = float(model_metrics["sim_cluster"].mean())
            
            # Copy-geo metrics if available
            if self.ori_flag and model_metrics["copy_geo"] is not None:
                sample_data[f"{model}_copy_geo"] = float(model_metrics["copy_geo"].mean())
        
        return sample_data
    
    def _calculate_fid(self, original_images, generated_images):
        """Calculate FID score between original and generated images"""
        if not self.ori_flag:
            return None
            
        print(f"\n{Fore.CYAN}Calculating FID score...{Style.RESET_ALL}")
        try:
            if len(original_images) > 1 and len(generated_images) > 1:
                fid_score = self.fid_metrics.calculate_fid(original_images, generated_images)
                print(f"{Fore.GREEN}FID Score: {fid_score:.4f}{Style.RESET_ALL}")
                return float(fid_score)
            else:
                print(f"{Fore.YELLOW}Not enough samples to calculate FID score{Style.RESET_ALL}")
                return None
        except Exception as e:
            print(f"{Fore.RED}Error calculating FID: {str(e)}{Style.RESET_ALL}")
            return None
    
    def _compute_statistics(self, values, remove_outliers=True, method="iqr", threshold=1.5, z_threshold=3.0):
        """Calculate comprehensive statistics for a list of values with outlier removal"""
        if not values:
            return None
            
        values_array = np.array(values)
        original_count = len(values_array)
        
        # Remove outliers if requested
        if remove_outliers:
            filtered_values, outliers_removed = self._remove_outliers(
                values_array, method, threshold, z_threshold
            )
            
            # Use filtered values for statistics
            if len(filtered_values) < original_count * 0.5:
                # If we've removed more than 50% of values, something is wrong
                # Fall back to using median-based statistics which are robust to outliers
                return self._compute_robust_statistics(values_array)
            
            # Calculate statistics on filtered values
            outliers_percent = (outliers_removed / original_count) * 100 if original_count > 0 else 0.0
            
            return {
                "mean": float(np.mean(filtered_values)),
                "std": float(np.std(filtered_values)),
                "min": float(np.min(filtered_values)),
                "max": float(np.max(filtered_values)),
                "median": float(np.median(filtered_values)),
                "q1": float(np.percentile(filtered_values, 25)),
                "q3": float(np.percentile(filtered_values, 75)),
                "outliers_removed": int(outliers_removed),
                "outliers_percent": float(outliers_percent)
            }
        
        # Return simple statistics if not removing outliers
        return {
            "mean": float(np.mean(values_array)),
            "std": float(np.std(values_array)),
            "min": float(np.min(values_array)),
            "max": float(np.max(values_array)),
            "median": float(np.median(values_array)),
            "q1": float(np.percentile(values_array, 25)),
            "q3": float(np.percentile(values_array, 75)),
            "outliers_removed": 0,
            "outliers_percent": 0.0
        }
    
    def _remove_outliers(self, values_array, method, threshold, z_threshold):
        """Remove outliers using the specified method and return filtered values"""
        if method == "iqr":
            # IQR method - commonly used for detecting outliers
            q1 = np.percentile(values_array, 25)
            q3 = np.percentile(values_array, 75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            mask = (values_array >= lower_bound) & (values_array <= upper_bound)
            filtered_values = values_array[mask]
            
        elif method == "zscore":
            # Z-score method - removes values beyond certain standard deviations
            mean = np.mean(values_array)
            std = np.std(values_array)
            if std == 0:  # Handle constant arrays
                filtered_values = values_array
            else:
                z_scores = np.abs((values_array - mean) / std)
                mask = z_scores < z_threshold
                filtered_values = values_array[mask]
                
        elif method == "percentile":
            # Percentile method - removes top and bottom 5%
            filtered_values = np.percentile(values_array, [5, 95])
            filtered_values = values_array[(values_array >= filtered_values[0]) & 
                                          (values_array <= filtered_values[1])]
        else:
            filtered_values = values_array
            
        outliers_removed = len(values_array) - len(filtered_values)
        return filtered_values, outliers_removed
    
    def _compute_robust_statistics(self, values_array):
        """Compute robust statistics that are not affected by outliers"""
        return {
            "mean": float(np.median(values_array)),  # Use median instead of mean
            "std": float(np.percentile(values_array, 75) - np.percentile(values_array, 25)),  # Use IQR as std
            "min": float(np.percentile(values_array, 5)),  # 5th percentile instead of min
            "max": float(np.percentile(values_array, 95)),  # 95th percentile instead of max
            "median": float(np.median(values_array)),
            "q1": float(np.percentile(values_array, 25)),
            "q3": float(np.percentile(values_array, 75)),
            "outliers_removed": 0,  # We're using percentiles instead of removal
            "outliers_percent": 0.0
        }
    
    def _calculate_summary_statistics(self, successful_count, start_time, fid_score=None):
        """Calculate summary statistics for all collected metrics"""
        if successful_count == 0:
            return {}
            
        # Calculate model-specific statistics
        summary = self._calculate_model_statistics()
        
        # Calculate average across all models
        summary["average"] = self._calculate_average_model_statistics(summary)
        
        # Add general metrics
        summary["general"] = self._calculate_general_statistics(start_time, fid_score)
        
        return summary
    
    def _calculate_model_statistics(self):
        """Calculate statistics for each face recognition model"""
        summary = {}
        
        for model in ["arcface", "facenet", "adaface"]:
            model_data = self.collectors.get_model_data(model)
            if not model_data or not model_data.get("sim_ref_list"):
                continue
                
            model_stats = {
                "sim_ref": self._compute_statistics(model_data["sim_ref_list"]),
                "non_matched_ref": self._compute_statistics(model_data["unmatched_ref_list"])
            }
            
            # Add original image metrics if available
            if self.ori_flag and model_data.get("sim_ori_list"):
                model_stats.update({
                    "sim_ori": self._compute_statistics(model_data["sim_ori_list"]),
                    "non_matched_ori": self._compute_statistics(model_data["unmatched_list"])
                })
                
                # Add copy-geo metrics if available
                if model_data.get("copy_geo_list"):
                    model_stats["copy_geo"] = self._compute_statistics(model_data["copy_geo_list"])
            
            # Add cluster metrics if available
            if self.name_flag and model_data.get("sim_cluster_list"):
                model_stats["sim_cluster"] = self._compute_statistics(model_data["sim_cluster_list"])
                
                # Calculate difference between original and cluster if both available
                if self.ori_flag and model_data.get("sim_ori_list"):
                    diff_ori_cluster = [
                        ori - cluster for ori, cluster in zip(
                            model_data["sim_ori_list"], model_data["sim_cluster_list"]
                        )
                    ]
                    model_stats["diff_ori_cluster"] = self._compute_statistics(diff_ori_cluster)
            
            summary[model] = model_stats
        
        return summary
    
    def _calculate_average_model_statistics(self, model_summary):
        """Calculate average statistics across all models"""
        models_with_data = [model for model in ["arcface", "facenet", "adaface"] if model in model_summary]
        
        if not models_with_data:
            return {}
            
        # Combine data across models
        combined_data = self._combine_model_data(models_with_data)
        
        # Calculate statistics on combined data
        avg_stats = {
            "sim_ref": self._compute_statistics(combined_data["all_sim_ref"]),
            "non_matched_ref": self._compute_statistics(combined_data["all_non_matched_ref"])
        }
        
        # Add conditional statistics
        if combined_data["all_sim_ori"]:
            avg_stats["sim_ori"] = self._compute_statistics(combined_data["all_sim_ori"])
        
        if combined_data["all_non_matched_ori"]:
            avg_stats["non_matched_ori"] = self._compute_statistics(combined_data["all_non_matched_ori"])
        
        if combined_data["all_sim_cluster"]:
            avg_stats["sim_cluster"] = self._compute_statistics(combined_data["all_sim_cluster"])
        
        if combined_data["all_copy_geo"]:
            avg_stats["copy_geo"] = self._compute_statistics(combined_data["all_copy_geo"])
        
        if combined_data["all_diff_ori_cluster"]:
            avg_stats["diff_ori_cluster"] = self._compute_statistics(combined_data["all_diff_ori_cluster"])
        
        return avg_stats
    
    def _combine_model_data(self, models):
        """Combine data from multiple models for averaging"""
        all_sim_ref = []
        all_non_matched_ref = []
        all_sim_ori = []
        all_non_matched_ori = []
        all_sim_cluster = []
        all_copy_geo = []
        all_diff_ori_cluster = []
        
        for model in models:
            model_data = self.collectors.get_model_data(model)
            
            # Combine reference similarity data
            all_sim_ref.extend(model_data["sim_ref_list"])
            all_non_matched_ref.extend(model_data["unmatched_ref_list"])
            
            # Combine original image data if available
            if self.ori_flag and model_data.get("sim_ori_list"):
                all_sim_ori.extend(model_data["sim_ori_list"])
                all_non_matched_ori.extend(model_data["unmatched_list"])
                
                # Combine copy-geo data if available
                if model_data.get("copy_geo_list"):
                    all_copy_geo.extend(model_data["copy_geo_list"])
            
            # Combine cluster data if available
            if self.name_flag and model_data.get("sim_cluster_list"):
                all_sim_cluster.extend(model_data["sim_cluster_list"])
        
        # Calculate differences if both original and cluster data are available
        if self.ori_flag and self.name_flag and all_sim_ori and all_sim_cluster:
            min_len = min(len(all_sim_ori), len(all_sim_cluster))
            all_diff_ori_cluster = [all_sim_ori[i] - all_sim_cluster[i] for i in range(min_len)]
        
        return {
            "all_sim_ref": all_sim_ref,
            "all_non_matched_ref": all_non_matched_ref,
            "all_sim_ori": all_sim_ori,
            "all_non_matched_ori": all_non_matched_ori,
            "all_sim_cluster": all_sim_cluster,
            "all_copy_geo": all_copy_geo,
            "all_diff_ori_cluster": all_diff_ori_cluster
        }
    
    def _calculate_general_statistics(self, start_time, fid_score):
        """Calculate general statistics from collected metrics"""
        general_data = self.collectors.get_general_data()
        
        general_metrics = {
            "clip_t": self._compute_statistics(general_data["sim_clip_t_list"]),
            "aesthetic": self._compute_statistics(general_data["aes_list"]),
            "anatomy": self._compute_statistics(general_data["anatomy_scores"]),
            "total_samples": self.collectors.total_count,
            "failed_samples": self.collectors.bad_count,
            "successful_samples": self.collectors.successful_count,
            "duration_seconds": time.time() - start_time
        }
        
        # Add CLIP image similarity if available
        if self.ori_flag and general_data.get("sim_clip_i_list"):
            general_metrics["clip_i"] = self._compute_statistics(general_data["sim_clip_i_list"])
        
        # Add FID score if available
        if self.ori_flag and fid_score is not None:
            general_metrics["fid"] = fid_score
        
        return general_metrics
    
    def _print_summary_table(self, summary, start_time):
        """Print evaluation summary as a formatted table"""
        reporter = SummaryReporter(
            summary, self.ori_flag, self.name_flag, start_time
        )
        reporter.print_summary()
    
    def __call__(self) -> Dict:
        """
        Run the evaluation process on all directories.
        
        Returns:
            Dict with evaluation results
        """
        print(f"\n{Fore.CYAN}╔══════════════════════════════════════╗")
        print(f"║  CELEB BENCHMARK EVALUATION TOOL   ║")
        print(f"╚══════════════════════════════════════╝{Style.RESET_ALL}\n")
        
        start_time = time.time()
        
        # Collect images for FID
        original_images = [] if self.ori_flag else None
        generated_images = []
        
        # Process all directories with progress bar
        for dir_idx, directory in enumerate(tqdm(self.list_dir, desc="Evaluating samples", unit="sample")):
            self.collectors.total_count += 1
            
            # Process directory path and metadata
            dir_path, metadata = self._process_directory_metadata(directory)
            if not metadata:
                self.collectors.bad_count += 1
                continue
            
            # Extract caption and names from metadata
            caption, names = self._extract_metadata_text(metadata, directory)
            
            # Load images
            ori_img, output_img = self._load_primary_images(dir_path, directory)
            if (self.ori_flag and not ori_img) or not output_img:
                self.collectors.bad_count += 1
                continue
            
            # Load reference images
            refs, ref_count = self._load_reference_images(dir_path)
            if ref_count == 0:
                print(f"{Fore.RED}No reference images found for {directory}, skipping...{Style.RESET_ALL}")
                self.collectors.bad_count += 1
                continue
            
            # Calculate metrics for this sample
            try:
                all_metrics = self._calculate_metrics_for_sample(
                    output_img, ori_img, refs, caption, names, dir_idx, directory
                )
                
                if not all_metrics:
                    self.collectors.bad_count += 1
                    continue
                
                # Collect metrics and add to results
                self._collect_sample_metrics(all_metrics, metadata, ref_count, output_img)
                
                self.collectors.successful_count += 1
                
                # Collect images for FID
                if self.ori_flag:
                    original_images.append(ori_img)
                generated_images.append(output_img)
                
            except Exception as e:
                print(f"{Fore.RED}Error processing {directory}: {e}{Style.RESET_ALL}")
                self.collectors.bad_count += 1
                continue
        
        # Calculate FID score
        fid_score = self._calculate_fid(original_images, generated_images)
        if fid_score is not None:
            self.results["fid"] = fid_score
        
        # Calculate summary statistics if we have successful samples
        if self.collectors.successful_count > 0:
            summary = self._calculate_summary_statistics(
                self.collectors.successful_count, start_time, fid_score
            )
            
            # Store summary in results
            self.results["summary"] = summary
            
            # Print summary table
            self._print_summary_table(summary, start_time)
        
        # Return complete results
        return self.results
    
    def _process_directory_metadata(self, directory):
        """Process directory path and load metadata"""
        dir_path = os.path.join(self.target_dir, directory)
        meta_path = os.path.join(dir_path, self.meta_file_name)
        
        # Check if metadata exists
        if not os.path.exists(meta_path):
            print(f"{Fore.YELLOW}Meta file not found for {directory}, skipping...{Style.RESET_ALL}")
            print(f"{meta_path}")
            return dir_path, None
        
        # Load metadata
        metadata = self._load_metadata(meta_path)
        if metadata:
            # Add directory path to metadata for reference
            metadata["path"] = dir_path
            
        return dir_path, metadata
    
    def _extract_metadata_text(self, metadata, directory):
        """Extract caption and names from metadata"""
        caption = metadata.get(self.caption_keyword, "")
        names = metadata.get(self.names_keyword, []) if self.name_flag else []
        
        # Log warnings for missing data
        if not caption:
            print(f"{Fore.YELLOW}Warning: No caption found for {directory}{Style.RESET_ALL}")
        
        if self.name_flag and not names:
            print(f"{Fore.YELLOW}Warning: No names found for {directory}{Style.RESET_ALL}")
        
        return caption, names
    
    def _load_primary_images(self, dir_path, directory):
        """Load original and output images"""
        ori_img = None
        if self.ori_flag:
            ori_img = self._load_image(os.path.join(dir_path, self.ori_file_name))
            if not ori_img:
                print(f"{Fore.RED}Original image not found for {directory}, skipping...{Style.RESET_ALL}")
                return None, None
        
        output_img = self._load_image(os.path.join(dir_path, self.output_file_name))
        if not output_img:
            print(f"{Fore.RED}Output image not found for {directory}, skipping...{Style.RESET_ALL}")
            return ori_img, None
        
        return ori_img, output_img


class MetricCollectors:
    """Helper class to collect and organize metrics during evaluation"""
    
    def __init__(self, use_original=True, use_names=False):
        # Statistics
        self.total_count = 0
        self.bad_count = 0
        self.successful_count = 0
        
        # Initialize model-specific collectors
        self.model_collectors = {}
        for model in ["arcface", "facenet", "adaface"]:
            self.model_collectors[model] = self._init_model_collectors(use_original, use_names)
        
        # Initialize general metrics collectors
        self.general_collectors = self._init_general_collectors(use_original)
    
    def _init_model_collectors(self, use_original, use_names):
        """Initialize metric collectors for a specific model"""
        collectors = {
            "sim_ref_list": [],
            "unmatched_ref_list": []
        }
        
        # Add collectors for original image metrics if needed
        if use_original:
            collectors.update({
                "sim_ori_list": [],
                "unmatched_list": [],
                "copy_geo_list": []
            })
        
        # Add collectors for name/cluster metrics if needed
        if use_names:
            collectors["sim_cluster_list"] = []
        
        return collectors
    
    def _init_general_collectors(self, use_original):
        """Initialize collectors for general (non-model-specific) metrics"""
        collectors = {
            "sim_clip_t_list": [],
            "aes_list": [],
            "anatomy_scores": []
        }
        
        # Add collectors for original image metrics if needed
        if use_original:
            collectors["sim_clip_i_list"] = []
        
        return collectors
    
    def add_model_metrics(self, model, metrics, use_original, use_names):
        """Add metrics for a specific model to collectors"""
        collectors = self.model_collectors[model]
        
        # Always add reference similarity metrics
        if metrics["sim_ref"] is not None:
            collectors["sim_ref_list"].append(float(metrics["sim_ref"].mean()))
            collectors["unmatched_ref_list"].append(float(metrics["non_matched_ref"]))
        
        # Add original image metrics if available
        if use_original and metrics["sim_ori"] is not None:
            collectors["sim_ori_list"].append(float(metrics["sim_ori"].mean()))
            collectors["unmatched_list"].append(float(metrics["non_matched_ori"]))
            
            # Add copy-geo metrics if available
            if metrics["copy_geo"] is not None:
                collectors["copy_geo_list"].append(float(metrics["copy_geo"].mean()))
        
        # Add cluster metrics if available
        if use_names and metrics["sim_cluster"] is not None:
            collectors["sim_cluster_list"].append(float(metrics["sim_cluster"].mean()))
    
    def add_general_metrics(self, clip_i, clip_t, aesthetic_score, anatomy_score):
        """Add general metrics to collectors"""
        # Always add text metrics
        self.general_collectors["sim_clip_t_list"].append(float(clip_t))
        self.general_collectors["aes_list"].append(float(aesthetic_score))
        self.general_collectors["anatomy_scores"].append(float(anatomy_score))
        
        # Add image metrics if available
        if clip_i is not None:
            self.general_collectors["sim_clip_i_list"].append(float(clip_i))
    
    def get_model_data(self, model):
        """Get collected data for a specific model"""
        return self.model_collectors.get(model, {})
    
    def get_general_data(self):
        """Get collected general metrics data"""
        return self.general_collectors


class SummaryReporter:
    """Helper class to generate and print evaluation summary reports"""
    
    def __init__(self, summary, ori_flag, name_flag, start_time):
        self.summary = summary
        self.ori_flag = ori_flag
        self.name_flag = name_flag
        self.start_time = start_time
        self.duration = time.time() - start_time
    
    def print_summary(self):
        """Print the complete summary report"""
        self._print_header()
        self._print_model_metrics()
        self._print_general_metrics()
        self._print_footer()
    
    def _print_header(self):
        """Print summary header with basic statistics"""
        general = self.summary.get("general", {})
        
        print(f"\n{Fore.CYAN}╔═══════════════════════════════════════════════════════╗")
        print(f"║                EVALUATION SUMMARY RESULTS               ║")
        print(f"╠═══════════════════════════════════════════════════════╣")
        print(f"║ {Fore.WHITE}Total samples:{Fore.CYAN}            {general.get('total_samples', 0):<27} ║")
        print(f"║ {Fore.WHITE}Successful samples:{Fore.CYAN}       {general.get('successful_samples', 0):<27} ║")
        print(f"║ {Fore.WHITE}Failed samples:{Fore.CYAN}           {general.get('failed_samples', 0):<27} ║")
    
    def _print_model_metrics(self):
        """Print metrics for each face recognition model"""
        for model_name in ["arcface", "facenet", "adaface", "average"]:
            if model_name not in self.summary:
                continue
                
            model_stats = self.summary[model_name]
            
            print(f"╠═══════════════════════════════════════════════════════╣")
            print(f"║ {Fore.YELLOW}{model_name.upper()} METRICS:{' ':>37} ║")
            
            # Print similarity metrics with original if available
            if self.ori_flag and "sim_ori" in model_stats:
                print(f"║ {Fore.CYAN}Face similarity with original:{' ':>25} ║")
                self._print_stat_row("sim_ori", model_stats)
            
            # Print similarity metrics with references
            print(f"║ {Fore.CYAN}Face similarity with reference:{' ':>24} ║")
            self._print_stat_row("sim_ref", model_stats)
            
            # Print similarity metrics with cluster if available
            if self.name_flag and "sim_cluster" in model_stats:
                print(f"║ {Fore.CYAN}Cluster similarity:{' ':>34} ║")
                self._print_stat_row("sim_cluster", model_stats)
            
            # Print difference metrics if available
            if self.ori_flag and "copy_geo" in model_stats:
                print(f"║ {Fore.CYAN}Difference (original vs reference):{' ':>18} ║")
                self._print_stat_row("copy_geo", model_stats)
            
            if self.ori_flag and self.name_flag and "diff_ori_cluster" in model_stats:
                print(f"║ {Fore.CYAN}Difference (original vs cluster):{' ':>20} ║")
                self._print_stat_row("diff_ori_cluster", model_stats)
            
            # Print non-matched metrics if available
            if self.ori_flag and "non_matched_ori" in model_stats:
                print(f"║ {Fore.CYAN}Non-matched faces (original):{' ':>24} ║")
                self._print_stat_row("non_matched_ori", model_stats)
            
            print(f"║ {Fore.CYAN}Non-matched faces (reference):{' ':>23} ║")
            self._print_stat_row("non_matched_ref", model_stats)
    
    def _print_general_metrics(self):
        """Print general metrics"""
        if "general" not in self.summary:
            return
            
        general = self.summary["general"]
        
        print(f"╠═══════════════════════════════════════════════════════╣")
        print(f"║ {Fore.YELLOW}GENERAL METRICS:{' ':>39} ║")
        
        # Print CLIP image similarity if available
        if self.ori_flag and "clip_i" in general:
            print(f"║ {Fore.CYAN}CLIP image similarity:{' ':>31} ║")
            self._print_stat_row("clip_i", general)
        
        # Print CLIP text similarity
        print(f"║ {Fore.CYAN}CLIP text similarity:{' ':>32} ║")
        self._print_stat_row("clip_t", general)
        
        # Print aesthetic score
        print(f"║ {Fore.CYAN}Aesthetic score:{' ':>36} ║")
        self._print_stat_row("aesthetic", general)
        
        # Print anatomy score
        print(f"║ {Fore.CYAN}Anatomy score:{' ':>38} ║")
        self._print_stat_row("anatomy", general)
        
        # Print FID if available
        if self.ori_flag and "fid" in general:
            print(f"║ {Fore.GREEN}FID score:{Fore.CYAN}                {general['fid']:.4f}{' ':>21} ║")
    
    def _print_footer(self):
        """Print summary footer"""
        print(f"╠═══════════════════════════════════════════════════════╣")
        print(f"║ {Fore.WHITE}Evaluation completed in {self.duration:.2f} seconds{' ':>13} ║")
        print(f"╚═══════════════════════════════════════════════════════╝{Style.RESET_ALL}")
    
    def _print_stat_row(self, metric_name, stat_dict, prefix=""):
        """Print statistics for a specific metric"""
        if not stat_dict or metric_name not in stat_dict:
            return
            
        stats = stat_dict[metric_name]
        if not stats:
            return
        
        # Print detailed stats for important metrics
        if metric_name in ["sim_ori", "copy_geo"]:
            print(f"║ {Fore.GREEN}{prefix}Mean:{Fore.CYAN}    {stats['mean']:.4f}{' ':>30} ║")
            print(f"║ {Fore.GREEN}{prefix}Median:{Fore.CYAN}  {stats['median']:.4f}{' ':>30} ║")
            print(f"║ {Fore.GREEN}{prefix}StdDev:{Fore.CYAN}  {stats['std']:.4f}{' ':>30} ║")
            print(f"║ {Fore.GREEN}{prefix}Min:{Fore.CYAN}     {stats['min']:.4f}{' ':>30} ║")
            print(f"║ {Fore.GREEN}{prefix}Max:{Fore.CYAN}     {stats['max']:.4f}{' ':>30} ║")
            print(f"║ {Fore.GREEN}{prefix}Q1:{Fore.CYAN}      {stats['q1']:.4f}{' ':>30} ║")
            print(f"║ {Fore.GREEN}{prefix}Q3:{Fore.CYAN}      {stats['q3']:.4f}{' ':>30} ║")
        else:
            # Only print mean for other metrics
            print(f"║ {Fore.GREEN}{prefix}Mean:{Fore.CYAN}    {stats['mean']:.4f}{' ':>30} ║")




def v200_single(target_dir, output_dir):
    evaler = BenchEval_Geo(
        target_dir=target_dir,
        output_dir=output_dir,
        ori_file_name="ori.jpg",
        output_file_name="out.jpg",
        ref_1_file_name="ref_1.jpg",
        # ref_2_file_name="ref_2.jpg",
        ref_2_file_name=None,
        caption_keyword="prompt",
        names_keyword=None
    )
    evaler()
def v202(target_dir, output_dir):
    evaler = BenchEval_Geo(
        target_dir=target_dir,
        output_dir=output_dir,
        ori_file_name="ori.jpg",
        output_file_name="out.jpg",
        ref_1_file_name="ref_0.jpg",
        ref_2_file_name="ref_1.jpg",
        # ref_2_file_name=None,
        caption_keyword="prompt",
        names_keyword=None
    )
    evaler()

if __name__ == "__main__":
    v202("../debug_test", "../debug_test/eval")