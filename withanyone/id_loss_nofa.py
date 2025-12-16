import os
import random
import json
from typing import List, Tuple, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import skimage.transform as trans
from PIL import Image
from scipy.optimize import linear_sum_assignment
from insightface.model_zoo import model_zoo
import insightface

try:
    from info_nce import InfoNCE, info_nce
except ImportError:
    InfoNCE = None
    info_nce = None

# Constants
REF_CLUSTER_CENTER = "/mnt/xuhengyuan/data/2person/v5/ref/npy/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

# ArcFace source points
ARCFACE_SRC = torch.tensor(
    [[38.2946, 51.6963], 
     [73.5318, 51.5014], 
     [56.0252, 71.7366],
     [41.5493, 92.3655], 
     [70.7299, 92.2041]],
    dtype=torch.float32, 
    device=DEVICE
)


def estimate_affine_torch(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    Estimate affine transformation matrix using PyTorch.
    Follows affine transform in skimage.
    """
    n, d = src.shape
    _coeffs = range(d * d + 1)
    
    src_matrix, src = _center_and_normalize_points_torch(src)
    dst_matrix, dst = _center_and_normalize_points_torch(dst)
    
    if not torch.all(torch.isfinite(src_matrix + dst_matrix)):
        params = torch.full((d + 1, d + 1), torch.nan, device=DEVICE)
        return params

    # params: a0, a1, a2, b0, b1, b2, c0, c1
    A = torch.zeros((n * d, (d + 1) ** 2), device=DEVICE)
    
    for ddim in range(d):
        A[ddim * n : (ddim + 1) * n, ddim * (d + 1) : ddim * (d + 1) + d] = src
        A[ddim * n : (ddim + 1) * n, ddim * (d + 1) + d] = 1
        A[ddim * n : (ddim + 1) * n, -d - 1 : -1] = src
        A[ddim * n : (ddim + 1) * n, -1] = -1
        A[ddim * n : (ddim + 1) * n, -d - 1 :] *= -dst[:, ddim : (ddim + 1)]

    # Select relevant columns, depending on params
    A = A[:, list(_coeffs) + [-1]]

    # Get the vectors that correspond to singular values
    _, _, V = torch.linalg.svd(A)

    # Check for degenerate case
    if torch.isclose(V[-1, -1], torch.tensor(0., device=DEVICE)):
        params = torch.full((d + 1, d + 1), torch.nan, device=DEVICE)
        return params

    H = torch.zeros((d + 1, d + 1), device=DEVICE)
    # Solution is right singular vector that corresponds to smallest singular value
    H.view(-1)[list(_coeffs)] = -V[-1, :-1] / V[-1, -1]
    H[d, d] = 1

    # De-center and de-normalize
    H = torch.linalg.inv(dst_matrix) @ H @ src_matrix

    # Correct for small errors
    H /= H[-1, -1].clone()

    return H


def _center_and_normalize_points_torch(points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Center and normalize points for affine estimation.
    """
    n, d = points.shape
    centroid = torch.mean(points, axis=0)

    centered = points - centroid
    rms = torch.sqrt(torch.sum(centered**2) / n)

    if rms == 0:
        return torch.full((d + 1, d + 1), torch.nan, device=DEVICE), torch.full_like(points, torch.nan)

    norm_factor = torch.sqrt(torch.tensor(d, device=DEVICE)) / rms
    part_matrix = norm_factor * torch.concat(
        (torch.eye(d, device=DEVICE), -centroid[:, None]), axis=1)
    matrix = torch.concat(
        (
            part_matrix, torch.tensor([[0,] * d + [1]], device=DEVICE),
        ),
        axis=0,
    )

    points_h = torch.vstack([points.T, torch.ones(n, device=DEVICE)])
    new_points_h = (matrix @ points_h).T

    new_points = new_points_h[:, :d]
    new_points /= new_points_h[:, d:]

    return matrix, new_points


def align_face(img: torch.Tensor, landmark: torch.Tensor, image_size: int = 112) -> torch.Tensor:
    """
    Align face using landmarks.
    
    Args:
        img: (H,W,C) - full image
        landmark: shape(5,2) - facial landmarks in full image
        image_size: Output image size
    """
    img_hei, img_wid = img.shape[:2]
    device = img.device
    float_dtype = img.dtype
    
    src = landmark.to(device=device, dtype=float_dtype)
    dst = ARCFACE_SRC.to(device=device, dtype=float_dtype)
    
    src = src / torch.tensor([img_wid, img_hei], dtype=float_dtype, device=device) * 2 - 1
    dst = dst / torch.tensor([image_size, image_size], dtype=float_dtype, device=device) * 2 - 1

    theta = estimate_affine_torch(dst, src)
    theta.unsqueeze_(0)
    
    # Process image tensor
    default_float_dtype = float_dtype
    img_tensor = img.permute((2, 0, 1)).contiguous()
    img_tensor = img_tensor / 255.0
    img_tensor = img_tensor.unsqueeze(0)
    
    output_size = torch.Size((1, 3, image_size, image_size))
    grid = F.affine_grid(theta[:, :2], output_size).to(device=device, dtype=default_float_dtype)
    img_tensor = F.grid_sample(img_tensor, grid, align_corners=False, mode='bicubic')
    img_tensor = img_tensor.squeeze(0)
    
    aligned_img = img_tensor.permute((1, 2, 0)) * 255
    
    return aligned_img


def estimate_norm_torch(lmk: Union[torch.Tensor, np.ndarray], image_size: int = 112, mode: str = 'arcface', device: torch.device = None) -> torch.Tensor:
    """
    PyTorch version of estimate_norm function.
    
    Args:
        lmk: 5 facial landmarks of shape (5, 2)
        image_size: Output image size
        mode: Alignment mode
        device: Device to place tensors on
        
    Returns:
        Transformation matrix of shape (2, 3)
    """
    if not isinstance(lmk, torch.Tensor):
        lmk = torch.tensor(lmk, dtype=torch.float32, device=device)
    else:
        device = lmk.device if device is None else device
        lmk = lmk.to(device)
    
    assert lmk.shape == (5, 2)
    assert image_size % 112 == 0 or image_size % 128 == 0
    
    arcface_dst_torch = torch.tensor(
        [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
         [41.5493, 92.3655], [70.7299, 92.2041]],
        dtype=torch.float32, device=device)
    
    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0
    else:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio
    
    dst = arcface_dst_torch * ratio
    dst[:, 0] += diff_x
    
    # Convert to numpy for transform estimation (required by scikit-image)
    lmk_np = lmk.detach().cpu().numpy()
    dst_np = dst.detach().cpu().numpy()
    
    tform = trans.SimilarityTransform()
    tform.estimate(lmk_np, dst_np)
    M_np = tform.params[0:2, :]
    
    M = torch.tensor(M_np, dtype=torch.float32, device=device)
    return M


class DET:
    def __init__(self):
        arcface_detect_path = "./models/scrfd_10g_bnkps.onnx"
        self.model_det = model_zoo.get_model(arcface_detect_path, providers=['CPUExecutionProvider'])
        self.model_det.prepare(ctx_id=0, det_thresh=0.4, input_size=(640, 640))

    def __call__(self, image: torch.Tensor):
        image = image.clone().detach().cpu().to(torch.float32).numpy().transpose(1, 2, 0)
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        bboxes, kpss = self.model_det.detect(image)
        return bboxes, kpss


def norm_crop_torch(img: Union[torch.Tensor, np.ndarray], landmark: Union[torch.Tensor, np.ndarray], image_size: int = 112, mode: str = 'arcface') -> torch.Tensor:
    """
    PyTorch version of norm_crop function using torch.nn.functional.grid_sample.
    """
    device = img.device if isinstance(img, torch.Tensor) else None
    
    if not isinstance(img, torch.Tensor):
        if isinstance(img, np.ndarray):
            if img.ndim == 3 and img.shape[2] in [1, 3, 4]:
                img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).float()
        else:
            from torchvision import transforms
            img = transforms.ToTensor()(img)
    
    if device is not None:
        img = img.to(device)
        
    need_squeeze = False
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
        need_squeeze = True
    
    M = estimate_norm_torch(landmark, image_size, mode, device=device)
    batch_size, channels, height, width = img.shape
    
    M_full = torch.eye(3, device=device)
    M_full[:2, :] = M
    
    src_norm = torch.tensor([
        [2.0/width, 0, -1],
        [0, 2.0/height, -1],
        [0, 0, 1]
    ], dtype=torch.float32, device=device)
    
    dst_norm = torch.tensor([
        [2.0/image_size, 0, -1],
        [0, 2.0/image_size, -1],
        [0, 0, 1]
    ], dtype=torch.float32, device=device)
    
    theta = src_norm @ torch.inverse(M_full) @ torch.inverse(dst_norm)
    theta = theta[:2, :].unsqueeze(0)
    
    grid = torch.nn.functional.affine_grid(theta, [batch_size, channels, image_size, image_size], align_corners=False)
    grid = grid.type_as(img)
    warped = torch.nn.functional.grid_sample(img, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    
    if need_squeeze:
        warped = warped.squeeze(0)
    
    return warped


def detect_face_pose(landmarks: np.ndarray, threshold: float = 0.78) -> str:
    """
    Determine if a face is front-facing or side-facing based on landmarks.
    
    Args:
        landmarks: 5-point facial landmarks [left_eye, right_eye, nose, left_mouth, right_mouth]
        threshold: Threshold for determining front vs side face (0.0-1.0)
    
    Returns:
        "front", "side", or "profile" string indicating the face pose
    """
    left_eye, right_eye = landmarks[0], landmarks[1]
    
    left_eye_x, right_eye_x = left_eye[0], right_eye[0]
    eye_distance = abs(right_eye_x - left_eye_x)
    
    face_width = np.max(landmarks[:, 0]) - np.min(landmarks[:, 0])
    eye_distance_ratio = eye_distance / (face_width + 1e-6)
    
    if eye_distance_ratio > threshold:
        return "front"
    elif eye_distance_ratio > threshold * 0.5:
        return "partial_profile"
    else:
        return "profile"


class IDLoss:
    def __init__(self, device: str = 'cuda', use_state_negative_pool: bool = False):
        self.device = device
        self.netArc = torch.load('./models/glintr100.pth', map_location=torch.device("cpu"))
        self.netArc = self.netArc.to(self.device, dtype=torch.bfloat16)
        self.netArc.eval()
        self.netArc.requires_grad_(True)
        self.netDet = DET()
        self.dtype = torch.bfloat16

        if use_state_negative_pool:
            self._init_negative_pool()

    def _init_negative_pool(self):
        """
        Load all npy embeddings from REF_CLUSTER_CENTER.
        """
        self.negative_pool = []
        for npy_file in os.listdir(REF_CLUSTER_CENTER):
            if npy_file.endswith('.npy'):
                npy_path = os.path.join(REF_CLUSTER_CENTER, npy_file)
                embedding = np.load(npy_path, allow_pickle=True)
                if isinstance(embedding, np.ndarray) and embedding.ndim == 2 and embedding.shape[1] == 512:
                    self.negative_pool.append(torch.tensor(embedding, dtype=self.dtype, device=self.device))
                elif len(embedding) == 512:
                    self.negative_pool.append(torch.tensor(embedding, dtype=self.dtype, device=self.device))
                
        if len(self.negative_pool) == 0:
            raise ValueError("No valid embeddings found in the negative pool directory.") 
        self.negative_pool = torch.stack(self.negative_pool, dim=0)

    def _iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        """
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        union_area = bbox1_area + bbox2_area - intersection_area
        
        if union_area == 0:
            return 0.0
        
        return intersection_area / union_area
    

    def get_arcface_embeddings(self, images: torch.Tensor, gt_images: torch.Tensor, check_side_views: bool = False, original_bboxes: Optional[List] = None, num_id: Optional[int] = None) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Get ArcFace embeddings for a batch of images, handling multiple faces per image.
        Optimized to use batch inference.
        Uses GT landmarks for alignment of both generated and GT images.
        """
        self.netArc.eval()
        # Freeze model weights. Gradients will still flow through the input images.
        self.netArc.requires_grad_(False)
        
        if len(images.shape) == 3:
            images = images.unsqueeze(0)
        
        batch_size = images.shape[0]
        
        all_aligned_faces = []
        all_aligned_gt_faces = []
        faces_per_image = []
        
        if num_id is None:
            num_id = len(original_bboxes[0]) if original_bboxes is not None else 0

        for i in range(batch_size):
            image = images[i]
            gt_image = gt_images[i]

            # Detection on GT image
            with torch.no_grad():
                bboxes, landmarks = self.netDet(gt_image)

            # Select landmarks based on num_id strategy
            if num_id == 1:
                if len(landmarks) == 0:
                    landmarks = [ARCFACE_SRC, ARCFACE_SRC]
                else:
                    landmarks = [landmarks[0], landmarks[0]]
            else:
                if len(landmarks) == 0:
                    landmarks = [ARCFACE_SRC, ARCFACE_SRC]
                elif len(landmarks) == 1:
                    landmarks = [landmarks[0], landmarks[0]]
                elif len(landmarks) >= 2:
                    landmarks = landmarks[:2]

            # Convert landmarks to tensors
            if isinstance(landmarks, (list, tuple, np.ndarray)):
                landmarks = [torch.tensor(lmk, dtype=torch.float32, device=self.device) 
                                if not isinstance(lmk, torch.Tensor) else lmk.to(self.device)
                                for lmk in landmarks]

            # Align faces
            for landmark in landmarks:
                # Align Generated Image (Gradients flow here)
                aimg = align_face(image.permute(1, 2, 0), landmark).permute(2, 0, 1)
                all_aligned_faces.append(aimg)

                # Align GT Image
                gt_aimg = align_face(gt_image.permute(1, 2, 0), landmark).permute(2, 0, 1)
                all_aligned_gt_faces.append(gt_aimg)
            
            faces_per_image.append(len(landmarks))
        
        # Batch Inference
        if not all_aligned_faces:
            empty = torch.empty(0, 512, device=self.device)
            return [empty] * batch_size, [empty] * batch_size

        # Stack and Normalize
        batch_tensor = torch.stack(all_aligned_faces)
        batch_gt_tensor = torch.stack(all_aligned_gt_faces)
        
        batch_tensor = (batch_tensor - 127.5) / 127.5
        batch_gt_tensor = (batch_gt_tensor - 127.5) / 127.5
        
        # Forward pass
        # Gen: requires grad flow
        embeddings_batch = self.netArc(batch_tensor.to(dtype=self.dtype))
        
        # GT: no grad needed
        with torch.no_grad():
            gt_embeddings_batch = self.netArc(batch_gt_tensor.to(dtype=self.dtype))

        # Unpack results
        all_embeddings = []
        all_gt_embeddings = []
        cursor = 0
        for count in faces_per_image:
            if count > 0:
                all_embeddings.append(embeddings_batch[cursor : cursor + count])
                all_gt_embeddings.append(gt_embeddings_batch[cursor : cursor + count])
                cursor += count
            else:
                empty = torch.empty(0, 512, device=self.device)
                all_embeddings.append(empty)
                all_gt_embeddings.append(empty)
        
        return all_embeddings, all_gt_embeddings
    
    
    def compute_id_loss_with_embeddings(self, generated_arcface_embeddings: List[torch.Tensor], ground_truth_arcface_embeddings: List[torch.Tensor], rec_bbox_A=None, rec_bbox_B=None, filter_out_side_views: bool = False, original_bboxes=None) -> torch.Tensor:
        """
        Compute identity loss using pre-computed embeddings.
        """
        id_losses = []

        for i in range(len(generated_arcface_embeddings)):
            gen_emb = generated_arcface_embeddings[i]
            gt_emb = ground_truth_arcface_embeddings[i]

            # Ideal case 1: both have 2 faces
            if gen_emb.shape[0] == 2 and gt_emb.shape[0] == 2:
                if original_bboxes is None:
                    cossim_AA = F.cosine_similarity(gen_emb[0], gt_emb[0], dim=0)
                    cossim_BB = F.cosine_similarity(gen_emb[1], gt_emb[1], dim=0)
                    cossim_AB = F.cosine_similarity(gen_emb[0], gt_emb[1], dim=0)
                    cossim_BA = F.cosine_similarity(gen_emb[1], gt_emb[0], dim=0)

                    if cossim_AA + cossim_BB > cossim_AB + cossim_BA:
                        id_loss = 1 - (cossim_AA + cossim_BB) / 2
                    else:
                        id_loss = 1 - (cossim_AB + cossim_BA) / 2
                else:
                    cossim_AA = F.cosine_similarity(gen_emb[0], gt_emb[0], dim=0)
                    cossim_BB = F.cosine_similarity(gen_emb[1], gt_emb[1], dim=0)
                    id_loss = 1 - (cossim_AA + cossim_BB) / 2
            
            # Ideal case 2: one face
            elif gen_emb.shape[0] == 1 and gt_emb.shape[0] == 1:
                cossim = F.cosine_similarity(gen_emb[0], gt_emb[0], dim=0)
                id_loss = 1 - cossim
            
            # Non-ideal cases
            else:
                # Return a dummy loss with grad to prevent graph breaks, or 0.0
                return torch.tensor(0.0, device=self.device, requires_grad=True)
            
            id_losses.append(id_loss)
        
        if not id_losses:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        id_loss = torch.mean(torch.stack(id_losses))
        return id_loss

    
    def compute_contrastive_loss(self, generated_embeddings: torch.Tensor, ground_truth_embeddings: torch.Tensor, 
                              generated_labels=None, ground_truth_labels=None, 
                              temperature: float = 0.07, max_negatives: int = 10) -> torch.Tensor:
        """
        Efficient InfoNCE for scenarios with fixed 2 faces per sample.
        
        Args:
            generated_embeddings: [B, 2, D] tensor
            ground_truth_embeddings: [B, 2, D] tensor
            temperature: Temperature coefficient
            max_negatives: Max negative samples per anchor
        """
        device = self.device
        dtype = self.dtype
    
        gen = generated_embeddings.to(device=device, dtype=dtype)
        gt = ground_truth_embeddings.to(device=device, dtype=dtype)
        
        # Fix shape mismatch
        if len(gen.shape) == 4:
            B, num_faces, extra_dim, D = gen.shape
            gen = gen.reshape(B, num_faces * extra_dim, D)
        elif len(gen.shape) == 3:
            B, num_faces, D = gen.shape
        else:
            raise ValueError(f"Unexpected shape for gen tensor: {gen.shape}")
        
        if len(gt.shape) == 3:
            if gt.shape[1] != gen.shape[1]:
                gt = gt[:, :gen.shape[1], :]
        
        # Normalize
        gen = F.normalize(gen, dim=2)
        gt  = F.normalize(gt, dim=2)
        
        faces_per_sample = gen.shape[1]  

        if faces_per_sample != 2:
            print(f"Warning: Expected 2 faces per sample, but got {faces_per_sample} faces. This loss function is designed for 2 faces only.")
        
        matched_gt = gt
    
        # Construct positive pairs
        anchors = gen.reshape(B * faces_per_sample, D)
        positives = matched_gt.reshape(B * faces_per_sample, D)
    
        # Construct negative pool
        neg_pool = torch.cat([gen, gt], dim=1).reshape(B * 2 * faces_per_sample, D)
        neg_pool = F.normalize(neg_pool, dim=1)
    
        all_losses = []
        for i in range(B * faces_per_sample):
            anchor = anchors[i]
            positive = positives[i]
    
            # Index range of current sample (prevent false negatives)
            sample_id = i // faces_per_sample
            exclude_start = sample_id * 2 * faces_per_sample
            exclude_idx = torch.arange(exclude_start, exclude_start + 2 * faces_per_sample, device=device)
    
            # Valid negative sample indices
            neg_indices = torch.ones(len(neg_pool), dtype=torch.bool, device=device)
            neg_indices[exclude_idx] = False
            neg_indices = torch.where(neg_indices)[0]
    
            # Randomly sample negatives
            if len(neg_indices) > max_negatives:
                perm = torch.randperm(len(neg_indices), device=device)[:max_negatives]
                neg_indices = neg_indices[perm]
    
            negatives = neg_pool[neg_indices]
    
            # Similarity calculation
            pos_sim = (anchor * positive).sum() / temperature
            neg_sims = (negatives @ anchor) / temperature
    
            # InfoNCE
            numerator = torch.exp(pos_sim)
            denominator = numerator + torch.sum(torch.exp(neg_sims))
            loss = -torch.log(numerator / denominator)
            all_losses.append(loss)
    
        return torch.stack(all_losses).mean()
    
    def compute_info_nce_loss(self, generated_embeddings: torch.Tensor, ground_truth_embeddings: torch.Tensor, extend_negative_pool: Optional[torch.Tensor] = None) -> torch.Tensor:
        device = self.device
        dtype = self.dtype
    
        gen = generated_embeddings.to(device=device, dtype=dtype)
        gt = ground_truth_embeddings.to(device=device, dtype=dtype)
        
        if len(gen.shape) == 4:
            B, num_faces, extra_dim, D = gen.shape
            gen = gen.reshape(B, num_faces * extra_dim, D)
        elif len(gen.shape) == 3:
            B, num_faces, D = gen.shape
        else:
            raise ValueError(f"Unexpected shape for gen tensor: {gen.shape}")
        
        if len(gt.shape) == 3:
            if gt.shape[1] != gen.shape[1]:
                gt = gt[:, :gen.shape[1], :]
        
        gen = F.normalize(gen, dim=2)
        gt  = F.normalize(gt, dim=2)
        
        faces_per_sample = gen.shape[1]  

        if faces_per_sample != 2:
            print(f"Warning: Expected 2 faces per sample, but got {faces_per_sample} faces. This loss function is designed for 2 faces only.")

        loss = InfoNCE(negative_mode='paired')

        # Construct query and positive
        total_samples = B * num_faces
        query = gen.reshape(total_samples, D)
        positive_keys = gt.reshape(total_samples, D)
        
        # Construct negative pool from gt negative samples
        negative_keys = []
        for i in range(total_samples):
            neg_indices = torch.ones(total_samples, dtype=torch.bool, device=device)
            neg_indices[i] = False
            neg_keys = positive_keys[neg_indices]
            neg_keys = neg_keys[torch.randperm(len(neg_keys), device=device)]
            negative_keys.append(neg_keys)

        negative_keys = torch.stack(negative_keys, dim=0)
        
        if extend_negative_pool is not None:
            ext_neg = extend_negative_pool.to(device=device, dtype=dtype)
            if len(ext_neg.shape) == 4:
                B_ext, num_faces_ext, num_samples_ext, D_ext = ext_neg.shape
                ext_neg = ext_neg.reshape(B_ext * num_faces_ext, num_samples_ext, D_ext)
            else:
                raise ValueError(f"Unexpected shape for extend_negative_pool: {ext_neg.shape}")
            negative_keys = torch.cat([negative_keys, ext_neg], dim=1)
            
        return loss(query, positive_keys, negative_keys)

    def region_diffusion_loss(self, decoded_images: torch.Tensor, ground_truth_image: torch.Tensor, bboxes_A: List, bboxes_B: List = None, 
                                   weights_A: Union[float, List[float]] = 1.0, weights_B: Union[float, List[float]] = 1.0, background_weight: float = -1,
                                   loss_type: str = 'mse', normalize_by_area: bool = True) -> torch.Tensor:
        """
        Optimized regional diffusion loss for face restoration.
        """
        b, c, h, w = decoded_images.shape
        loss_list = []
        background_losses = []

        decoded_images = decoded_images.float()
        ground_truth_image = ground_truth_image.float()
        
        if not isinstance(weights_A, (list, tuple)):
            weights_A = [weights_A] * b
        if bboxes_B is not None and not isinstance(weights_B, (list, tuple)):
            weights_B = [weights_B] * b
        
        for i in range(b):
            region_masks = []
            
            # Handle A bounding box
            if bboxes_A is not None:
                bbox_A = bboxes_A[i]
                y1_A, x1_A = max(0, int(bbox_A[1])), max(0, int(bbox_A[0]))
                y2_A, x2_A = min(h, int(bbox_A[3])), min(w, int(bbox_A[2]))
                
                if y2_A > y1_A and x2_A > x1_A:
                    decoded_region_A = decoded_images[i, :, y1_A:y2_A, x1_A:x2_A]
                    ground_truth_region_A = ground_truth_image[i, :, y1_A:y2_A, x1_A:x2_A]
                    
                    if loss_type == 'mse':
                        region_loss_A = torch.nn.functional.mse_loss(decoded_region_A, ground_truth_region_A, reduction='mean')
                    elif loss_type == 'l1':
                        region_loss_A = torch.nn.functional.l1_loss(decoded_region_A, ground_truth_region_A, reduction='mean')
                    elif loss_type == 'smooth_l1':
                        region_loss_A = torch.nn.functional.smooth_l1_loss(decoded_region_A, ground_truth_region_A, reduction='mean')
                    
                    if normalize_by_area:
                        area_A = (y2_A - y1_A) * (x2_A - x1_A)
                        region_loss_A = region_loss_A / (area_A + 1e-6)
                    
                    loss_list.append(weights_A[i] * region_loss_A)
                    
                    if background_weight > 0:
                        mask_A = torch.zeros((1, h, w), device=decoded_images.device)
                        mask_A[:, y1_A:y2_A, x1_A:x2_A] = 1
                        region_masks.append(mask_A)
            
            # Handle B bounding box
            if bboxes_B is not None:
                bbox_B = bboxes_B[i]
                y1_B, x1_B = max(0, int(bbox_B[1])), max(0, int(bbox_B[0]))
                y2_B, x2_B = min(h, int(bbox_B[3])), min(w, int(bbox_B[2]))
                
                if y2_B > y1_B and x2_B > x1_B:
                    decoded_region_B = decoded_images[i, :, y1_B:y2_B, x1_B:x2_B]
                    ground_truth_region_B = ground_truth_image[i, :, y1_B:y2_B, x1_B:x2_B]
                    
                    if loss_type == 'mse':
                        region_loss_B = torch.nn.functional.mse_loss(decoded_region_B, ground_truth_region_B, reduction='mean')
                    elif loss_type == 'l1':
                        region_loss_B = torch.nn.functional.l1_loss(decoded_region_B, ground_truth_region_B, reduction='mean')
                    elif loss_type == 'smooth_l1':
                        region_loss_B = torch.nn.functional.smooth_l1_loss(decoded_region_B, ground_truth_region_B, reduction='mean')
                    
                    if normalize_by_area:
                        area_B = (y2_B - y1_B) * (x2_B - x1_B)
                        region_loss_B = region_loss_B / (area_B + 1e-6)
                    
                    loss_list.append(weights_B[i] * region_loss_B)
                    
                    if background_weight > 0:
                        mask_B = torch.zeros((1, h, w), device=decoded_images.device)
                        mask_B[:, y1_B:y2_B, x1_B:x2_B] = 1
                        region_masks.append(mask_B)
            
            # Calculate background loss
            if background_weight > 0 and region_masks:
                combined_mask = torch.clamp(torch.sum(torch.stack(region_masks), dim=0), 0, 1)
                background_mask = 1 - combined_mask
                
                decoded_bg = decoded_images[i] * background_mask
                ground_truth_bg = ground_truth_image[i] * background_mask
                
                if torch.sum(background_mask) > 0:
                    if loss_type == 'mse':
                        bg_loss = torch.nn.functional.mse_loss(decoded_bg, ground_truth_bg, reduction='mean')
                    elif loss_type == 'l1':
                        bg_loss = torch.nn.functional.l1_loss(decoded_bg, ground_truth_bg, reduction='mean')
                    elif loss_type == 'smooth_l1':
                        bg_loss = torch.nn.functional.smooth_l1_loss(decoded_bg, ground_truth_bg, reduction='mean')
                    
                    background_losses.append(background_weight * bg_loss)
        
        all_losses = loss_list + background_losses
        if all_losses:
            total_loss = torch.mean(torch.stack(all_losses))
        else:
            total_loss = torch.nn.functional.mse_loss(decoded_images, ground_truth_image)
            
        return total_loss

    def get_arcface_embeddings_with_features(self, images: torch.Tensor, expected_num_faces: List[int] = []) -> Tuple[List, torch.Tensor, bool, List]:
        """
        Get both pooled ArcFace embeddings and pre-pooling hidden features.
        """
        self.netArc.eval()
        self.netArc.requires_grad_(True)
        
        target_module = None
        for name, module in self.netArc.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d) and "251" in name:
                target_module = module
                break
        
        if target_module is None:
            for name, module in self.netArc.named_modules():
                if isinstance(module, torch.nn.BatchNorm2d) and module.num_features == 512:
                    target_module = module
                    print(f"Using alternative layer for hook: {name} with 512 features")
                    break
                    
        if target_module is None:
            print("Warning: Could not find suitable layer for hook. Hidden features may not be captured.")
        
        if len(images.shape) == 3:
            images = images.unsqueeze(0)
        
        batch_size = images.shape[0]
        all_embeddings = []
        all_hidden_features = []
        all_bboxes = []
    
        for batch_idx in range(batch_size):
            image = images[batch_idx]
            expected_num = expected_num_faces[batch_idx] if expected_num_faces else 2
            
            image_embeddings = []
            image_hidden_feature_np = []
            force_align = False
            
            with torch.no_grad():
                bboxes, landmarks = self.netDet(image)
            
            all_bboxes.append(bboxes)
            
            if len(landmarks) < 2:
                landmarks = [ARCFACE_SRC] * 2
                force_align = True
            elif len(landmarks) > 2:
                landmarks = landmarks[:2]
                force_align = expected_num <= 2
            
            if isinstance(landmarks, (list, tuple, np.ndarray)):
                new_landmarks = []
                for lmk in landmarks:
                    if isinstance(lmk, np.ndarray):
                        new_landmarks.append(torch.from_numpy(lmk).float().to(image.device))
                    elif isinstance(lmk, torch.Tensor):
                        new_landmarks.append(lmk.float().to(image.device))
                    new_landmarks.append(lmk)
                landmarks = new_landmarks
            
            print(f"Processing image {batch_idx} with {len(landmarks)} faces detected")
            for landmark_idx, landmark in enumerate(landmarks):
                features_holder = []
                
                def hook_fn(module, input, output):
                    features_holder.append(output.detach().clone())
                
                hook = None
                if target_module is not None:
                    hook = target_module.register_forward_hook(hook_fn)
                
                try:
                    aimg = align_face(image.permute(1, 2, 0), landmark).permute(2, 0, 1)
                    aimg = aimg.unsqueeze(0).to(dtype=torch.float32)
                    aimg = (aimg - 127.5) / 127.5
                    id_face = aimg
                    
                    embedding = self.netArc(id_face.to(dtype=self.dtype))
                    image_embeddings.append(embedding)
                    
                    if features_holder:
                        image_hidden_feature_np.append(features_holder[0])
                    else:
                        # Handle case where hook didn't capture anything
                        pass
                    
                finally:
                    if hook is not None:
                        hook.remove()
                        
            image_hidden_features = torch.stack(image_hidden_feature_np, dim=0)
            
            all_embeddings.append(image_embeddings)
            all_hidden_features.append(image_hidden_features)

        all_hidden_features = torch.stack(all_hidden_features, dim=0)
        
        return all_embeddings, all_hidden_features, force_align, all_bboxes
    
    def __call__(self, decoded_images: torch.Tensor, ground_truth_arcface_embeddings: torch.Tensor, ground_truth_image: torch.Tensor, bboxes_A: List, bboxes_B: List = None, regional_mse_weight: float = 3) -> torch.Tensor:
        """
        Compute the ID loss and regional diffusion loss.
        """
        id_loss = self.compute_id_loss(decoded_images, ground_truth_arcface_embeddings)
        
        if regional_mse_weight > 0:
            regional_loss = self.region_diffusion_loss(decoded_images, ground_truth_image, bboxes_A, bboxes_B=bboxes_B)
            return id_loss + regional_mse_weight * regional_loss
        
        return id_loss


def single_face_preserving_resize(img: Image.Image, face_bbox: List[float], target_size: int = 512) -> Optional[Image.Image]:
    """
    Resize image while ensuring a single face is preserved in the output.
    """
    x1, y1, x2, y2 = map(int, face_bbox)
    
    if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
        return None

    face_width = x2 - x1
    face_height = y2 - y1
    if face_width > img.height or face_height > img.width:
        return None

    if img.width > img.height:
        square_size = img.height
        left_max = x1
        right_min = x2 - square_size
        
        if right_min <= left_max:
            start = random.randint(int(right_min), int(left_max)) if right_min < left_max else int(right_min)
            start = max(0, min(start, img.width - square_size))
        else:
            face_center = (x1 + x2) // 2
            start = max(0, min(face_center - (square_size // 2), img.width - square_size))
        
        cropped_img = img.crop((start, 0, start + square_size, square_size))
    else:
        square_size = img.width
        top_max = y1
        bottom_min = y2 - square_size
        
        if bottom_min <= top_max:
            start = random.randint(int(bottom_min), int(top_max)) if bottom_min < top_max else int(bottom_min)
            start = max(0, min(start, img.height - square_size))
        else:
            face_center = (y1 + y2) // 2
            start = max(0, min(face_center - (square_size // 2), img.height - square_size))
        
        cropped_img = img.crop((0, start, square_size, start + square_size))
    
    cropped_img = cropped_img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    return cropped_img


if __name__ == "__main__":
    # Test for the IDLoss class
    image_paths = [
        "/data/MIBM/BenCon/benchmarks/v200/untar_cn/0bb9e7100a32c4324bbd87e38ed0a11dba8876c0.jpg",
        "/data/MIBM/BenCon/benchmarks/v200/untar_cn/0bb9e7100a32c4324bbd87e38ed0a11dba8876c0.jpg"
    ]
    images = []
    images_np = []
    for image_path in image_paths:
        if not os.path.exists(image_path):
            print(f"File not found: {image_path}")
            continue
            
        img = Image.open(image_path).convert("RGB")
        json_path = image_path.replace(".jpg", ".json")
        
        if os.path.exists(json_path):
            json_dict = json.load(open(json_path))
            img_face = single_face_preserving_resize(img, json_dict["bboxes"][0])
            if img_face:
                img_face.save("debug_face.jpg")
                img_face = np.array(img_face) 
                images.append(torch.tensor(img_face, dtype=torch.float32).permute(2, 0, 1))
                images_np.append(cv2.cvtColor(img_face, cv2.COLOR_RGB2BGR))

    if len(images) >= 2:
        id_loss_model = IDLoss()
        official_model = insightface.app.FaceAnalysis(name="antelopev2", root="./models", providers=['CUDAExecutionProvider'])
        official_model.prepare(ctx_id=0, det_thresh=0.4)
        
        arcface_embeddings_1_official = official_model.get(images_np[0])
        arcface_embeddings_2_official = official_model.get(images_np[1])
        
        if arcface_embeddings_1_official and arcface_embeddings_2_official:
            arcface_embeddings_1_official = torch.tensor(arcface_embeddings_1_official[0].embedding, dtype=torch.float32).unsqueeze(0)
            arcface_embeddings_2_official = torch.tensor(arcface_embeddings_2_official[0].embedding, dtype=torch.float32).unsqueeze(0)
            print(f"Official embeddings shapes: {arcface_embeddings_1_official.shape}, {arcface_embeddings_2_official.shape}")
            
            cos_sim = torch.nn.functional.cosine_similarity(arcface_embeddings_1_official, arcface_embeddings_2_official)
            print("Cosine similarity between official embeddings:", cos_sim.item())

            images = torch.stack(images).to(id_loss_model.device, dtype=torch.float32)
            print("Images shape:", images.shape)
            embeddings = id_loss_model.get_arcface_embeddings(images, images) # Passing images as gt_images for test
            print("Embeddings shapes:", [emb[0].shape for emb in embeddings[0]])
            
            if len(embeddings[0]) >= 2 and len(embeddings[0][0]) > 0 and len(embeddings[0][1]) > 0:
                cos_ours = torch.nn.functional.cosine_similarity(embeddings[0][0][0], embeddings[0][1][0])
                print("Cosine similarity between our embeddings:", cos_ours.item())
                
                cos_AA = torch.nn.functional.cosine_similarity(embeddings[0][0][0].to("cpu"), arcface_embeddings_1_official)
                cos_BB = torch.nn.functional.cosine_similarity(embeddings[0][1][0].to("cpu"), arcface_embeddings_2_official)
                print("Cosine similarity between our embeddings and official embeddings:", cos_AA.item(), cos_BB.item())