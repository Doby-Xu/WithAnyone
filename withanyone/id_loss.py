import os
import json
import random
from typing import List, Tuple, Optional, Union, Any

import torch
import torch.nn.functional as F
import numpy as np
import skimage.transform as trans
from PIL import Image
from scipy.optimize import linear_sum_assignment
from insightface.model_zoo import model_zoo
from info_nce import InfoNCE

# ==============================================================================
# Constants & Configuration
# ==============================================================================

REF_CLUSTER_CENTER = "/mnt/xuhengyuan/data/2person/v5/ref/npy/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

# Standard 5 facial landmarks for ArcFace alignment
ARCFACE_SRC = torch.tensor(
    [[38.2946, 51.6963],
     [73.5318, 51.5014],
     [56.0252, 71.7366],
     [41.5493, 92.3655],
     [70.7299, 92.2041]],
    dtype=torch.float32,
    device=DEVICE
)

# ==============================================================================
# Helper Functions: Geometry & Alignment
# ==============================================================================

def _center_and_normalize_points_torch(points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Centers and normalizes a set of points.
    """
    n, d = points.shape
    centroid = torch.mean(points, axis=0)

    centered = points - centroid
    rms = torch.sqrt(torch.sum(centered**2) / n)

    if rms == 0:
        return torch.full((d + 1, d + 1), torch.nan, device=points.device), torch.full_like(points, torch.nan)

    norm_factor = torch.sqrt(torch.tensor(d, device=points.device)) / rms
    part_matrix = norm_factor * torch.concat(
        (torch.eye(d, device=points.device), -centroid[:, None]), axis=1
    )
    matrix = torch.concat(
        (part_matrix, torch.tensor([[0,] * d + [1]], device=points.device)),
        axis=0,
    )

    points_h = torch.vstack([points.T, torch.ones(n, device=points.device)])
    new_points_h = (matrix @ points_h).T

    new_points = new_points_h[:, :d]
    new_points /= new_points_h[:, d:]

    return matrix, new_points


def estimate_affine_torch(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    Estimates the affine transformation matrix from src points to dst points using PyTorch.
    Follows the logic of skimage.transform.estimate_transform('affine', ...).
    """
    n, d = src.shape
    _coeffs = range(d * d + 1)

    src_matrix, src = _center_and_normalize_points_torch(src)
    dst_matrix, dst = _center_and_normalize_points_torch(dst)

    if not torch.all(torch.isfinite(src_matrix + dst_matrix)):
        return torch.full((d + 1, d + 1), torch.nan, device=DEVICE)

    # Construct matrix A
    A = torch.zeros((n * d, (d + 1) ** 2), device=DEVICE)
    for ddim in range(d):
        A[ddim * n : (ddim + 1) * n, ddim * (d + 1) : ddim * (d + 1) + d] = src
        A[ddim * n : (ddim + 1) * n, ddim * (d + 1) + d] = 1
        A[ddim * n : (ddim + 1) * n, -d - 1 : -1] = src
        A[ddim * n : (ddim + 1) * n, -1] = -1
        A[ddim * n : (ddim + 1) * n, -d - 1 :] *= -dst[:, ddim : (ddim + 1)]

    A = A[:, list(_coeffs) + [-1]]

    # SVD
    _, _, V = torch.linalg.svd(A)

    # Check for rank-defective transform
    if torch.isclose(V[-1, -1], torch.tensor(0., device=DEVICE)):
        return torch.full((d + 1, d + 1), torch.nan, device=DEVICE)

    H = torch.zeros((d + 1, d + 1), device=DEVICE)
    H.view(-1)[list(_coeffs)] = -V[-1, :-1] / V[-1, -1]
    H[d, d] = 1

    # De-center and de-normalize
    H = torch.linalg.inv(dst_matrix) @ H @ src_matrix

    # Correct small numerical errors
    H /= H[-1, -1].clone()

    return H


def align_face(img: torch.Tensor, landmark: torch.Tensor, image_size: int = 112) -> torch.Tensor:
    """
    Aligns a face in an image based on landmarks using affine transformation.

    Args:
        img: (H, W, C) full image tensor.
        landmark: (5, 2) facial landmarks.
        image_size: Output size (square).

    Returns:
        Aligned face image (H, W, C).
    """
    img_hei, img_wid = img.shape[:2]
    device = img.device
    float_dtype = img.dtype

    src = landmark.to(device=device, dtype=float_dtype)
    dst = ARCFACE_SRC.to(device=device, dtype=float_dtype)

    # Normalize coordinates to [-1, 1]
    src = src / torch.tensor([img_wid, img_hei], dtype=float_dtype, device=device) * 2 - 1
    dst = dst / torch.tensor([image_size, image_size], dtype=float_dtype, device=device) * 2 - 1

    theta = estimate_affine_torch(dst, src)
    theta.unsqueeze_(0)

    # Process image tensor: (H, W, C) -> (1, C, H, W)
    img_tensor = img.permute((2, 0, 1)).contiguous()
    img_tensor = img_tensor / 255.0
    img_tensor = img_tensor.unsqueeze(0)

    output_size = torch.Size((1, 3, image_size, image_size))
    grid = F.affine_grid(theta[:, :2], output_size).to(device=device, dtype=float_dtype)
    
    # Grid sample expects (N, C, H, W)
    img_tensor = F.grid_sample(img_tensor, grid, align_corners=False, mode='bicubic')
    img_tensor = img_tensor.squeeze(0)

    # Convert back to (H, W, C) and scale to 0-255
    aligned_img = img_tensor.permute((1, 2, 0)) * 255

    return aligned_img


def estimate_norm_torch(lmk: Union[torch.Tensor, np.ndarray], image_size: int = 112, 
                        mode: str = 'arcface', device: torch.device = None) -> torch.Tensor:
    """
    PyTorch version of estimate_norm function using SimilarityTransform.
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

    # Convert to numpy for scikit-image SimilarityTransform
    lmk_np = lmk.detach().cpu().numpy()
    dst_np = dst.detach().cpu().numpy()

    tform = trans.SimilarityTransform()
    tform.estimate(lmk_np, dst_np)
    M_np = tform.params[0:2, :]

    M = torch.tensor(M_np, dtype=torch.float32, device=device)
    return M


def norm_crop_torch(img: Union[torch.Tensor, np.ndarray], landmark: Union[torch.Tensor, np.ndarray], 
                    image_size: int = 112, mode: str = 'arcface') -> torch.Tensor:
    """
    PyTorch version of norm_crop function using grid_sample.
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

    # Create full 3x3 matrix
    M_full = torch.eye(3, device=device)
    M_full[:2, :] = M

    src_norm = torch.tensor([
        [2.0 / width, 0, -1],
        [0, 2.0 / height, -1],
        [0, 0, 1]
    ], dtype=torch.float32, device=device)

    dst_norm = torch.tensor([
        [2.0 / image_size, 0, -1],
        [0, 2.0 / image_size, -1],
        [0, 0, 1]
    ], dtype=torch.float32, device=device)

    theta = src_norm @ torch.inverse(M_full) @ torch.inverse(dst_norm)
    theta = theta[:2, :].unsqueeze(0)

    grid = F.affine_grid(theta, [batch_size, channels, image_size, image_size], align_corners=False)
    grid = grid.type_as(img)
    warped = F.grid_sample(img, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

    if need_squeeze:
        warped = warped.squeeze(0)

    return warped


def detect_face_pose(landmarks: np.ndarray, threshold: float = 0.78) -> str:
    """
    Determine if a face is front-facing or side-facing based on landmarks.
    Landmarks order: left_eye, right_eye, nose, left_mouth, right_mouth.
    """
    left_eye, right_eye = landmarks[0], landmarks[1]

    # Calculate eye distance ratio (horizontal)
    left_eye_x, right_eye_x = left_eye[0], right_eye[0]
    eye_distance = abs(right_eye_x - left_eye_x)

    # Calculate face width from bounding box of landmarks
    face_width = np.max(landmarks[:, 0]) - np.min(landmarks[:, 0])

    # Normalized eye distance ratio
    eye_distance_ratio = eye_distance / (face_width + 1e-6)

    if eye_distance_ratio > threshold:
        return "front"
    elif eye_distance_ratio > threshold * 0.5:
        return "partial_profile"
    else:
        return "profile"


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
        # Crop width to make a square
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
        # Crop height to make a square
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


# ==============================================================================
# Classes: Detection & Loss
# ==============================================================================

class DET:
    """Wrapper for InsightFace detection model."""
    def __init__(self):
        arcface_detect_path = "./models/scrfd_10g_bnkps.onnx"
        self.model_det = model_zoo.get_model(arcface_detect_path, providers=['CPUExecutionProvider'])
        self.model_det.prepare(ctx_id=0, det_thresh=0.4, input_size=(640, 640))

    def __call__(self, image: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        # Convert tensor to numpy (H, W, C)
        image_np = image.clone().detach().cpu().to(torch.float32).numpy().transpose(1, 2, 0)
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)

        bboxes, kpss = self.model_det.detect(image_np)
        return bboxes, kpss


class IDLoss:
    """
    Identity Loss calculator using ArcFace embeddings.
    Includes support for regional diffusion loss and contrastive loss.
    """
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
        """Loads embeddings from REF_CLUSTER_CENTER into a tensor pool."""
        self.negative_pool = []
        if not os.path.exists(REF_CLUSTER_CENTER):
             raise ValueError(f"Reference cluster path not found: {REF_CLUSTER_CENTER}")

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
        """Calculates Intersection over Union (IoU) between two bounding boxes."""
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


    def get_arcface_embeddings(self, images: torch.Tensor, check_side_views: bool = False, 
                                original_bboxes: Optional[List[Any]] = None) -> Tuple[List[torch.Tensor], bool]:
        """
        Get ArcFace embeddings for a batch of images.
        Handles face detection, alignment, and matching with original bounding boxes.
        Optimized to use batch inference for ArcFace.
        """
        self.netArc.eval()
        # Freeze model weights for ID loss. Gradients will still flow through the input images.
        self.netArc.requires_grad_(False)

        if len(images.shape) == 3:
            images = images.unsqueeze(0)

        batch_size = images.shape[0]
        force_align_global = False

        all_aligned_faces = []
        faces_per_image = []

        # 1. Detection and Alignment (Per image logic)
        for i in range(batch_size):
            image = images[i]
            
            # Detection (No gradients needed for bounding box regression)
            with torch.no_grad():
                bboxes, landmarks = self.netDet(image)
                
                if check_side_views:
                    poses = [detect_face_pose(lmk) for lmk in landmarks]
                    if not all(p == "front" for p in poses):
                        force_align_global = True
                    
                    if any((bbox[2] - bbox[0] < 80 or bbox[3] - bbox[1] < 80) for bbox in bboxes):
                        force_align_global = True

            # Handle landmark count mismatches
            force_align_local = False
            if len(landmarks) != 2:
                if len(landmarks) == 0:
                    landmarks = [ARCFACE_SRC, ARCFACE_SRC]
                    force_align_local = True
                elif len(landmarks) == 1:
                    landmarks = [landmarks[0], ARCFACE_SRC]
                    force_align_local = True
                elif len(landmarks) > 2:
                    landmarks = [landmarks[0], landmarks[1]]
                    force_align_local = True
            else:
                # Match detected faces to original bounding boxes using IoU
                if original_bboxes is not None and not force_align_local:
                    original_bbox = original_bboxes[i]
                    iou_0 = self._iou(original_bbox[0], bboxes[0])
                    iou_1 = self._iou(original_bbox[1], bboxes[0])

                    if iou_1 > iou_0:
                        landmarks = [landmarks[1], landmarks[0]]
                    else:
                        landmarks = [landmarks[0], landmarks[1]]

            if force_align_local:
                force_align_global = True

            # Convert landmarks to tensors
            if isinstance(landmarks, (list, tuple, np.ndarray)):
                landmarks = [torch.tensor(lmk, dtype=torch.float32) for lmk in landmarks]

            # Align faces (Gradients must flow through image)
            for landmark in landmarks:
                # Align: (H, W, C) -> (C, H, W)
                # align_face expects (H, W, C)
                aimg = align_face(image.permute(1, 2, 0), landmark).permute(2, 0, 1)
                all_aligned_faces.append(aimg)
            
            faces_per_image.append(len(landmarks))

        # 2. Batch Inference
        if not all_aligned_faces:
            return [torch.empty(0, 512, device=self.device) for _ in range(batch_size)], True

        # Stack: (N_total, 3, 112, 112)
        batch_tensor = torch.stack(all_aligned_faces)
        
        # Normalize
        batch_tensor = (batch_tensor - 127.5) / 127.5
        
        # Forward pass
        embeddings_batch = self.netArc(batch_tensor.to(dtype=self.dtype))

        # 3. Unpack results
        all_embeddings = []
        cursor = 0
        for count in faces_per_image:
            if count > 0:
                # Extract slice (count, 512)
                img_emb = embeddings_batch[cursor : cursor + count]
                all_embeddings.append(img_emb)
                cursor += count
            else:
                all_embeddings.append(torch.empty(0, 512, device=self.device))

        return all_embeddings, force_align_global


    def compute_id_loss_with_embeddings(self, generated_arcface_embeddings: List[torch.Tensor], 
                                        ground_truth_arcface_embeddings: List[torch.Tensor], 
                                        rec_bbox_A=None, rec_bbox_B=None, filter_out_side_views=False, 
                                        force_align=False, original_bboxes=None):
        """
        Compute identity loss using pre-computed embeddings.
        """
        id_losses = []

        for i in range(len(generated_arcface_embeddings)):
            gen_emb = generated_arcface_embeddings[i] # (N, 512)
            gt_emb = ground_truth_arcface_embeddings[i] # (N, 512)

            # Ensure shapes match roughly
            if gen_emb.shape[0] == 2 and gt_emb.shape[0] == 2:
                # Cosine similarity: (A, B) . (A, B)
                # We want to maximize similarity, so minimize 1 - sim
                
                cossim_AA = F.cosine_similarity(gen_emb[0], gt_emb[0], dim=0)
                cossim_BB = F.cosine_similarity(gen_emb[1], gt_emb[1], dim=0)

                if original_bboxes is None:
                    # Check for swapped faces
                    cossim_AB = F.cosine_similarity(gen_emb[0], gt_emb[1], dim=0)
                    cossim_BA = F.cosine_similarity(gen_emb[1], gt_emb[0], dim=0)

                    if cossim_AA + cossim_BB > cossim_AB + cossim_BA:
                        id_loss = 1 - (cossim_AA + cossim_BB) / 2
                    else:
                        id_loss = 1 - (cossim_AB + cossim_BA) / 2
                else:
                    id_loss = 1 - (cossim_AA + cossim_BB) / 2

            elif gen_emb.shape[0] == 1 and gt_emb.shape[0] == 1:
                cossim = F.cosine_similarity(gen_emb[0], gt_emb[0], dim=0)
                id_loss = 1 - cossim
            else:
                # Mismatch in face counts or empty
                return None, True
            
            id_losses.append(id_loss)

        if not id_losses:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        return torch.mean(torch.stack(id_losses))


    def compute_contrastive_loss(self, generated_embeddings: torch.Tensor, ground_truth_embeddings: torch.Tensor, 
                                 generated_labels=None, ground_truth_labels=None, 
                                 temperature: float = 0.07, max_negatives: int = 10) -> torch.Tensor:
        """
        Computes contrastive loss (InfoNCE style).
        """
        device = self.device
        dtype = self.dtype

        gen = generated_embeddings.to(device=device, dtype=dtype)
        gt = ground_truth_embeddings.to(device=device, dtype=dtype)

        # Reshape if 4D: (B, num_faces, extra_dim, D) -> (B, num_faces * extra_dim, D)
        if len(gen.shape) == 4:
            B, num_faces, extra_dim, D = gen.shape
            gen = gen.reshape(B, num_faces * extra_dim, D)
        elif len(gen.shape) == 3:
            B, num_faces, D = gen.shape
        else:
            raise ValueError(f"Unexpected shape for gen tensor: {gen.shape}")

        # Match ground truth shape
        if len(gt.shape) == 3 and gt.shape[1] != gen.shape[1]:
            gt = gt[:, :gen.shape[1], :]

        gen = F.normalize(gen, dim=2)
        gt = F.normalize(gt, dim=2)

        faces_per_sample = gen.shape[1]
        if faces_per_sample != 2:
            print(f"Warning: Expected 2 faces per sample, but got {faces_per_sample}. Loss designed for 2 faces.")

        # Step 1: Match identity
        # Determine best matching order for the first two faces
        sim_00 = (gen[:, 0] * gt[:, 0]).sum(dim=1)
        sim_01 = (gen[:, 0] * gt[:, 1]).sum(dim=1)
        sim_10 = (gen[:, 1] * gt[:, 0]).sum(dim=1)
        sim_11 = (gen[:, 1] * gt[:, 1]).sum(dim=1)

        keep_order = (sim_00 + sim_11 >= sim_01 + sim_10)

        # Reorder ground truth based on matching
        matched_gt = torch.empty_like(gt)
        matched_gt[:, 0] = torch.where(keep_order.view(-1, 1), gt[:, 0], gt[:, 1])
        matched_gt[:, 1] = torch.where(keep_order.view(-1, 1), gt[:, 1], gt[:, 0])
        if faces_per_sample > 2:
            matched_gt[:, 2:] = gt[:, 2:]

        # Step 2: Construct positive pairs
        anchors = gen.reshape(B * faces_per_sample, D)
        positives = matched_gt.reshape(B * faces_per_sample, D)

        # Step 3: Construct negative pool
        neg_pool = torch.cat([gen, gt], dim=1).reshape(B * 2 * faces_per_sample, D)
        neg_pool = F.normalize(neg_pool, dim=1)

        all_losses = []
        for i in range(B * faces_per_sample):
            anchor = anchors[i]
            positive = positives[i]

            # Determine range to exclude (current sample's faces) to avoid false negatives
            sample_id = i // faces_per_sample
            exclude_start = sample_id * 2 * faces_per_sample
            exclude_idx = torch.arange(exclude_start, exclude_start + 2 * faces_per_sample, device=device)

            # Select valid negative indices
            neg_indices = torch.ones(len(neg_pool), dtype=torch.bool, device=device)
            neg_indices[exclude_idx] = False
            neg_indices = torch.where(neg_indices)[0]

            # Randomly sample negatives
            if len(neg_indices) > max_negatives:
                perm = torch.randperm(len(neg_indices), device=device)[:max_negatives]
                neg_indices = neg_indices[perm]

            negatives = neg_pool[neg_indices]

            # Calculate similarities
            pos_sim = (anchor * positive).sum() / temperature
            neg_sims = (negatives @ anchor) / temperature

            # InfoNCE Loss
            numerator = torch.exp(pos_sim)
            denominator = numerator + torch.sum(torch.exp(neg_sims))
            loss = -torch.log(numerator / denominator)
            all_losses.append(loss)

        return torch.stack(all_losses).mean()

    def compute_info_nce_loss(self, generated_embeddings: torch.Tensor, ground_truth_embeddings: torch.Tensor, 
                              extend_negative_pool: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Wrapper for InfoNCE loss using the info_nce library.
        """
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

        if len(gt.shape) == 3 and gt.shape[1] != gen.shape[1]:
            gt = gt[:, :gen.shape[1], :]

        gen = F.normalize(gen, dim=2)
        gt = F.normalize(gt, dim=2)

        faces_per_sample = gen.shape[1]
        if faces_per_sample != 2:
            print(f"Warning: Expected 2 faces per sample, but got {faces_per_sample}.")

        loss_fn = InfoNCE(negative_mode='paired')

        # Construct query and positive keys
        total_samples = B * num_faces
        query = gen.reshape(total_samples, D)
        positive_keys = gt.reshape(total_samples, D)

        # Construct negative keys
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

        return loss_fn(query, positive_keys, negative_keys)

    


    def __call__(self, decoded_images: torch.Tensor, ground_truth_arcface_embeddings: List[torch.Tensor], 
                 ground_truth_image: torch.Tensor, bboxes_A: List[Any], bboxes_B: Optional[List[Any]] = None, 
                 regional_mse_weight: float = 3) -> torch.Tensor:
        """
        Main entry point: Compute ID loss + Regional Diffusion loss.
        """
        id_loss, _ = self.compute_id_loss(decoded_images, ground_truth_arcface_embeddings)
        
        if regional_mse_weight > 0:
            regional_loss = self.region_diffusion_loss(decoded_images, ground_truth_image, bboxes_A, bboxes_B=bboxes_B)
            return id_loss + regional_mse_weight * regional_loss
        
        return id_loss


# ==============================================================================
# Main Execution Block (Test)
# ==============================================================================

if __name__ == "__main__":
    import insightface
    import cv2
    
    # Test configuration
    image_paths = [
        "/data/MIBM/BenCon/benchmarks/v200/untar_cn/0bb9e7100a32c4324bbd87e38ed0a11dba8876c0.jpg",
        "/data/MIBM/BenCon/benchmarks/v200/untar_cn/0bb9e7100a32c4324bbd87e38ed0a11dba8876c0.jpg"
    ]
    
    images = []
    images_np = []
    
    for image_path in image_paths:
        img = Image.open(image_path).convert("RGB")
        json_path = image_path.replace(".jpg", ".json")
        
        if os.path.exists(json_path):
            json_dict = json.load(open(json_path))
            img_face = single_face_preserving_resize(img, json_dict["bboxes"][0])
            if img_face:
                img_face.save("debug_face.jpg")
                img_face_np = np.array(img_face)
                images.append(torch.tensor(img_face_np, dtype=torch.float32).permute(2, 0, 1))
                images_np.append(cv2.cvtColor(img_face_np, cv2.COLOR_RGB2BGR))
        else:
            print(f"JSON not found for {image_path}")

    if not images:
        print("No valid images found for testing.")
        exit()

    # Initialize models
    id_loss_model = IDLoss()
    official_model = insightface.app.FaceAnalysis(name="antelopev2", root="./models", providers=['CUDAExecutionProvider'])
    official_model.prepare(ctx_id=0, det_thresh=0.4)

    # Official InsightFace inference
    arcface_embeddings_1_official = official_model.get(images_np[0])
    arcface_embeddings_2_official = official_model.get(images_np[1])
    
    emb1_off = torch.tensor(arcface_embeddings_1_official[0].embedding, dtype=torch.float32).unsqueeze(0)
    emb2_off = torch.tensor(arcface_embeddings_2_official[0].embedding, dtype=torch.float32).unsqueeze(0)
    
    print(f"Official embeddings shapes: {emb1_off.shape}, {emb2_off.shape}")
    cos_sim = F.cosine_similarity(emb1_off, emb2_off)
    print("Cosine similarity between official embeddings:", cos_sim.item())

    # IDLoss model inference
    images_tensor = torch.stack(images).to(id_loss_model.device, dtype=torch.float32)
    print("Images shape:", images_tensor.shape)
    
    embeddings, _ = id_loss_model.get_arcface_embeddings(images_tensor)
    print("Embeddings shapes:", [emb[0].shape for emb in embeddings])
    
    cos_ours = F.cosine_similarity(embeddings[0][0], embeddings[1][0])
    print("Cosine similarity between our embeddings:", cos_ours.item())
    
    cos_AA = F.cosine_similarity(embeddings[0][0].to("cpu"), emb1_off)
    cos_BB = F.cosine_similarity(embeddings[1][0].to("cpu"), emb2_off)
    print("Cosine similarity between our embeddings and official embeddings:", cos_AA.item(), cos_BB.item())

    # Test with predicted image and ground truth
    gt_path = "/data/MIBM/UNO/tmp/ff6497adbba76f22a7fe6f377f0176a2eb2c25a3.npy"
    pred_path = "/data/MIBM/UNO/debug_output.png"
    
    if os.path.exists(gt_path) and os.path.exists(pred_path):
        gt_embeddings = np.load(gt_path, allow_pickle=True).item()["embeddings"]
        
        pred_image = Image.open(pred_path).convert("RGB")
        pred_image = np.array(pred_image)
        pred_image = torch.tensor(pred_image, dtype=torch.float32).permute(2, 0, 1)
        pred_image = pred_image.unsqueeze(0).to(id_loss_model.device, dtype=torch.float32)
        
        print("Predicted image shape:", pred_image.shape)
        
        id_loss_val, _ = id_loss_model.compute_id_loss(
            pred_image, 
            [torch.tensor(gt_embeddings, dtype=torch.float32).to(id_loss_model.device)]
        )
        if id_loss_val is not None:
            print("ID Loss:", id_loss_val.item())
        else:
            print("ID Loss calculation failed (no faces detected).")
    else:
        print("Ground truth or prediction file not found for final test.")