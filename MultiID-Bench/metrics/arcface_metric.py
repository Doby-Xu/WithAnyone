import insightface
import cv2
import numpy as np
import os
# use torch cos similarity
from torch.nn import functional as F
import scipy.optimize
import torch

from PIL import Image
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from facenet_pytorch import MTCNN, InceptionResnetV1
from AdaFace.inference import load_pretrained_model, to_input

from facexlib.detection import init_detection_model
from facexlib.utils.face_restoration_helper import FaceRestoreHelper


# import numpy as np
# import torch

def copy_geo(gen, target, ref, eps=1e-6):

    if isinstance(gen, torch.Tensor):
        gen = gen.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    if isinstance(ref, torch.Tensor):
        ref = ref.detach().cpu().numpy()
    

    def normalize(x):
        return x / (np.linalg.norm(x) + eps)
    g, t, r = normalize(gen), normalize(target), normalize(ref)
    

    def cos(a, b):
        return np.clip(np.dot(a, b), -1.0, 1.0)
    s_gt, s_gr, s_tr = cos(g, t), cos(g, r), cos(t, r)
    

    theta_gt = np.arccos(s_gt)
    theta_gr = np.arccos(s_gr)
    theta_tr = np.arccos(s_tr)
    
    if theta_tr < eps: 
        return 0.0 
    
    return float((theta_gt - theta_gr) / theta_tr)


class ArcFace_Metrics():
    def __init__(self, model_path = "./", ref_dir =  "./data/ref/npy/", device='cuda'):
        self.device = device

        self.model = insightface.app.FaceAnalysis(name = "buffalo_l", root=model_path, providers=['CUDAExecutionProvider'])
        self.model.prepare(ctx_id=0, det_thresh=0.45)

        self.face_helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=112,  
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='jpg',
            use_parse=False
        )

        self.adaface_model = load_pretrained_model('ir_50')
        self.facenet_model = InceptionResnetV1(pretrained='vggface2').eval()

        
        self.cos = F.cosine_similarity
        self.ref_dir = ref_dir
    
    
    
    def get_embeddings_adaface(self, img):
        self.face_helper.clean_all()
        if isinstance(img, Image.Image):
            # to cv2 bgr np
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        self.face_helper.read_image(img)
        self.face_helper.get_face_landmarks_5(only_center_face=False) 
        self.face_helper.align_warp_face()

        aligned_faces = self.face_helper.cropped_faces
        embeddings = []
        for i, aligned_face in enumerate(aligned_faces):

            aligned_rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)

            bgr_input = to_input(aligned_rgb)

            feature, _ = self.adaface_model(bgr_input)
            
            embeddings.append(feature.detach().cpu())
        if len(embeddings) == 0:
            return None

        return torch.stack(embeddings).squeeze(1)  

    def get_embeddings_facenet(self, img):
        self.face_helper.clean_all()
        if isinstance(img, Image.Image):
            # to cv2 bgr np
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        self.face_helper.read_image(img)
        self.face_helper.get_face_landmarks_5(only_center_face=False)  
        self.face_helper.align_warp_face()
        aligned_faces = self.face_helper.cropped_faces
        embeddings = []
        for i, aligned_face in enumerate(aligned_faces):
            aligned_face = torch.tensor(aligned_face).float() / 255.0  # Normalize to [0, 1]
            aligned_face = aligned_face.permute(2, 0, 1)  # Change to (C, H, W) format
            image_embedding = self.facenet_model(aligned_face.unsqueeze(0))
            embeddings.append(image_embedding.detach().cpu())
        if len(embeddings) == 0:
            return None

        return torch.stack(embeddings).squeeze(1) 
    def get_embeddings(self, img, model = "None"):
        if model == "adaface":
            return self.get_embeddings_adaface(img)
        elif model == "facenet":
            # raise NotImplementedError("Facenet model is not implemented yet")
            return self.get_embeddings_facenet(img)
        elif model == "arcface":
            if not isinstance(img, np.ndarray):
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            res = self.model.get(img)
            if len(res) == 0:
                return None
            # return res.embedding
            # return torch.stack([r.embedding for r in res])
            # [r.embedding for r in res] is list of np.array, need to convert to torch.tensor
            return torch.stack([torch.from_numpy(r.embedding) for r in res])
        else:
            raise ValueError("Unknown model type: {}".format(model))
            
    def get_multi_img_embeddings_adaface(self, imgs):
        embeddings = []
        for img in imgs:
            embedding = self.get_embeddings_adaface(img)
            if embedding is not None:
                embeddings.append(embedding)
        return torch.stack(embeddings).squeeze(1)  
    def get_multi_img_embeddings_facenet(self, imgs):
        embeddings = []
        for img in imgs:
            embedding = self.get_embeddings_facenet(img)
            if embedding is not None:
                embeddings.append(embedding)
        return torch.stack(embeddings).squeeze(1)
    
    def get_multi_img_embeddings(self, imgs, model = "None"):
        if model == "adaface":
            return self.get_multi_img_embeddings_adaface(imgs)
        elif model == "facenet":
            # raise NotImplementedError("Facenet model is not implemented yet")
            return self.get_multi_img_embeddings_facenet(imgs)
            
        elif model == "arcface":
            
            results = []
            for img in imgs:
                if not isinstance(img, np.ndarray):
                    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    # img = np.array(img)
                res = self.model.get(img)
                if len(res) == 0:
                    pass
                else:
                    results.append(res)
            # convert them to torch.tensor
            # if any of the images have no faces, return None
            if len(results) != len(imgs):
                print("Warning: Some images have no faces detected")
                return None
            
        

            return torch.stack([torch.from_numpy(r[0].embedding) for r in results]) 
        else:
            raise ValueError("Unknown model type: {}".format(model))

    
    def compare_mutli_person_on_2_images(self, img1, img2, model="None"):
        
        embs_1 = self.get_embeddings(img1, model=model)
        embs_2 = self.get_embeddings(img2, model=model)

        # if either is None, return None
        if embs_1 is None or embs_2 is None:
            print("Warning: No faces detected in one or both images")
            if embs_1 is None:
                print("No faces detected in image 1")
            if embs_2 is None:
                print("No faces detected in image 2")
            return None

        if embs_1.shape[0] != embs_2.shape[0]:
            # raise ValueError('The number of faces in two images are not the same')
            print("Warning: The number of faces in two images are not the same")
            return None
        
        sim_matrix =  self.cos(embs_1.unsqueeze(1), embs_2.unsqueeze(0), dim=2)


        sim_matrix_np = sim_matrix.detach().cpu().numpy()
        cost_matrix = -sim_matrix_np  
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)

   
        row_ind = torch.from_numpy(row_ind).to(sim_matrix.device)
        col_ind = torch.from_numpy(col_ind).to(sim_matrix.device)
        selected_scores = sim_matrix[row_ind, col_ind]

        return selected_scores
    
    def compare_mutli_person_on_2_images_with_ref(self, img, refs, model="None"):
        
        embs_1 = self.get_embeddings(img, model=model)
        embs_2 = self.get_multi_img_embeddings(refs, model=model)
        # print("embs_1.shape: ", embs_1.shape)
        # print("embs_2.shape: ", embs_2.shape)
        if embs_1.shape[0] != embs_2.shape[0]:
            # raise ValueError('The number of faces in two images are not the same')
            print("Warning: The number of faces in two images are not the same")
            return None

        sim_matrix =  self.cos(embs_1.unsqueeze(1), embs_2.unsqueeze(0), dim=2)
        sim_matrix_np = sim_matrix.detach().cpu().numpy()
        cost_matrix = -sim_matrix_np  
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)


        row_ind = torch.from_numpy(row_ind).to(sim_matrix.device)
        col_ind = torch.from_numpy(col_ind).to(sim_matrix.device)
        selected_scores = sim_matrix[row_ind, col_ind]

        return selected_scores
    
    def calculate_copy_geo_multi_person(self, gen_img, target_img, ref_imgs, model="arcface"):
        """
        Calculate the copy_geo metric for multiple people across generated, target, and reference images.
        
        Args:
            gen_img: The generated image
            target_img: The target (original) image
            ref_imgs: A list of reference images, each containing one person
            model: The face embedding model to use ("arcface", "adaface", or "facenet")
            
        Returns:
            A tensor of copy_geo scores for each matched person, or None if face counts don't match
        """
        # Get embeddings for all images
        gen_embs = self.get_embeddings(gen_img, model=model)
        target_embs = self.get_embeddings(target_img, model=model)
        ref_embs = self.get_multi_img_embeddings(ref_imgs, model=model)
        
        # Check if we have valid embeddings
        if gen_embs is None or target_embs is None or ref_embs is None:
            print("Warning: Could not detect faces in one or more images")
            return None
        
        # Check if counts match
        if gen_embs.shape[0] != target_embs.shape[0] or gen_embs.shape[0] != ref_embs.shape[0]:
            print("Warning: The number of faces doesn't match between images")
            print(f"Generated image: {gen_embs.shape[0]} faces")
            print(f"Target image: {target_embs.shape[0]} faces")
            print(f"Reference images: {ref_embs.shape[0]} faces")
            return None
        
        # Match faces between generated and target images
        gen_target_sim = self.cos(gen_embs.unsqueeze(1), target_embs.unsqueeze(0), dim=2)
        gen_target_cost = -gen_target_sim.detach().cpu().numpy()  # Convert to cost for minimization
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(gen_target_cost)
        
        # Match faces between generated and reference images
        gen_ref_sim = self.cos(gen_embs.unsqueeze(1), ref_embs.unsqueeze(0), dim=2)
        gen_ref_cost = -gen_ref_sim.detach().cpu().numpy()  # Convert to cost for minimization
        row_ind_gr, col_ind_gr = scipy.optimize.linear_sum_assignment(gen_ref_cost)
        
        # Match faces between target and reference images
        target_ref_sim = self.cos(target_embs.unsqueeze(1), ref_embs.unsqueeze(0), dim=2)
        target_ref_cost = -target_ref_sim.detach().cpu().numpy()  # Convert to cost for minimization
        row_ind_tr, col_ind_tr = scipy.optimize.linear_sum_assignment(target_ref_cost)
        
        # Calculate copy_geo for each matched face
        copy_geo_scores = []
        for i in range(len(row_ind)):
            gen_idx = row_ind[i]
            target_idx = col_ind[i]
            # Find the corresponding reference face for this pair
            ref_idx = col_ind_gr[np.where(row_ind_gr == gen_idx)[0][0]]
            
            # Get the embeddings for the matched triplet
            gen_emb = gen_embs[gen_idx]
            target_emb = target_embs[target_idx]
            ref_emb = ref_embs[ref_idx]
            
            # Calculate copy_geo for this triplet
            score = copy_geo(gen_emb, target_emb, ref_emb)
            copy_geo_scores.append(score)
        
        return torch.tensor(copy_geo_scores)
    def compare_faces_with_confusion_matrix(self, img1, img2, model="None"):
        """
        Compare faces between two images and return:
        1. The full similarity matrix (confusion matrix)
        2. The list of scores for matched face pairs (diagonal elements after optimal assignment)
        3. The mean score of non-matched face pairs (non-diagonal elements)
        """
        # Get embeddings for faces in both images
        embs_1 = self.get_embeddings(img1, model=model)
        embs_2 = self.get_embeddings(img2, model=model)
        
        if embs_1 is None or embs_2 is None:
            print("Warning: No faces detected in one or both images")
            if embs_1 is None:
                print("No faces detected in image 1")
            if embs_2 is None:
                print("No faces detected in image 2")
            return None, None, None
        
        # Calculate similarity matrix (cosine similarity between all pairs of embeddings)
        sim_matrix = self.cos(embs_1.unsqueeze(1), embs_2.unsqueeze(0), dim=2)
        sim_matrix_np = sim_matrix.detach().cpu().numpy()
        
        # Use Hungarian algorithm to find optimal matching
        cost_matrix = -sim_matrix_np  # Convert to minimization problem (maximize similarity)
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
        
        # Extract matched scores (diagonal elements after optimal assignment)
        matched_scores = sim_matrix_np[row_ind, col_ind]
        
        # Create mask to identify non-matched pairs
        mask = np.ones_like(sim_matrix_np, dtype=bool)
        for r, c in zip(row_ind, col_ind):
            mask[r, c] = False
        
        # Calculate mean of non-matched scores (non-diagonal elements)
        non_matched_mean = np.mean(sim_matrix_np[mask]) if np.any(mask) else None
        
        return sim_matrix_np, matched_scores, non_matched_mean
    
    def compare_faces_with_confusion_matrix_with_ref(self, img, refs, model="None"):
        """
        Compare faces between an image and multiple reference images and return:
        1. The full similarity matrix
        2. The list of scores for matched face pairs
        3. The mean score of non-matched face pairs
        
        Each reference image should contain one face.
        """
        embs_1 = self.get_embeddings(img, model=model)
        embs_2 = self.get_multi_img_embeddings(refs,    model=model)
        
        if embs_1 is None or embs_2 is None:
            print("Warning: No faces detected in image or references")
            if embs_1 is None:
                print("No faces detected in image")
            if embs_2 is None:
                print("No faces detected in references")
            return None, None, None
        
        # Calculate similarity matrix
        sim_matrix = self.cos(embs_1.unsqueeze(1), embs_2.unsqueeze(0), dim=2)
        sim_matrix_np = sim_matrix.detach().cpu().numpy()
        
        # Use Hungarian algorithm to find optimal matching
        cost_matrix = -sim_matrix_np  # Convert to minimization problem
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
        
        # Extract matched scores
        matched_scores = sim_matrix_np[row_ind, col_ind]
        
        # Create mask to identify non-matched pairs
        mask = np.ones_like(sim_matrix_np, dtype=bool)
        for r, c in zip(row_ind, col_ind):
            mask[r, c] = False
        
        # Calculate mean of non-matched scores
        non_matched_mean = np.mean(sim_matrix_np[mask]) if np.any(mask) else None
        
        return sim_matrix_np, matched_scores, non_matched_mean
    
    def compare_faces_with_confusion_matrix_with_ref_embeddings(self, img, ref_embeddings, model="None"):
        """
        Compare faces between an image and multiple reference embeddings and return:
        1. The full similarity matrix
        2. The list of scores for matched face pairs
        3. The mean score of non-matched face pairs
        
        Each reference embedding should correspond to one face.
        """
        embs_1 = self.get_embeddings(img, model=model)
        
        if embs_1 is None or ref_embeddings is None:
            print("Warning: No faces detected in image or references")
            return None, None, None
        
        # Calculate similarity matrix
        sim_matrix = self.cos(embs_1.unsqueeze(1), ref_embeddings.unsqueeze(0), dim=2)
        sim_matrix_np = sim_matrix.detach().cpu().numpy()
        
        # Use Hungarian algorithm to find optimal matching
        cost_matrix = -sim_matrix_np
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
        # Extract matched scores
        matched_scores = sim_matrix_np[row_ind, col_ind]
        # Create mask to identify non-matched pairs
        mask = np.ones_like(sim_matrix_np, dtype=bool)
        for r, c in zip(row_ind, col_ind):
            mask[r, c] = False
        # Calculate mean of non-matched scores
        non_matched_mean = np.mean(sim_matrix_np[mask]) if np.any(mask) else None
        return sim_matrix_np, matched_scores, non_matched_mean
    
    def compare_faces_with_confusion_matrix_with_ref_names(self, img, names, model="None"):
        ref_embeddings = self.retrive_cluster_center(names)
        return self.compare_faces_with_confusion_matrix_with_ref_embeddings(img, ref_embeddings, model=model)
    
    def retrive_cluster_center(self, names):
        num_name = len(names)
        cluster_center = []
        for i in range(num_name):
            name = names[i]
            npy = np.load(os.path.join(self.ref_dir, name + ".npy"))


            cluster_center.append(npy)
        cluster_center = np.array(cluster_center)
        return torch.from_numpy(cluster_center)






    
