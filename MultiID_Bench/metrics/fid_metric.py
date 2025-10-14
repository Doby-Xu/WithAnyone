
from PIL import Image
import os
import json
import numpy as np
import torch
from torchvision import transforms
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
try:
    from pytorch_fid.inception import InceptionV3
except ImportError:
    # If you don't have pytorch_fid installed, you can pip install it
    print("Installing pytorch-fid...")
    os.system("pip install pytorch-fid")
    from pytorch_fid.inception import InceptionV3


class FID_Metric:
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Load InceptionV3 model
        self.dims = 2048
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[self.dims]
        self.model = InceptionV3([block_idx]).to(self.device)
        self.model.eval()
        
        # Define image transformation
        self.preprocess = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def get_activations(self, images):
        """Calculate activations of Inception model for a list of images."""
        batch_size = 32
        all_features = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_tensor = torch.stack([self.preprocess(img) for img in batch_images]).to(self.device)
            
            with torch.no_grad():
                pred = self.model(batch_tensor)[0]
            
            # If model output is not pooled, use global average pooling
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
                
            features = pred.squeeze().cpu().numpy()
            all_features.append(features)
            
        return np.concatenate(all_features, axis=0)
    
    def calculate_statistics(self, activations):
        """Calculate mean and covariance matrix of activations."""
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        return mu, sigma
    
    def calculate_fid(self, real_images, generated_images):
        """Calculate FID between two sets of images."""
        # Get activations
        real_activations = self.get_activations(real_images)
        gen_activations = self.get_activations(generated_images)
        
        # Calculate statistics
        mu1, sigma1 = self.calculate_statistics(real_activations)
        mu2, sigma2 = self.calculate_statistics(gen_activations)
        
        # Calculate FID
        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        covmean = linalg.sqrtm(sigma1.dot(sigma2))
        
        # Numerical stability check
        if np.iscomplexobj(covmean):
            covmean = covmean.real
            
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid
