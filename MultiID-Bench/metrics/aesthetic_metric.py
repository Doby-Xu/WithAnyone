import json
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
from typing import List, Tuple, Any
from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip


class Aesthetic_Metric:
    def __init__(self, rank = 0, device = "cuda") -> None:
        self.rank = rank
        self.device = torch.device(device)

        # load model and preprocessor using new API
        model, preprocessor = convert_v2_5_from_siglip(
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        # move model to target device with bfloat16 as in example
        model = model.to(torch.bfloat16).to(self.device)
        self.model = model
        self.preprocessor = preprocessor

        # keep the old lightweight preprocessors for compatibility with existing code paths
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((378,378)),
            transforms.ToTensor(),          
        ])
        self.preprocess_pil = transforms.Compose([
            transforms.Resize((378,378)),
            transforms.ToTensor(),          
        ])

    @torch.inference_mode()
    def inference_aesthetic_predictor_v2_5(self, image: torch.Tensor) -> float:
        """
        Expecting image to be a torch.Tensor of shape [B,3,H,W] already normalized similar to preprocessor output.
        This will cast to bfloat16 and move to device, then call the new model API and return a single float score
        for the batch (squeezed). If batch size >1, returns the first element to keep previous behavior.
        """
        # ensure tensor on correct device and dtype
        image = image.to(self.device, dtype=torch.bfloat16)
        output = self.model(image)
        # logits -> squeeze -> cpu float
        logits = output.logits.squeeze()
        # If batch, take first element to preserve original behavior that returned a single score
        if logits.dim() > 0:
            score = logits[0].float().cpu().item()
        else:
            score = logits.float().cpu().item()
        return score

    def inference_pil(self, image: Image.Image) -> float:
        # use provided preprocessor to get pixel_values tensor
        pixel_values = (
            self.preprocessor(images=image, return_tensors="pt")
            .pixel_values.to(torch.bfloat16)
            .to(self.device)
        )
        score = self.inference_aesthetic_predictor_v2_5(pixel_values)
        return score

    def __call__(self, image: Any) -> list[float]:
        if isinstance(image, torch.Tensor):
            return self.inference_aesthetic_predictor_v2_5(image)
        elif isinstance(image, Image.Image):
            return self.inference_pil(image)
        else:
            raise TypeError("Input must be a torch.Tensor or PIL.Image")

if __name__ == "__main__":
    # Example usage
    
    aesthetic_metric = Aesthetic_Metric(rank=0, device="cuda")
    image = torch.randn(1, 3, 378, 378).to("cuda")  # Example tensor
    scores = aesthetic_metric(image)
    print(scores)