# general_torch_model.py

import torch
import numpy as np
import torch.nn as nn
from typing import Optional, Union, List


class GeneralTorchModel(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        n_class: int = 10,
        im_mean: Optional[List[float]] = None,
        im_std: Optional[List[float]] = None
    ):
        super(GeneralTorchModel, self).__init__()
        self.model = model.eval()
        self.num_queries = 0
        self.im_mean = im_mean
        self.im_std = im_std
        self.n_class = n_class

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = self.preprocess(image)
        return self.model(image)

    def predict_prob(self, image: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            # 1) NumPy → Tensor
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image).float()
            # 2) Validate
            if not torch.is_tensor(image):
                raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(image)}")
            # 3) Batch dim
            if image.dim() == 3:
                image = image.unsqueeze(0)
            # 4) Device
            image = image.to(next(self.model.parameters()).device)
            # 5) Normalize & infer
            image = self.preprocess(image)
            logits = self.model(image)
            self.num_queries += image.size(0)
            return logits

    def predict_label(self, image: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        logits = self.predict_prob(image)
        return logits.argmax(dim=1)

    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        processed = image.float()
        if self.im_mean is not None and self.im_std is not None:
            mean = torch.tensor(self.im_mean, device=processed.device).view(1, -1, 1, 1)
            std = torch.tensor(self.im_std, device=processed.device).view(1, -1, 1, 1)
            processed = (processed - mean) / std
        return processed
