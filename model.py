# model.py
"""
Model definitions and loading for MobileNetV2.

- get_mobilenet_v2(num_classes, pretrained): returns a MobileNetV2 with the
classifier head replaced for the requested number of classes.
- load_model(weights_path, num_classes, device): instantiates the model then
loads a saved state_dict (weights) and switches to eval mode on the target device.
"""

import torch
import torch.nn as nn
from torchvision import models

def get_mobilenet_v2(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Create a MobileNetV2 and adapt final classifier to `num_classes`.

    Args:
        num_classes: Number of categories in your dataset.
        pretrained: If True, start from ImageNet weights for faster/better convergence.
    """
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
    in_feats = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_feats, num_classes)
    return model

def load_model(weights_path: str, num_classes: int, device: str = "cpu") -> nn.Module:
    """Load weights into a MobileNetV2 and return an eval‑mode model on `device`."""
    model = get_mobilenet_v2(num_classes=num_classes, pretrained=False)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model
