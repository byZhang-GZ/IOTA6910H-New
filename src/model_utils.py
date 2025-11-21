from typing import Optional

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def build_model(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    # replace classification head for CIFAR-10
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def get_device(preferred: Optional[str] = None) -> torch.device:
    if preferred:
        preferred = preferred.lower()
        if preferred == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if preferred == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if preferred in {"cpu", "cuda", "mps"}:
            return torch.device(preferred)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
