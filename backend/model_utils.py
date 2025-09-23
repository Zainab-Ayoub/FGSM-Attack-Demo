from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_pretrained_model(num_classes: int | None = None) -> nn.Module:
    """Load a lightweight pretrained model for demo purposes.

    Uses torchvision's resnet18 pretrained on ImageNet. For MNIST-like inputs, we still
    use the same backbone but input must be 3x224x224 normalized to ImageNet stats.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()
    return model


def imagenet_preprocess() -> transforms.Compose:
    weights = models.ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()
    return preprocess


def imagenet_categories() -> list[str]:
    weights = models.ResNet18_Weights.DEFAULT
    cats: list[str] = list(weights.meta.get("categories", []))
    return cats


def pil_to_tensor_for_model(img: Image.Image) -> torch.Tensor:
    preprocess = imagenet_preprocess()
    tensor = preprocess(img).unsqueeze(0)
    return tensor


def denormalize_to_display(t: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=t.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=t.device).view(1, 3, 1, 1)
    x = t * std + mean
    return torch.clamp(x, 0.0, 1.0)


def top1_label_from_logits(logits: torch.Tensor) -> int:
    return int(torch.argmax(logits, dim=1).item())


def imagenet_norm_stats(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return mean, std


def normalized_bounds_for_imagenet(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    mean, std = imagenet_norm_stats(device)
    min_t = (0.0 - mean) / std
    max_t = (1.0 - mean) / std
    return min_t, max_t