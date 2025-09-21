from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass
class AttackResult:
    adversarial_image: torch.Tensor
    clean_pred: int
    adv_pred: int
    success: bool


class Attack:
    """Fast Gradient Sign Method (FGSM) attack implementation.

    References:
      - Goodfellow et al. (2015): Explaining and Harnessing Adversarial Examples
    """

    def __init__(self, model: nn.Module, epsilon: float = 0.1, clamp_min: float | torch.Tensor = 0.0, clamp_max: float | torch.Tensor = 1.0):
        self.model = model
        self.epsilon = float(epsilon)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.model.eval()

    @torch.no_grad()
    def predict_label(self, logits: torch.Tensor) -> int:
        return int(torch.argmax(logits, dim=1).item())

    def run(self, image: torch.Tensor, label: torch.Tensor | None = None) -> AttackResult:
        """Run FGSM attack on a single image tensor already in the model's input space.

        Notes:
            - The input is expected to be preprocessed (e.g., normalized) exactly as the model expects.
            - Clamping is performed using `clamp_min`/`clamp_max`, which should match bounds in that
              preprocessed space (e.g., ImageNet-normalized min/max), not raw [0,1] unless applicable.

        Args:
            image: Tensor of shape (1, C, H, W) with requires_grad False.
            label: Optional ground-truth label tensor of shape (1,), if None uses clean prediction.

        Returns:
            AttackResult with adversarial image and predictions.
        """

        if image.ndim != 4 or image.size(0) != 1:
            raise ValueError("Expected image of shape (1, C, H, W)")

        image = image.clone().detach().requires_grad_(True)

        # Forward pass on clean image
        logits = self.model(image)
        clean_pred = self.predict_label(logits)

        if label is None:
            label_tensor = torch.tensor([clean_pred], device=image.device)
        else:
            label_tensor = label.to(image.device)

        loss = nn.CrossEntropyLoss()(logits, label_tensor)
        self.model.zero_grad(set_to_none=True)
        loss.backward()

        # FGSM: x_adv = x + eps * sign(grad_x)
        grad_sign = image.grad.detach().sign()
        adv_image = image.detach() + self.epsilon * grad_sign
        adv_image = torch.clamp(adv_image, self.clamp_min, self.clamp_max)

        with torch.no_grad():
            adv_logits = self.model(adv_image)
        adv_pred = self.predict_label(adv_logits)

        success = adv_pred != clean_pred

        return AttackResult(adversarial_image=adv_image.detach(), clean_pred=clean_pred, adv_pred=adv_pred, success=success)


