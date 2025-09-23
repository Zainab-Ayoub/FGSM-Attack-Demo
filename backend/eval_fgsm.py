from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Tuple

from PIL import Image

from .fgsm import Attack
from .model_utils import (
    load_pretrained_model,
    pil_to_tensor_for_model,
    get_device,
    normalized_bounds_for_imagenet,
)


def evaluate_on_sample_images(image_paths: List[Path], epsilons: List[float]) -> None:
    device = get_device()
    model = load_pretrained_model().to(device).eval()
    clamp_min, clamp_max = normalized_bounds_for_imagenet(device)

    out_csv = Path("backend/results_fgsm.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "epsilon", "clean_pred", "adv_pred", "success"])

        for img_path in image_paths:
            pil = Image.open(img_path).convert("RGB")
            x = pil_to_tensor_for_model(pil).to(device)
            for eps in epsilons:
                attacker = Attack(
                    model,
                    epsilon=eps,
                    clamp_min=clamp_min,
                    clamp_max=clamp_max,
                )
                result = attacker.run(x)
                writer.writerow([str(img_path), eps, result.clean_pred, result.adv_pred, int(result.success)])


if __name__ == "__main__":
    samples_dir = Path("samples")
    paths = []
    if samples_dir.exists():
        for p in samples_dir.iterdir():
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                paths.append(p)

    if not paths:
        print("No sample images found in ./samples. Place a few images to evaluate.")
    epsilons = [0.0, 0.01, 0.05, 0.1, 0.2]
    evaluate_on_sample_images(paths, epsilons)
    print("Saved results to backend/results_fgsm.csv")