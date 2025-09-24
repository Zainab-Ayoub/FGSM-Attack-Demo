from __future__ import annotations

import base64 # to turn binary data into text and back
import io # to treat bytes in memory as files
from typing import Any, Dict

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware # to allow requests from different origins like frontend
from fastapi.responses import JSONResponse
from PIL import Image # to read the uploaded image and prep it for the model
import torch # to do the math

from .fgsm import Attack

from .model_utils import (
    load_pretrained_model,
    pil_to_tensor_for_model,
    denormalize_to_display,
    get_device,
    normalized_bounds_for_imagenet,
    imagenet_categories,
)


app = FastAPI(title="FGSM Attack API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _load_model() -> None:
    global MODEL, DEVICE, CATEGORIES
    DEVICE = get_device()
    MODEL = load_pretrained_model().to(DEVICE).eval()
    CATEGORIES = imagenet_categories()


def _tensor_to_b64_img(t: torch.Tensor) -> str:
    # Expect (1, 3, H, W) in [0,1]
    t = t.squeeze(0).detach().cpu().clamp(0, 1)
    img = Image.fromarray((t.permute(1, 2, 0).numpy() * 255).astype("uint8"))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

@app.post("/attack")
async def attack(image: UploadFile = File(...), epsilon: float = Form(0.1)) -> JSONResponse:
    if image.content_type not in {"image/png", "image/jpeg"}:
        raise HTTPException(status_code=415, detail="Unsupported file type. Please upload PNG or JPEG.")

    data = await image.read()
    try:
        pil_img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image file. Please upload a valid PNG or JPEG.") from exc

    x = pil_to_tensor_for_model(pil_img).to(DEVICE)

    clamp_min, clamp_max = normalized_bounds_for_imagenet(DEVICE)
    attacker = Attack(MODEL, epsilon=float(epsilon), clamp_min=clamp_min, clamp_max=clamp_max)
    result = attacker.run(x)

    clean_disp = denormalize_to_display(x)
    adv_disp = denormalize_to_display(result.adversarial_image)
    adv_b64 = _tensor_to_b64_img(adv_disp)
    clean_b64 = _tensor_to_b64_img(clean_disp)

    response: Dict[str, Any] = {
        "clean_prediction": result.clean_pred,
        "adversarial_prediction": result.adv_pred,
        "clean_label": (CATEGORIES[result.clean_pred] if CATEGORIES and result.clean_pred < len(CATEGORIES) else None),
        "adversarial_label": (CATEGORIES[result.adv_pred] if CATEGORIES and result.adv_pred < len(CATEGORIES) else None),
        "attack_success": result.success,
        "adversarial_image_base64": adv_b64,
        "clean_image_base64": clean_b64,
        "epsilon": float(epsilon),
    }
    return JSONResponse(response)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)