from .efficient_sam import build_efficient_sam_vitt
from .segment_anything import sam_model_registry


def build_vit_encoder(vit_name, checkpoint):
    assert vit_name in [
        "vit_b",
        "vit_h",
        "efficient_sam_vitt",
    ], f"Unknown vit model {vit_name}"

    if vit_name == "vit_b":
        model = sam_model_registry["vit_b"](checkpoint=checkpoint)
    elif vit_name == "vit_h":
        model = sam_model_registry["vit_h"](checkpoint=checkpoint)
    elif vit_name == "efficient_sam_vitt":
        model = build_efficient_sam_vitt(checkpoint=checkpoint)
    model.eval()

    # Freeze the model
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the normalization layer
    for name, param in model.named_parameters():
        if "image_encoder.neck.3" in name or "image_encoder.neck.1" in name:
            param.requires_grad = True

    return model.image_encoder
