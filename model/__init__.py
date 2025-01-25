import torch
import torch.nn as nn
from torchvision.transforms import Resize

from .efficient_sam import build_efficient_sam_vitt
from .mask_decoder import MaskDecoder
from .mscan import build_mscan
from .segment_anything import sam_model_registry


class ASPS(nn.Module):
    def __init__(self, vit_settings, cnn_settings):
        super().__init__()

        # Build the ViT
        if vit_settings.name == "vit_b":
            self.vit = sam_model_registry["vit_b"](vit_settings.checkpoint)
        elif vit_settings.name == "vit_l":
            self.vit = sam_model_registry["vit_l"](vit_settings.checkpoint)
        elif vit_settings.name == "efficient_sam_vitt":
            self.vit = build_efficient_sam_vitt(vit_settings.checkpoint)
        else:
            raise ValueError("Unknown ViT model")

        for name, param in self.vit.named_parameters():
            if name not in ["image_encoder.neck.3", "image_encoder.neck.1"]:
                param.requires_grad = False
        self.vit = self.vit.eval()
        self.vit = self.vit.image_encoder

        # Build the CNN
        self.cnn = build_mscan(cnn_settings)

        if cnn_settings.model == "tiny":
            mscan_dim = 256
        else:
            mscan_dim = 512

        self.mask_decoder = MaskDecoder(
            model_type=vit_settings.name, transformer_dim=256, cnn_dim=mscan_dim
        )
        self.cnn_image_size = cnn_settings.image_size_cnn

    def forward(self, images, multimask_output=False):
        vit_embeddings, interm_embeddings = self.vit(images)

        scaled_images = Resize((self.cnn_image_size, self.cnn_image_size), antialias=True)(images)
        cnn_feature = self.cnn(scaled_images)[4]

        masks, iou_pred = self.mask_decoder(
            cnn_feature=cnn_feature,
            image_embeddings=vit_embeddings,
            interm_embeddings=interm_embeddings,
            multimask_output=multimask_output,
        )

        uncertainty_p = 1 - torch.sigmoid(torch.abs(masks))
        return torch.sigmoid(masks), torch.sigmoid(iou_pred), uncertainty_p
