import torch
import torch.nn as nn
from torchvision import transforms

from .cnn_encoder import mscan_model_registry
from .modelling.mask_decoder import MaskDecoder
from .vit_encoder import build_vit_encoder


class ASPS(nn.Module):
    def __init__(self, vit_model, vit_pretrained, cnn_model, cnn_pretrained, cnn_image_size):
        super().__init__()

        self.vit = build_vit_encoder(vit_model, vit_pretrained)
        self.cnn = mscan_model_registry[cnn_model](checkpoint=cnn_pretrained)

        self.cnn_preprocess = transforms.Resize(cnn_image_size)

        if cnn_model == "tiny":
            cnn_out_dim = 256
        else:
            cnn_out_dim = 512

        self.mask_decoder = MaskDecoder(
            model_type=vit_model, transformer_dim=256, cnn_dim=cnn_out_dim
        )

    def forward(self, input, multimask_output=False):
        image_embeddings, interm_embeddings = self.vit(input)

        resized_input = torch.stack([self.cnn_preprocess(image) for image in input])

        _, _, H, W = resized_input.shape
        cnn_embeddings = self.cnn.get_blocks(resized_input, H, W)
        cnn_feature = cnn_embeddings[4]

        masks, iou_pred = self.mask_decoder(
            cnn_feature=cnn_feature,
            image_embeddings=image_embeddings,
            interm_embeddings=interm_embeddings,
            multimask_output=multimask_output,
        )

        uncertainty_p = 1 - torch.sigmoid(torch.abs(masks))
        return torch.sigmoid(masks), torch.sigmoid(iou_pred), uncertainty_p
