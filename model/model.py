import torch
import torch.nn as nn
from torch import Tensor
from functools import partial
import torchvision.transforms as transforms
from typing import Optional, List, Dict, Any
from segment_anything.modeling.sam import Sam
from segment_anything.modeling.image_encoder import ImageEncoderViT
from segment_anything.modeling.prompt_encoder import PromptEncoder
from segment_anything.modeling.mask_decoder import MaskDecoder
from segment_anything.modeling.transformer import TwoWayTransformer

class ViTSettings:
    model_type:                         str
    encoder_embedding_dimension:        int
    encoder_depth:                      int
    encoder_number_of_attention_heads:  int
    encoder_global_attention_indexes:   List[int]
    checkpoint:                         Optional[str]

    def __init__(
        self,
        model_type:                         str,
        encoder_embedding_dimension:        int,
        encoder_depth:                      int,
        encoder_number_of_attention_heads:  int,
        encoder_global_attention_indexes:   List[int],
        checkpoint:                         Optional[str]
    ) -> None:
        self.model_type                         = model_type
        self.encoder_embedding_dimension        = encoder_embedding_dimension
        self.encoder_depth                      = encoder_depth
        self.encoder_number_of_attention_heads  = encoder_number_of_attention_heads
        self.encoder_global_attention_indexes   = encoder_global_attention_indexes
        self.checkpoint                         = checkpoint

class CNNSettings:
    mscan_dimension:    int 
    image_size:         int

    def __init__(
        self,
        mscan_dimension:    int,
        image_size:         int
    ) -> None:
        self.mscan_dimension    =   mscan_dimension
        self.image_size         =   image_size

class ASPS(nn.Module):

    def __init__(
        self, 
        vit_settings: ViTSettings,
        cnn_settings: CNNSettings
    ) -> None:
        super().__init__()

        if vit_settings is not None:
            prompt_embed_dim        = 256
            image_size              = 1024
            vit_patch_size          = 16
            image_embedding_size    = image_size // vit_patch_size
            sam = Sam(
                image_encoder   =   ImageEncoderViT(
                    depth               =   vit_settings.encoder_depth,
                    embed_dim           =   vit_settings.encoder_embedding_dimension,
                    img_size            =   image_size,
                    mlp_ratio           =   4,
                    norm_layer          =   partial(torch.nn.LayerNorm, eps=1E-6),
                    num_heads           =   vit_settings.encoder_number_of_attention_heads,
                    patch_size          =   vit_patch_size,
                    qkv_bias            =   True,
                    use_rel_pos         =   True,
                    global_attn_indexes =   vit_settings.encoder_global_attention_indexes,
                    window_size         =   14,
                    out_chans           =   prompt_embed_dim,
                ),

                prompt_encoder  =   PromptEncoder(
                    embed_dim               =   prompt_embed_dim,
                    image_embedding_size    =   (image_embedding_size, image_embedding_size),
                    input_image_size        =   (image_size, image_size),
                    mask_in_chans           =   16,
                ),

                mask_decoder    =   MaskDecoder(
                    num_multimask_outputs   =   3,
                    transformer             =   TwoWayTransformer(
                        depth           =   2,
                        embedding_dim   =   prompt_embed_dim,
                        mlp_dim         =   2048,
                        num_heads       =   8,
                    ),
                    transformer_dim         =   prompt_embed_dim,
                    iou_head_depth          =   3,
                    iou_head_hidden_dim     =   256,
                ),
                pixel_mean      =   [123.675, 116.28, 103.53],
                pixel_std       =   [58.395, 57.12, 57.375],
            )
            sam.eval()
            
            if vit_settings.checkpoint is not None:
                with open(vit_settings.checkpoint, "rb") as f:
                    state_dict = torch.load(f)
                sam.load_state_dict(state_dict)

            sam = sam.cuda()
            sam.eval()

            self.image_encoder = sam.image_encoder

        if cnn_settings is not None:
            self.cnn_image_size = cnn_settings.image_size

        self.mask_decoder = MaskDecoder(
            model_type          =   vit_settings.model_type,
            transformer_dim     =   256,
            cnn_dim             =   cnn_settings.mscan_dimension,
        )

    def forward(
        self,
        batched_input,
        multimask_output: bool,
    ) -> List[Dict[str, Tensor]]:
        """
        Processes a batch of images, decodes mask predictions, and returns a list
        of dictionaries (one per image) with keys:
        - 'masks':              Batched mask predictions after applying sigmoid.
        - 'iou_predictions':    The IoU (mask quality) predictions after sigmoid.
        - 'uncertainty_p':      Uncertainty values computed from the masks.
        """
        #THIS MUST BE REVIEWED LATER
        scaled_images = torch.stack(
            [transforms.Resize(self.cnn_image_size)(image) for image in batched_input]
        )

        image_embeddings, interm_embeddings = self.image_encoder(batched_input)
        #END OF REVIEW
        
        # Extract features from the scaled images.
        features = self.encoder(scaled_images)
        cnn_feature = features[4]

        # Get the mask and IoU predictions from the mask decoder.
        masks, iou_pred = self.mask_decoder(
            cnn_feature         =   cnn_feature,
            image_embeddings    =   image_embeddings,
            interm_embeddings   =   interm_embeddings,
            multimask_output    =   multimask_output,
        )
        
        # Compute uncertainty as 1 - sigmoid(|masks|); higher value means higher uncertainty.
        uncertainty_p = 1 - torch.sigmoid(torch.abs(masks))
        
        # Apply sigmoid to the masks and IoU predictions.
        masks_sigmoid = torch.sigmoid(masks)
        iou_sigmoid   = torch.sigmoid(iou_pred)
        
        # Create a list of dictionaries, one per image in the batch.
        output: List[Dict[str, Tensor]] = []
        batch_size = masks_sigmoid.shape[0]
        for i in range(batch_size):
            output.append({
                "masks":            masks_sigmoid[i],
                "iou_predictions":  iou_sigmoid[i],
                "uncertainty_p":    uncertainty_p[i],
            })
        
        return output