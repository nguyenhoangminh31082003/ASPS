import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, List, Dict, Any
import torchvision.transforms as transforms


class ASPS(nn.Module):

    def __init__(
        self, 
        vit_settings, 
        cnn_settings
    ) -> None:
        super().__init__()
    
    def forward(
        self,
        batched_input,
        image_embeddings, 
        interm_embeddings,
        multimask_output: bool,
    ) -> List[Dict[str, Tensor]]:
        """
        Processes a batch of images, decodes mask predictions, and returns a list
        of dictionaries (one per image) with keys:
        - 'masks': Batched mask predictions after applying sigmoid.
        - 'iou_predictions': The IoU (mask quality) predictions after sigmoid.
        - 'low_res_logits': Uncertainty values computed from the masks.
        """
        # Resize each image to the desired CNN input size and stack them into one tensor.
        scaled_images = torch.stack(
            [transforms.Resize(self.cnn_image_size)(image) for image in batched_input]
        )
        
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