import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import InterpolationMode, resize
import cv2
import numpy as np

def resize_box_to_target(box, original_size, target_size=256):
    """
    the input bounding box is obtained from the original image
    here, we rescale it to the coordinates of the resized image

    Parameters
    ----------
    box : numpy.ndarray
        bounding box coordinates in the original image
    original_size : tuple
        the original size of the image

    Returns
    -------
    numpy.ndarray
        bounding box coordinates in the resized image
    """
    new_box = np.zeros_like(box)
    ratio = target_size / max(original_size)
    for i in range(len(box)):
        new_box[i] = int(box[i] * ratio)

    return new_box

def pad_image(image_tensor: torch.Tensor, target_size: int = 256) -> torch.Tensor:
    """
    Pad image tensor to target_size.
    Expects a PyTorch tensor with shape CxHxW (for image) or HxW (for mask).
    Pads with zeros.
    """
    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError(f"Input image must be a torch.Tensor, but got {type(image_tensor)}")

    # Get H and W based on tensor dimensions
    if image_tensor.dim() == 4: # NxCxHxW
        h, w = image_tensor.shape[-2:]
    else:
        raise ValueError(f"Input image tensor must be 2D (HxW) or 3D (CxHxW), but got {image_tensor.dim()} dimensions.")

    padh = target_size - h
    padw = target_size - w

    # F.pad expects padding in the order (pad_left, pad_right, pad_top, pad_bottom) for the last two dimensions
    # For a CxHxW tensor, this means (W_left, W_right, H_top, H_bottom, C_front, C_back)
    # We want to pad at the bottom and right, so (0, padw, 0, padh)
    padding = (0, padw, 0, padh) # (left, right, top, bottom)

    return F.pad(image_tensor, padding, mode='constant', value=0)

class resize_longest_side(object):
    """
    Resizes the longest side of an image/tensor to target_length while keeping the aspect ratio.
    Can be used as part of torchvision.transforms.Compose.

    Args:
        target_length (int): The desired length of the longest side.
        interpolation (torchvision.transforms.InterpolationMode): Interpolation mode for resizing.
            Use InterpolationMode.NEAREST for masks to preserve discrete values.
    """
    def __init__(self, target_length=256, interpolation=InterpolationMode.BICUBIC):
        self.target_length = target_length
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL Image or torch.Tensor): Image or tensor to be resized.
                If torch.Tensor, it is expected to have [..., H, W] shape.
        Returns:
            PIL Image or torch.Tensor: Resized image or tensor.
        """
        oldh, oldw = img.shape[-2:]
        scale = self.target_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        
        neww, newh = int(neww + 0.5), int(newh + 0.5)
        target_size = (newh, neww) # torchvision.transforms.Resize expects (H, W)

        return resize(img, target_size, interpolation=self.interpolation)

class _MedSAM_Lite(nn.Module):
    def __init__(
            self, 
            image_encoder, 
            mask_decoder,
            prompt_encoder
        ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

    def forward(self, image, box_np):
        image_embedding = self.image_encoder(image) # (B, 256, 64, 64)
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            box_torch = torch.as_tensor(box_np, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :] # (B, 1, 4)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=box_np,
            masks=None,
        )
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embedding, # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
          ) # (B, 1, 256, 256)

        return low_res_masks

    @torch.no_grad()
    def postprocess_masks(self, masks, new_size, original_size):
        """
        Do cropping and resizing

        Parameters
        ----------
        masks : torch.Tensor
            masks predicted by the model
        new_size : tuple
            the shape of the image after resizing to the longest side of 256
        original_size : tuple
            the original shape of the image

        Returns
        -------
        torch.Tensor
            the upsampled mask to the original size
        """
        # Crop
        masks = masks[..., :new_size[0], :new_size[1]]
        # Resize
        masks = F.interpolate(
            masks,
            size=(original_size[0], original_size[1]),
            mode="bilinear",
            align_corners=False,
        )

        return masks

@torch.no_grad()
def litemedsam_inference(medsam_model, img_embed, box_256, new_size, original_size, return_logits=False):
    """
    Perform inference using the LiteMedSAM model.

    Args:
        medsam_model (MedSAMModel): The MedSAM model.
        img_embed (torch.Tensor): The image embeddings.
        box_256 (numpy.ndarray): The bounding box coordinates.
        new_size (tuple): The new size of the image.
        original_size (tuple): The original size of the image.
    Returns:
        tuple: A tuple containing the segmented image and the intersection over union (IoU) score.
    """
    box_torch = torch.as_tensor(box_256[None, None, ...], dtype=torch.float, device=img_embed.device)
    
    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points = None,
        boxes = box_torch,
        masks = None,
    )
    low_res_logits, iou = medsam_model.mask_decoder(
        image_embeddings=img_embed, # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
        multimask_output=False
    )

    low_res_pred = medsam_model.postprocess_masks(low_res_logits, new_size, original_size)
    if return_logits:
        low_res_pred = low_res_pred.squeeze().cpu().numpy()
        return low_res_pred, iou
    low_res_pred = torch.sigmoid(low_res_pred)  
    low_res_pred = low_res_pred.squeeze().cpu().numpy()
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)

    return medsam_seg, iou
