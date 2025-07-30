import torch
import torch.nn.functional as F
import numpy as np


@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, H, W, return_logits=False):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, iou = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )
    low_res_logits = F.interpolate(
        low_res_logits,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )

    if return_logits:
        # low_res_logits = low_res_logits.squeeze().cpu().numpy()
        return low_res_logits, iou
    else:
        low_res_pred = torch.sigmoid(low_res_logits)  
        low_res_pred = low_res_pred.squeeze().cpu().numpy()
        medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
        return medsam_seg, iou
