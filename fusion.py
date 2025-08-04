import os
import numpy as np
import torch
import torch.nn as nn

def ImageLevelFusion(models, mask_path, mask_filename):
    min_loss = np.inf
    data = (None, None)
    for model_name in models:
        mask_path_full = os.path.join(mask_path, model_name, os.path.basename(mask_filename)[:-4] + "_mask_logits.npz")
        cur = np.load(mask_path_full)
        dice = cur["dice_loss"]
        bce = cur["bce_loss"]
        loss = bce + dice
        if loss < min_loss:
            data = (model_name, cur)

    return data

def RegionLevelFusion(models, mask_path):

    return

def UnsupervisedFusion(models, mask_path):

    return

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, mask):
        """
        pred: [B, C, H, W]
        mask: [B, C, H, W]
        
        Where C is the number of channels/masks (e.g., 4 in your case).
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."
        
        # Apply sigmoid to the predicted mask logits
        p = torch.sigmoid(pred)
        
        # The key change is to flatten only the spatial dimensions.
        # We want to preserve the batch (B) and channel (C) dimensions.
        # This transforms [B, C, H, W] -> [B, C, H*W]
        p_flat = p.view(p.shape[0], p.shape[1], -1) 
        mask_flat = mask.view(mask.shape[0], mask.shape[1], -1)
        
        # Calculate intersection and union.
        # We sum over the flattened spatial dimension (dim=2).
        # This will give us a tensor of shape [B, C].
        intersection = torch.sum(p_flat * mask_flat, dim=2) 
        union = torch.sum(p_flat, dim=2) + torch.sum(mask_flat, dim=2)
        
        # Calculate the Dice coefficient for each channel for each item in the batch
        # The result is a tensor of shape [B, C]
        dice_coefficient = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # The loss for each channel and each item in the batch
        dice_loss = 1.0 - dice_coefficient
        
        # Finally, average the loss across the entire batch (B) AND channels (C)
        # to get a single scalar for backpropagation.
        return torch.mean(dice_loss)

class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=1.0, bce_weight=1.0):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, pred_logits, ground_truth_masks):
        # Calculate individual losses
        dice_loss = self.dice_loss(pred_logits, ground_truth_masks)
        bce_loss = self.bce_loss(pred_logits, ground_truth_masks)
        
        # Combine the losses with weights
        combined_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        
        return combined_loss, bce_loss, dice_loss # Return individual losses for monitoring

if __name__ == "__main__":
    ImageLevelFusion(["MedSAM"], "/Users/i/ICL/fusion/code/data/17K/SAMed2Dv1/mask_logits/", "mr_00--AMOS2022--amos_0596--y_0081--0002_000.png")
