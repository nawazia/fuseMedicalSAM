import os
import numpy as np
import cv2
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
        loss = bce.mean() + dice
        if loss < min_loss:
            data = (model_name, cur)

    return data

def RegionLevelFusion(models, mask_path, mask_filename):
    """
    Creates a composite mask logit by selecting the best logit for each pixel
    from a set of models, based on the lowest pixel-wise BCE loss.

    Args:
        models (list): A list of model names.
        mask_path (str): The base path where model data is stored.
        mask_filename (str): The filename of the original mask.

    Returns:
        tuple: A tuple containing the composite mask logit (np.ndarray) and
               the corresponding composite BCE scores (np.ndarray).
    """
    comp_mask, comp_bce = None, None
    for model_name in models:
        mask_path_full = os.path.join(mask_path, model_name, os.path.basename(mask_filename)[:-4] + "_mask_logits.npz")
        cur = np.load(mask_path_full)
        cur_mask = cur["mask_logits"]
        cur_bce = cur["bce_loss"]
        if comp_mask is None:
            comp_mask = cur_mask
            comp_bce = cur_bce
        else:
            # Find the pixels where the current model has a lower BCE loss
            # This creates a boolean mask of True where the condition is met
            is_better_pixel = cur_bce < comp_bce

            # Use the boolean mask to update only those pixels in the composite arrays
            # np.where is another option, but this is often more concise
            comp_mask[is_better_pixel] = cur_mask[is_better_pixel]
            comp_bce[is_better_pixel] = cur_bce[is_better_pixel]
    
    result = {}
    result["mask_logits"] = comp_mask
    return ("Composite", result)

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

class SegmentationLoss(nn.Module):
    def __init__(self, dice_weight=1.0, bce_weight=1.0):
        super(SegmentationLoss, self).__init__()
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

class DistillationLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        # Use BCEWithLogitsLoss for numerical stability
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, student_logits, teacher_logits):
        # We need a "soft" target for the student. This is the teacher's softened output.
        # F.sigmoid is used to get probabilities
        # The output of sigmoid will be in the range [0, 1]
        soft_targets = torch.sigmoid(teacher_logits / self.temperature)
        
        # We then compute the Binary Cross-Entropy between the student's
        # temperature-scaled logits and the soft targets.
        # F.binary_cross_entropy_with_logits is more numerically stable than
        # applying sigmoid and then BCE loss.
        # Note: The student's logits are also scaled by the temperature.
        loss = self.bce_loss(student_logits / self.temperature, soft_targets)
        
        # A common practice is to scale the distillation loss by T^2 to keep
        # its magnitude comparable to other losses.
        return loss * (self.temperature ** 2)

class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=1.0, bce_weight=1.0, lambda_weight=0.9, temperature=1.0):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.distillation_loss = DistillationLoss(temperature=temperature)

        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.lambda_weight = lambda_weight

    def forward(self, pred_logits, ground_truth_masks, teacher_logits=None):
        dice_loss = self.dice_loss(pred_logits, ground_truth_masks)
        bce_loss = self.bce_loss(pred_logits, ground_truth_masks)
        
        segmentation_loss = (self.bce_weight * bce_loss) + (self.dice_weight * dice_loss)
        if teacher_logits is not None:
            distillation_loss = self.distillation_loss(pred_logits, teacher_logits)
            combined_loss = self.lambda_weight * segmentation_loss + \
                            (1 - self.lambda_weight) * distillation_loss
            
            return combined_loss, bce_loss, dice_loss, distillation_loss
        else:
            # For validation/testing, there are no teacher logits.
            # We return the segmentation loss as the primary metric.
            return segmentation_loss, bce_loss, dice_loss, torch.tensor(0.0)


if __name__ == "__main__":
    ImageLevelFusion(["MedSAM"], "/Users/i/ICL/fusion/code/data/17K/SAMed2Dv1/mask_logits/", "mr_00--AMOS2022--amos_0596--y_0081--0002_000.png")
