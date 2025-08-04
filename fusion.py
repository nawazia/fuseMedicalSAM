import os
import numpy as np

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
