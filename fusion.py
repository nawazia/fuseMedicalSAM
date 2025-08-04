import os
import numpy as np

def ImageLevelFusion(models, mask_path, mask_filename):
    min_loss = np.inf
    for model_name in models:
        mask_path = os.path.join(mask_path, model_name, os.path.basename(mask_filename)[:-4] + "_mask_logits.npz")
        data = np.load(mask_path)

    return data

def RegionLevelFusion(models, mask_path):

    return

def UnsupervisedFusion(models, mask_path):

    return
