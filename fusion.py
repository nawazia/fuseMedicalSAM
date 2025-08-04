import os
import numpy as np

def ImageLevelFusion(models, mask_path, mask_filename):
    min_loss = np.inf
    for model_name in models:
        print(mask_path)
        print(model_name)
        print(os.path.basename(mask_filename)[:-4] + "_mask_logits.npz")
        mask_path = os.path.join(mask_path, model_name, os.path.basename(mask_filename)[:-4] + "_mask_logits.npz")
        print(mask_path)
        data = np.load(mask_path)

    return data

def RegionLevelFusion(models, mask_path):

    return

def UnsupervisedFusion(models, mask_path):

    return
