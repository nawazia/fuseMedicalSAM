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

def loss_function():
    return

if __name__ == "__main__":
    ImageLevelFusion(["MedSAM"], "/Users/i/ICL/fusion/code/data/17K/SAMed2Dv1/mask_logits/", "mr_00--AMOS2022--amos_0596--y_0081--0002_000.png")
