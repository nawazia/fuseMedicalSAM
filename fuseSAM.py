import argparse
import os
import glob
import time
import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from dataset import MiniMSAMDataset
from models import load_model, calculate_segmentation_losses
from fusion import ImageLevelFusion, RegionLevelFusion, UnsupervisedFusion


def knowledge_externalization(models : list,
                              dataset : MiniMSAMDataset,
                              save_path : str = "mask_logits", device = "cpu", num_workers=0, colab = False):
    '''
    1. Create bounding box prompts per image
    2. For each model:
        1. Generate mask logits
        2. Calculate Dice, BCE, IoU losses
        3. Save mask logits & losses to disk (.npz)

    Parameters
    ----------
    models : list
        List of SAM models to be used for fusion.
    dataset : MiniMSAMDataset
        Dataset for the dataset containing images.
    save_path : str
        Path where the mask logits will be saved. Defaults to "mask_logits".
    device : str
        Device from ["cuda", "mps", "cpu"]. Defaults to "cpu".
    colab : bool
        Flag to indicate Colab use. Defaults to False.

    Returns
    -------
    save_path : str
        Path where the mask logits will be saved. Defaults to "mask_logits".
    '''
    os.makedirs(save_path, exist_ok=True)

    num_masks = dataset.get_num_masks()

    for model_name in models:
        # check if all logits already exist
        if len(glob.glob(os.path.join(save_path, model_name))) == num_masks:
            print(f"All mask logits for {model_name} already exist, skipping model...")
            continue
        t = time.time()
        dataset.set_transforms(model_name)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)
        
        os.makedirs(os.path.join(save_path, model_name), exist_ok=True)
        # load model
        model = load_model(model_name, device, colab)
        if args.device == "mps":
            gpu_memory_bytes = torch.mps.current_allocated_memory()
        else:
            gpu_memory_bytes = torch.cuda.memory_allocated()
        print(f"VRAM memory allocated: {gpu_memory_bytes / (1024**2):.2f} MB")
        print(f"Loaded model: {model_name}")
        loss = []
        for i, data in enumerate(tqdm.tqdm(dataloader)):
            mask_filenames = data['mask_filenames']
            mask_filenames = [os.path.basename(f[0]) for f in data['mask_filenames']] # Flatten list of lists and get base filename
            all_logits_exist = True
            for mask_filename in mask_filenames:
                expected_logit_path = os.path.join(save_path, model_name, f"{mask_filename[:-4]}_mask_logits.npz")
                if not os.path.exists(expected_logit_path):
                    all_logits_exist = False
                    break # No need to check further for this image

            if all_logits_exist:
                print(f"All mask logits for image associated with {mask_filenames[0]} already exist, skipping image...")
                continue

            data['image'] = data['image'].float().to(device)
            data["boxes"] = data['boxes'].to(device)

            # Generate mask logits
            # cv2.imwrite(f"img.tif", (data['image'].float().squeeze()[0].cpu().numpy()).astype(np.float32))
            mask_logits = model(data)
            gt = data["original_masks"].to(device)
            # calculate losses
            dice, bce, iou = calculate_segmentation_losses(gt, mask_logits.permute(1, 0, 2, 3))
            loss.extend(bce)
            # print(f"Mask logits shape: {mask_logits.shape}, dtype: {mask_logits.dtype}")
            # Save mask logits & losses
            mask_logits = mask_logits.squeeze(1).cpu().numpy()
            for j, mask_filename in enumerate(mask_filenames):
                mask_save_path = os.path.join(save_path, model_name, f"{mask_filename[:-4]}_mask_logits")
                print(f"Saving mask logits to: {mask_save_path}")
                np.savez_compressed(mask_save_path + ".npz",
                                    mask_logits=mask_logits[j].astype(np.float32),
                                    dice_loss=np.array(dice[j], dtype=np.float32),
                                    bce_loss=np.array(bce[j], dtype=np.float32),
                                    iou_loss=np.array(iou[j], dtype=np.float32),
                                    )
                # cv2.imwrite(mask_save_path + ".tif", (mask_logits[j]).astype(np.float32))
            if (i + 1) % 100 == 0:  # Print every 1000 iterations
                print("Mean BCE loss:",np.mean(loss[-100:]))
            #     break
        print(f"Finished processing {model_name} in {time.time() - t:.2f} seconds")
    return save_path

def fuse(models : list,
        dataset : MiniMSAMDataset,
        mask_path : str = "mask_logits",
        save_path : str = "fused", colab = False):
    '''
    For each mask:
    1. Load .npz for models in models
    2. Calculate fusion loss
    3. Argmin, save to save_path

    Parameters
    ----------
    models : list
        List of SAM models to be used for fusion.
    dataset : MiniMSAMDataset
        Dataset for the dataset containing images.
    mask_path : str
        Path where the mask logits are be saved. Defaults to "mask_logits".
    save_path : str
        Path where the fused mask_logits will be saved. Defaults to "fused".
    colab : bool
        Flag to indicate Colab use. Defaults to False.

    Returns
    -------
    None
    '''
    os.makedirs(save_path, exist_ok=True)
    dataset.set_simple()
    for i, data in enumerate(tqdm.tqdm(dataset)):
        mask_filenames = data["mask_filenames"]
        print(mask_filenames)
        # load images from mask_path/model_name
        for mask_filename in mask_filenames:
            best_model, best_data = ImageLevelFusion(models, mask_path, mask_filename)
            mask_save_path = os.path.join(save_path, f"{mask_filename[:-4]}_mask_logits")
            print(f"Saving mask logits to: {mask_save_path}")
            np.savez_compressed(save_path + ".npz", **best_data)
            print(best_model)
            test = np.load(save_path + ".npz")
            print(test)
        break
    dataset.unset_simple()
    return 0

def main(data_path: str, json_path: str, device: str = "cpu", num_workers=0, colab=False):
    dataset = MiniMSAMDataset(data_path, json_path, "train")

    models = ["MedSAM", "SAM4Med", "SAM-Med2D"]#, "Med-SA"]
    print(f"Models to be used: {models}")
    mask_path = knowledge_externalization(models, dataset, save_path=os.path.join(data_path, "mask_logits"), device=device, num_workers=num_workers, colab=colab)

    fuse(models, dataset, mask_path=mask_path, save_path=os.path.join(data_path, "fused"), colab=colab)
    return



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fuse SAM models with dataset.")
    parser.add_argument("--data_path", required=True, help="Path to dir containing images and masks (and created mask_logits) folders")
    parser.add_argument("--json_path", required=True, help="Path to JSON files containing image-mask pairs.")
    parser.add_argument("--device", default="cpu", help="Device to run the model on (default: cpu)")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers (default: 0)")
    parser.add_argument("--colab", action="store_true", help="Run on Colab (default: False)")
    args = parser.parse_args()

    main(args.data_path, args.json_path, device=args.device, num_workers=args.num_workers, colab=args.colab)
