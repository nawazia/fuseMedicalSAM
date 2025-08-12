import argparse
import os
import glob
import time
from concurrent.futures import ThreadPoolExecutor
import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import MiniMSAMDataset
from models import load_model, calculate_segmentation_losses
from fusion import ImageLevelFusion, RegionLevelFusion, UnsupervisedFusion
from fusion import CombinedLoss


def knowledge_externalization(models : list,
                              dataset : MiniMSAMDataset,
                              save_path : str = "mask_logits", device = "cpu", num_workers = 0, colab = False):
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
        if len(glob.glob(os.path.join(save_path, model_name, "*.npz"))) == num_masks:
            print(f"All mask logits for {model_name} already exist, skipping model...")
            continue
        t = time.time()
        dataset.set_transforms(model_name)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)
        
        os.makedirs(os.path.join(save_path, model_name), exist_ok=True)
        # load model
        model = load_model(model_name, device, colab)
        model.eval()
        if args.device == "mps":
            gpu_memory_bytes = torch.mps.current_allocated_memory()
        else:
            gpu_memory_bytes = torch.cuda.memory_allocated()
        print(f"VRAM memory allocated: {gpu_memory_bytes / (1024**2):.2f} MB")
        print(f"Loaded model: {model_name}")
        loss = []
        for i, data in enumerate(tqdm.tqdm(dataloader)):
            mask_filenames = [os.path.basename(f[0]) for f in data['mask_filenames']] # Flatten list of lists and get base filename
            all_logits_exist = True
            for mask_filename in mask_filenames:
                expected_logit_path = os.path.join(save_path, model_name, f"{mask_filename[:-4]}_mask_logits.npz")
                if not os.path.exists(expected_logit_path):
                    all_logits_exist = False
                    break # No need to check further for this image

            if all_logits_exist:
                print(f"All mask logits for image associated with {data['image_filename']} already exist, skipping image...")
                continue

            data['image'] = data['image'].float().to(device)
            data["boxes"] = data['boxes'].to(device)

            # Generate mask logits
            # cv2.imwrite(f"img.tif", (data['image'].float().squeeze()[0].cpu().numpy()).astype(np.float32))
            mask_logits, iou_preds = model(data)               # [4, 1, 208, 174]
            assert mask_logits.dim() == 4
            gt = data["original_masks"].to(device)  # [1, 4, 208, 174]
            # calculate losses
            dice, bce, iou = calculate_segmentation_losses(gt, mask_logits.permute(1, 0, 2, 3)) # bce is now raw
            loss.extend(bce.mean(axis=(-1, -2)).flatten().tolist())
            # print(f"Mask logits shape: {mask_logits.shape}, dtype: {mask_logits.dtype}")
            # Save mask logits & losses
            mask_logits = mask_logits.squeeze(1).detach().cpu().numpy()
            iou_preds = iou_preds.squeeze(1).detach().cpu().numpy()
            for j, mask_filename in enumerate(mask_filenames):
                mask_save_path = os.path.join(save_path, model_name, f"{mask_filename[:-4]}_mask_logits")
                print(f"Saving mask logits to: {mask_save_path}")
                np.savez_compressed(mask_save_path + ".npz",
                                    mask_logits=mask_logits[j].astype(np.float32),
                                    iou_preds=iou_preds[j].astype(np.float32),
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

def process_single_mask_thread(models, method, mask_path, save_path, mask_filename, counts):
    """
    Function to process a single mask filename.
    Designed to be run by a threading worker.
    """
    mask_save_path = os.path.join(save_path, f"{os.path.basename(mask_filename)[:-4]}_mask_logits")
    
    if os.path.exists(mask_save_path + ".npz"):
        print(f"{mask_save_path}.npz already exists, skipping...")
        return None # Indicate skipping
    if method == "i":
        best_model, best_data = ImageLevelFusion(models, mask_path, mask_filename)
    elif method == "r":
        best_model, best_data = RegionLevelFusion(models, mask_path, mask_filename)
    elif method == "u":
        best_model, best_data = UnsupervisedFusion(models, mask_path, mask_filename)
    else:
        raise NotImplementedError()
    assert isinstance(best_data["mask_logits"], np.ndarray), mask_filename
    print(f"Saving mask logits to: {mask_save_path}.npz")
    np.savez_compressed(mask_save_path + ".npz", **best_data)
    
    # Update shared counts dictionary. No Manager needed for threads.
    # If the update is not atomic (e.g., if you had a complex calculation for the new value),
    # you might need a lock:
    # with counts_lock:
    counts[best_model] = counts.get(best_model, 0) + 1
        
    return best_model # Return info for main process if needed

def fuse_multithread(models: list,
                     dataset: MiniMSAMDataset,
                     mask_path: str = "mask_logits",
                     save_path: str = "fused",
                     method: str = "i",
                     max_workers: int = None): # New parameter for number of threads
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
    max_workers : int, optional
        Maximum number of threads to use. Defaults to min(32, os.cpu_count() + 4).

    Returns
    -------
    save_path : str
        Path where the fused mask_logits are saved. Defaults to "fused".
    '''
    save_path = f"{save_path}_{method}"
    os.makedirs(save_path, exist_ok=True)
    dataset.set_simple(True)

    if len(glob.glob(os.path.join(save_path, "*.npz"))) == dataset.get_num_masks():
        print(f"All fused mask logits already exist, skipping...")
        return save_path
    
    counts = dict() # Regular dictionary for threads

    if max_workers is None:
        max_workers = min(32, os.cpu_count() + 4) # Common default for ThreadPoolExecutor

    print(f"Using {max_workers} threads for fusion.")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, data in enumerate(dataset):
            mask_filenames = data["mask_filenames"]
            for mask_filename in mask_filenames:
                future = executor.submit(process_single_mask_thread, models, method, mask_path, save_path, mask_filename, counts)
                futures.append(future)

        # Use tqdm to show progress as futures complete
        for future in tqdm.tqdm(futures, total=len(futures)):
            future.result() # Wait for each future to complete

    print("Fusion complete. Model counts:")
    print(counts)
    return save_path

def eval_post_epoch(model, test_dataloader, criterion, device, fancy=False):
    model.eval()
    test_losses = []
    test_bce_losses = []
    test_dice_losses = []

    if fancy:
        modality_dice = {}
        dataset_dice = {}

    pbar_test = tqdm.tqdm(test_dataloader, desc="Testing")
    with torch.no_grad():
        for data in pbar_test:
            data['image'] = data['image'].to(device).float()
            data["boxes"] = data['boxes'].to(device)
            gt = data["original_masks"].to(device)
            mask_logits, iou_preds = model(data)
            mask_logits = mask_logits.permute(1, 0, 2, 3) 

            gt = gt.float()
            mask_logits = mask_logits.float()
            combined_loss, bce_loss, dice_loss, _ = criterion(mask_logits, gt)

            test_losses.append(combined_loss.item())
            test_bce_losses.append(bce_loss.item())
            test_dice_losses.append(dice_loss.item())

            if fancy:
                info = os.path.basename(data["image_filename"][0]).split("--")
                modality = info[0]
                mscores = modality_dice.get(modality, [])
                mscores.append(dice_loss.item())
                modality_dice[modality] = mscores

                dataset = info[1]
                dscores = dataset_dice.get(dataset, [])
                dscores.append(dice_loss.item())
                dataset_dice[dataset] = dscores
            
            pbar_test.set_postfix({
                'test_loss': f'{combined_loss.item():.4f}',
            })

    pbar_test.close()
    avg_val_loss = sum(test_losses) / len(test_losses)
    avg_val_bce = sum(test_bce_losses) / len(test_bce_losses)
    avg_val_dice = sum(test_dice_losses) / len(test_dice_losses)
    
    print(f"Avg Test Loss: {avg_val_loss:.4f} | Avg Test BCE: {avg_val_bce:.4f} | Avg Test Dice: {avg_val_dice:.4f}")
    if fancy:
        print("---Modality Scores---")
        for mod, scores in modality_dice.items():
            print(f"{mod}: {np.mean(scores)}")

        print("---Dataset Scores---")
        for dat, scores in dataset_dice.items():
            print(f"{dat}: {np.mean(scores)}")
    return

def continual_training(target : str, dataset : MiniMSAMDataset, test_dataset : MiniMSAMDataset, fused_path : str = "fused", device="cpu", num_workers=0, colab=False, epochs=10):
    '''
    1. Load target model in train mode.


    Parameters
    ----------
    target : str
        Pre-trained finetuned SAM to be used for target model.
    dataset : MiniMSAMDataset
        Dataset for the dataset containing images.
    fused_path : str
        Path where the fused mask_logits are saved. Defaults to "fused".
    device : str
        Device from ["cuda", "mps", "cpu"]. Defaults to "cpu".
    max_workers : int
        Maximum number of threads to use. Defaults to min(32, os.cpu_count() + 4).
    colab : bool
        Flag to indicate Colab use. Defaults to False.

    Returns
    -------
    None
    '''
    dataset.set_transforms(target)
    test_dataset.set_transforms(target)
    dataset.set_fused(fused_path)
    model = load_model(target, device, colab)
    if args.device == "mps":
        gpu_memory_bytes = torch.mps.current_allocated_memory()
    else:
        gpu_memory_bytes = torch.cuda.memory_allocated()
    print(f"VRAM memory allocated: {gpu_memory_bytes / (1024**2):.2f} MB")
    print(f"Loaded model: {target}")
    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = CombinedLoss()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    eval_post_epoch(model, test_dataloader, criterion, device, fancy=True)
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        model.train()
        pbar_train = tqdm.tqdm(dataloader, desc="Training")
        for i, data in enumerate(pbar_train):
            # mask_filenames = [os.path.basename(f[0]) for f in data['mask_filenames']] # Flatten list of lists and get base filename
            data['image'] = data['image'].to(device).float()
            data["boxes"] = data['boxes'].to(device)

            # Generate mask logits
            mask_logits, iou_preds = model(data)                   # [4, 1, 208, 174]
            assert mask_logits.dim() == 4
            gt = data["original_masks"].to(device)      # [1, 4, 208, 174]
            teacher_logits = data["teacher_logits"].to(device)
            # calculate losses
            optimizer.zero_grad()
            gt = gt.float()
            mask_logits = mask_logits.float()
            combined_loss, bce_loss, dice_loss, distillation_loss = criterion(mask_logits.permute(1, 0, 2, 3), gt, teacher_logits)
            combined_loss.backward()
            optimizer.step()
            pbar_train.set_postfix({
                'total_loss': f'{combined_loss.item():.4f}',
                'bce_loss': f'{bce_loss.item():.4f}',
                'dice_loss': f'{dice_loss.item():.4f}',
                'distillation_loss': f'{distillation_loss.item():.4f}'
            })
        pbar_train.close()
        eval_post_epoch(model, test_dataloader, criterion, device)
    
    print("Training complete!")
    eval_post_epoch(model, test_dataloader, criterion, device, fancy=True)
    return model

def main(data_path: str, json_path: str, device: str = "cpu", fusion="i", num_workers=0, epochs=10, colab=False):
    dataset = MiniMSAMDataset(data_path, json_path, "train")

    target = "SAM-Med2D"
    models = ["MedSAM", "SAM4Med", "SAM-Med2D"]#, "Med-SA"]
    print(f"Models to be used: {models}")
    # KE
    mask_path = knowledge_externalization(models, dataset, save_path=os.path.join(data_path, "mask_logits"), device=device, num_workers=num_workers, colab=colab)
    # Fusion
    fused_path = fuse_multithread(models, dataset, mask_path=mask_path, save_path=os.path.join(data_path, "fused"), method=fusion, max_workers=num_workers)
    dataset.set_simple(False)
    # Continual training
    test_dataset = MiniMSAMDataset(data_path, json_path, "test")
    model = continual_training(target, dataset, test_dataset, fused_path, device=device, num_workers=num_workers, colab=colab, epochs=epochs)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fuse SAM models with dataset.")
    parser.add_argument("--data_path", required=True, help="Path to dir containing images and masks (and created mask_logits) folders")
    parser.add_argument("--json_path", required=True, help="Path to JSON files containing image-mask pairs.")
    parser.add_argument("--device", default="cpu", help="Device to run the model on (default: cpu)")
    parser.add_argument("--fusion", default="i", help="Fusion method, choose from ['i', 'r', 'u'] (default: i)")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers (default: 0)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs (default: 10)")
    parser.add_argument("--colab", action="store_true", help="Run on Colab (default: False)")
    args = parser.parse_args()

    main(args.data_path, args.json_path, device=args.device, fusion=args.fusion, num_workers=args.num_workers, epochs=args.epochs, colab=args.colab)
