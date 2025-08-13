import argparse
import os
import glob
import time
from concurrent.futures import ThreadPoolExecutor
import io
import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.cloud import storage
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import MiniMSAMDataset
from models import load_model, calculate_segmentation_losses
from fusion import ImageLevelFusion, RegionLevelFusionGCS, UnsupervisedFusionGCS
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
    gcs : bool
        Flag to indicate gcs use. Defaults to False.

    Returns
    -------
    save_path : str
        Path where the mask logits will be saved. Defaults to "mask_logits".
    '''
    if not save_path.startswith("gs://"):
        raise ValueError("GCS is enabled, but 'save_path' does not start with 'gs://'")

    # Split the GCS path into bucket name and blob prefix
    path_parts = save_path[5:].split("/", 1)
    bucket_name = path_parts[0]
    blob_prefix = path_parts[1] if len(path_parts) > 1 else ""      # path/to/mask_logits

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    print("bucket_name:", bucket_name)

    num_masks = dataset.get_num_masks()
    print("Number of masks:", num_masks)

    for model_name in models:
        # check if all logits already exist
        model_blob_prefix = f"{blob_prefix}/{model_name}/"
        # List all objects under the model's prefix and check the count
        blob_list = list(bucket.list_blobs(prefix=model_blob_prefix))
        if len(blob_list) == num_masks:
            print(f"All mask logits for {model_name} already exist, skipping model...")
            continue
        elif len(blob_list) > num_masks:
            raise TypeError(f"more files than masks: {len(blob_list)} files and {num_masks} masks")
        else:
            print(f"Found {len(blob_list)} out of {num_masks}")
        
        t = time.time()
        dataset.set_transforms(model_name)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)
        
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
                blob_name = f"{blob_prefix}/{model_name}/{mask_filename[:-4]}_mask_logits.npz"
                if not bucket.blob(blob_name).exists():
                    all_logits_exist = False
                    break

            if all_logits_exist:
                print(f"All mask logits for image associated with {mask_filenames[0]} already exist, skipping image...")
                continue

            data['image'] = data['image'].to(device).float()
            data["boxes"] = data['boxes'].to(device)

            # Generate mask logits
            # cv2.imwrite(f"img.tif", (data['image'].float().squeeze()[0].cpu().numpy()).astype(np.float32))
            mask_logits, iou_preds = model(data)               # [4, 1, 208, 174]
            assert mask_logits.dim() == 4
            gt = data["original_masks"].to(device)  # [1, 4, 208, 174]
            # calculate losses
            dice, bce, iou = calculate_segmentation_losses(gt, mask_logits.permute(1, 0, 2, 3))
            loss.extend(bce.mean(axis=(-1, -2)).flatten().tolist())
            # print(f"Mask logits shape: {mask_logits.shape}, dtype: {mask_logits.dtype}")
            # Save mask logits & losses
            mask_logits = mask_logits.squeeze(1).detach().cpu().numpy()
            iou_preds = iou_preds.squeeze(1).detach().cpu().numpy()
            for j, mask_filename in enumerate(mask_filenames):
                mem_file = io.BytesIO()
                np.savez_compressed(mem_file,
                                    mask_logits=mask_logits[j].astype(np.float32),
                                    iou_preds=iou_preds[j].astype(np.float32),
                                    dice_loss=np.array(dice[j], dtype=np.float32),
                                    bce_loss=np.array(bce[j], dtype=np.float32),
                                    iou_loss=np.array(iou[j], dtype=np.float32),
                                    )
                mem_file.seek(0)  # Rewind the in-memory file to the beginning
                
                blob_name = f"{blob_prefix}/{model_name}/{mask_filename[:-4]}_mask_logits.npz"
                print(f"Saving mask logits to GCS blob: {blob_name}")
                blob = bucket.blob(blob_name)
                blob.upload_from_file(mem_file, content_type='application/octet-stream')

            if (i + 1) % 100 == 0:  # Print every 1000 iterations
                print("Mean BCE loss:",np.mean(loss[-100:]))
            #     break
        print(f"Finished processing {model_name} in {time.time() - t:.2f} seconds")
    return save_path


def process_single_mask_thread(models, bucket, method, mask_path_prefix, save_path_prefix, mask_filename, counts):
    """
    Function to process a single mask filename.
    Designed to be run by a threading worker, with GCS support.
    """
    assert bucket is not None, "bucket must be set for gcs"
    # Construct the GCS blob name for the final fused file
    mask_save_blob_name = f"{save_path_prefix}/{os.path.basename(mask_filename)[:-4]}_mask_logits.npz"

    # Check if the fused file already exists in the GCS bucket
    if bucket.blob(mask_save_blob_name).exists():
        # print(f"Fused file for {mask_save_blob_name} already exists, skipping...")
        return None # Indicate skipping

    # It must use the 'bucket' object and the 'mask_path_prefix' to load
    # the individual model logits from GCS.
    if method == "i":
        best_model, best_data = ImageLevelFusion(models, bucket, mask_path_prefix, mask_filename)
    elif method == "r":
        best_model, best_data = RegionLevelFusionGCS(models, bucket, mask_path_prefix, mask_filename)
    elif method == "u":
        best_model, best_data = UnsupervisedFusionGCS(models, bucket, mask_path_prefix, mask_filename)
    else:
        raise NotImplementedError()
    assert isinstance(best_data["mask_logits"], np.ndarray), mask_filename
    print(f"Saving fused logits to GCS blob: {mask_save_blob_name}")
    
    # Save the data to an in-memory buffer, then upload to GCS
    if bucket is not None:
        mem_file = io.BytesIO()
        np.savez_compressed(mem_file, **best_data)
        mem_file.seek(0)
        
        blob = bucket.blob(mask_save_blob_name)
        blob.upload_from_file(mem_file, content_type='application/octet-stream')
    else:
        np.savez_compressed(mask_save_blob_name, **best_data)

    counts[best_model] = counts.get(best_model, 0) + 1
        
    return best_model

def fuse_multithread(models: list,
                     dataset: MiniMSAMDataset,
                     mask_path: str = "mask_logits",
                     save_path: str = "fused",
                     method: str = "i",
                     max_workers: int = None,
                    ):
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
    if not mask_path.startswith("gs://") or not save_path.startswith("gs://"):
        raise ValueError("GCS is enabled, but 'mask_path' or 'save_path' do not start with 'gs://'")
    save_path = f"{save_path}_{method}"
    
    # Parse GCS paths
    client = storage.Client()
    mask_path_parts = mask_path[5:].split("/", 1)
    save_path_parts = save_path[5:].split("/", 1)
    bucket_name = mask_path_parts[0]
    
    mask_path_prefix = mask_path_parts[1] if len(mask_path_parts) > 1 else ""
    save_path_prefix = save_path_parts[1] if len(save_path_parts) > 1 else ""
    
    bucket = client.bucket(bucket_name)

    dataset.set_simple(True)

    num_masks = dataset.get_num_masks()
    blob_list = list(bucket.list_blobs(prefix=save_path_prefix))
    if len(blob_list) == num_masks:
        print(f"All fused mask logits already exist, skipping...")
        return save_path
    elif len(blob_list) > num_masks:
        raise TypeError(f"more files than masks: {len(blob_list)} files and {num_masks} masks")
    else:
        print(f"Found {len(blob_list)} out of {num_masks}")

    counts = dict() # Regular dictionary for threads

    if max_workers is None:
        max_workers = min(32, os.cpu_count() + 4) # Common default for ThreadPoolExecutor

    print(f"Using {max_workers} threads for fusion.")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, data in enumerate(dataset):
            mask_filenames = data["mask_filenames"]
            for mask_filename in mask_filenames:
                future = executor.submit(process_single_mask_thread, models, bucket, method, mask_path_prefix, save_path_prefix, mask_filename, counts)
                futures.append(future)

        # Use tqdm to show progress as futures complete
        for future in tqdm.tqdm(futures, total=len(futures)):
            future.result() # Wait for each future to complete

    print("Fusion complete. Model counts:")
    print(counts)
    return save_path

def eval_post_epoch(model, dataloader, criterion, device, split="Test", debug=False, fancy=False):
    model.eval()
    test_losses = []
    test_bce_losses = []
    test_dice_losses = []
    if fancy:
        modality_dice = {}
        dataset_dice = {}

    pbar_test = tqdm.tqdm(dataloader, desc=split)
    with torch.no_grad():
        for data in pbar_test:
            if debug:
                print("image:", data["image_filename"])
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
                modality = info[0].split("_")[0]
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
    
    print(f"Avg {split} Loss: {avg_val_loss:.4f} | Avg {split} BCE: {avg_val_bce:.4f} | Avg {split} Dice: {avg_val_dice:.4f}")
    if fancy:
        print("---Modality Scores---")
        for mod, scores in sorted(modality_dice.items(), key=lambda item: 1 - np.mean(item[1])):
            print(f"{mod}: {1 - np.mean(scores)}")

        print("---Dataset Scores---")
        for dat, scores in sorted(dataset_dice.items(), key=lambda item: 1 - np.mean(item[1])):
            print(f"{dat}: {1 - np.mean(scores)}")
    return

def external_eval(model, target, criterion, num_workers=0, device="cuda"):
    # first eval on kits
    kits_path = os.path.join("data", "kits23")
    kits_ds = MiniMSAMDataset("sam-med2d-17k", kits_path, os.path.join(kits_path, "KiTS23.json"), "test")
    kits_ds.set_transforms(target)
    kits_dl = DataLoader(kits_ds, batch_size=1, shuffle=False, num_workers=num_workers)
    model.eval()
    test_bce_losses = []
    test_dice_losses = []

    pbar_test = tqdm.tqdm(kits_dl, desc="Testing")
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

            test_bce_losses.append(bce_loss.item())
            test_dice_losses.append(dice_loss.item())
            
            pbar_test.set_postfix({
                'test_loss': f'{combined_loss.item():.4f}',
            })

    pbar_test.close()
    avg_val_bce = sum(test_bce_losses) / len(test_bce_losses)
    avg_val_dice = sum(test_dice_losses) / len(test_dice_losses)
    
    print(f"Avg KiTS23 BCE: {avg_val_bce:.4f} | Avg KiTS23 Dice: {avg_val_dice:.4f}")
    
    # first eval on kits
    segrap_path = os.path.join("data", "segrap23")
    segrap_ds = MiniMSAMDataset("sam-med2d-17k", segrap_path, os.path.join(segrap_path, "segrap23.json"), "test")
    segrap_ds.set_transforms(target)
    segrap_dl = DataLoader(segrap_ds, batch_size=1, shuffle=False, num_workers=num_workers)
    model.eval()
    test_bce_losses = []
    test_dice_losses = []

    pbar_test = tqdm.tqdm(segrap_dl, desc="Testing")
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

            test_bce_losses.append(bce_loss.item())
            test_dice_losses.append(dice_loss.item())
            
            pbar_test.set_postfix({
                'test_loss': f'{combined_loss.item():.4f}',
            })

    pbar_test.close()
    avg_val_bce = sum(test_bce_losses) / len(test_bce_losses)
    avg_val_dice = sum(test_dice_losses) / len(test_dice_losses)
    
    print(f"Avg SegRap2023 BCE: {avg_val_bce:.4f} | Avg SegRap2023 Dice: {avg_val_dice:.4f}")
    return 0

def continual_training(target : str, dataset : MiniMSAMDataset, val_dataset : MiniMSAMDataset, test_dataset : MiniMSAMDataset, fused_path : str = "fused", device="cpu", num_workers=0, colab=False, debug=False, epochs=10):
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
    dataset.set_debug(debug)
    val_dataset.set_transforms(target)
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
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    external_eval(model, target, criterion, num_workers, device)
    eval_post_epoch(model, test_dataloader, criterion, device, debug, fancy=True)
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        model.train()
        pbar_train = tqdm.tqdm(dataloader, desc="Training")
        for i, data in enumerate(pbar_train):
            if debug:
                print("image:", data["image_filename"])
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
        eval_post_epoch(model, val_dataloader, criterion, device, split="Val")
    
    print("Training complete!")
    eval_post_epoch(model, test_dataloader, criterion, device, fancy=True)
    return model


def main(target: str, data_path: str, json_path: str, device: str = "cpu", fusion="i", num_workers=0, epochs=10, debug=False):
    dataset = MiniMSAMDataset("sam-med2d-17k", data_path, json_path, "train")
    val_dataset = MiniMSAMDataset("sam-med2d-17k", data_path, json_path, "val")
    test_dataset = MiniMSAMDataset("sam-med2d-17k", data_path, json_path, "test")

    models = ["MedSAM", "SAM4Med", "SAM-Med2D"]#, "Med-SA"]
    assert target in models
    print(f"Models to be used: {models}")
    # KE
    save_path = os.path.join("gs://sam-med2d-17k", data_path, "mask_logits")

    mask_path = knowledge_externalization(models, dataset, save_path=save_path, device=device, num_workers=num_workers, colab=True)
    # Fusion
    fused_path = fuse_multithread(models, dataset, mask_path=mask_path, save_path=os.path.join(os.path.dirname(mask_path), "fused"), method=fusion, max_workers=num_workers)
    dataset.set_simple(False)
    # Continual training
    model = continual_training(target, dataset, val_dataset, test_dataset, f"fused_{fusion}", device=device, num_workers=num_workers, colab=True, debug=debug, epochs=epochs)
    torch.save(model.state_dict(), f"/content/drive/My Drive/fused_{target}_{fusion}.pth")
    external_eval(model, target, CombinedLoss(), num_workers=num_workers, device=device)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fuse SAM models with dataset.")
    parser.add_argument("--data_path", required=True, help="Path to dir containing images and masks (and created mask_logits) folders")
    parser.add_argument("--json_path", required=True, help="Path to JSON files containing image-mask pairs.")
    parser.add_argument("--device", default="cpu", help="Device to run the model on (default: cpu)")
    parser.add_argument("--fusion", default="i", help="Fusion method, choose from ['i', 'r', 'u'] (default: i)")
    parser.add_argument("--target", default="SAM-Med2D", help="Target mode, choose from ['SAM-Med2D', 'SAM4Med', 'MedSAM'] (default: SAM-Med2D)")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers (default: 0)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs (default: 10)")
    parser.add_argument("--debug", action="store_true", help="Debug mode (default: False)")
    args = parser.parse_args()

    main(args.target, args.data_path, args.json_path, device=args.device, fusion=args.fusion, num_workers=args.num_workers, epochs=args.epochs, debug=args.debug)
