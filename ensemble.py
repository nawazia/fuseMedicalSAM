import argparse
import os
import re
import random
import json
import glob

import numpy as np
import cv2
from tqdm import tqdm
import h5py
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode
from MedSAM.segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
# from MedSAM.segment_anything.modeling.mask_decoder import MaskDecoder
# from MedSAM.segment_anything.modeling.prompt_encoder import PromptEncoder
# from MedSAM.segment_anything.modeling.transformer import TwoWayTransformer
from MedSAM.tiny_vit_sam import TinyViT
from MedSAM.CVPR24_acdc import resize_longest_side, pad_image, MedSAM_Lite, resize_box_to_256, medsam_inference
from SAM4Med.segment_anything import sam_model_registry, SamPredictor
from SAM4Med.segment_anything.utils.transforms import ResizeLongestSide

class ACDCDataset(Dataset):
    def __init__(self, base_dir=None, split='train', transform=None, fold="fold1", sup_type="label"):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.sup_type = sup_type
        self.transform = transform
        # if self.split == "train" or self.split == "val":
        #     train_ids, test_ids = self._get_fold_ids(fold)
        if self.split == 'train':
            self.all_volumes = os.listdir(
                self._base_dir + "/ACDC_training_volumes")
            self.sample_list = []
            self.sample_list.extend(self.all_volumes)
            # self.all_slices = os.listdir(
            #     self._base_dir + "/ACDC_training_slices")
            # self.sample_list = []
            # for ids in train_ids:
            #     new_data_list = list(filter(lambda x: re.match(
            #         '{}.*'.format(ids), x) != None, self.all_slices))
            #     self.sample_list.extend(new_data_list)
        # elif self.split == 'val':
        #     self.all_volumes = os.listdir(
        #         self._base_dir + "/ACDC_training_volumes")
        #     self.sample_list = []
        #     for ids in test_ids:
        #         new_data_list = list(filter(lambda x: re.match(
        #             '{}.*'.format(ids), x) != None, self.all_volumes))
        #         self.sample_list.extend(new_data_list)
        else:
            self.all_volumes = os.listdir(
                self._base_dir + "/ACDC_testing_volumes")
            self.sample_list = []
            self.sample_list.extend(self.all_volumes)

        # if num is not None and self.split == "train":
        #     self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def _get_fold_ids(self, fold):
        all_cases_set = ["patient{:0>3}".format(i) for i in range(1, 101)]
        fold1_testing_set = [
            "patient{:0>3}".format(i) for i in range(1, 21)]
        fold1_training_set = [
            i for i in all_cases_set if i not in fold1_testing_set]

        fold2_testing_set = [
            "patient{:0>3}".format(i) for i in range(21, 41)]
        fold2_training_set = [
            i for i in all_cases_set if i not in fold2_testing_set]

        fold3_testing_set = [
            "patient{:0>3}".format(i) for i in range(41, 61)]
        fold3_training_set = [
            i for i in all_cases_set if i not in fold3_testing_set]

        fold4_testing_set = [
            "patient{:0>3}".format(i) for i in range(61, 81)]
        fold4_training_set = [
            i for i in all_cases_set if i not in fold4_testing_set]

        fold5_testing_set = [
            "patient{:0>3}".format(i) for i in range(81, 101)]
        fold5_training_set = [
            i for i in all_cases_set if i not in fold5_testing_set]
        if fold == "fold1":
            return [fold1_training_set, fold1_testing_set]
        elif fold == "fold2":
            return [fold2_training_set, fold2_testing_set]
        elif fold == "fold3":
            return [fold3_training_set, fold3_testing_set]
        elif fold == "fold4":
            return [fold4_training_set, fold4_testing_set]
        elif fold == "fold5":
            return [fold5_training_set, fold5_testing_set]
        elif fold == "MAAGfold":
            training_set = ["patient{:0>3}".format(i) for i in [37, 50, 53, 100, 38, 19, 61, 74, 97, 31, 91, 35, 56, 94, 26, 69, 46, 59, 4, 89,
                                71, 6, 52, 43, 45, 63, 93, 14, 98, 88, 21, 28, 99, 54, 90]]
            validation_set = ["patient{:0>3}".format(i) for i in [84, 32, 27, 96, 17, 18, 57, 81, 79, 22, 1, 44, 49, 25, 95]]
            return [training_set, validation_set]
        elif fold == "MAAGfold70":
            training_set = ["patient{:0>3}".format(i) for i in [37, 50, 53, 100, 38, 19, 61, 74, 97, 31, 91, 35, 56, 94, 26, 69, 46, 59, 4, 89,
                                71, 6, 52, 43, 45, 63, 93, 14, 98, 88, 21, 28, 99, 54, 90, 2, 76, 34, 85, 70, 86, 3, 8, 51, 40, 7, 13, 47, 55, 12, 58, 87, 9, 65, 62, 33, 42,
                               23, 92, 29, 11, 83, 68, 75, 67, 16, 48, 66, 20, 15]]
            validation_set = ["patient{:0>3}".format(i) for i in [84, 32, 27, 96, 17, 18, 57, 81, 79, 22, 1, 44, 49, 25, 95]]
            return [training_set, validation_set]
        elif "MAAGfold" in fold:
            training_num = int(fold[8:])
            training_set = random.sample(["patient{:0>3}".format(i) for i in [37, 50, 53, 100, 38, 19, 61, 74, 97, 31, 91, 35, 56, 94, 26, 69, 46, 59, 4, 89,
                                71, 6, 52, 43, 45, 63, 93, 14, 98, 88, 21, 28, 99, 54, 90, 2, 76, 34, 85, 70, 86, 3, 8, 51, 40, 7, 13, 47, 55, 12, 58, 87, 9, 65, 62, 33, 42,
                               23, 92, 29, 11, 83, 68, 75, 67, 16, 48, 66, 20, 15]], training_num)
            print("total {} training samples: {}".format(training_num, training_set))
            validation_set = ["patient{:0>3}".format(i) for i in [84, 32, 27, 96, 17, 18, 57, 81, 79, 22, 1, 44, 49, 25, 95]]
            return [training_set, validation_set]
        else:
            return "ERROR KEY"

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir +
                            "/ACDC_training_volumes/{}".format(case), 'r')
        elif self.split == "val":
            h5f = h5py.File(self._base_dir +
                            "/ACDC_training_volumes/{}".format(case), 'r')
        else:
            h5f = h5py.File(self._base_dir +
                            "/ACDC_testing_volumes/{}".format(case), 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        gt = h5f['label'][:]
        sample = {'image': image, 'label': label, 'gt': gt}
        # if self.split == "train":
        #     image = h5f['image'][:]
        #     label = h5f[self.sup_type][:]
        #     sample = {'image': image, 'label': label, 'gt': gt}
        #     if self.transform is not None:
        #         sample = self.transform(sample)
        # else:
        #     image = h5f['image'][:]
        #     label = h5f['label'][:]
        #     sample = {'image': image, 'label': label, 'gt': gt}
        sample["idx"] = idx
        return sample
    
def process(gt_data, image_data, device, sam_model, sam_transform):
    # if it is rgb, select the first channel
    if len(gt_data.shape) == 3:
        gt_data = gt_data[:, :, 0]
    assert len(gt_data.shape) == 2, "ground truth should be 2D"

    # resize ground truth image
    # resize_gt = sam_transform.apply_image(gt_data, interpolation=InterpolationMode.NEAREST) # ResizeLong (resized_h, 1024)
    # gt_data = sam_model.preprocess_for_gt(resize_gt)

    # exclude tiny objects (considering multi-object)
    gt = gt_data.copy()
    label_list = np.unique(gt_data)[1:]
    del_lab = [] # for check
    for label in label_list:
        gt_single = (gt_data == label) + 0
        if np.sum(gt_single) <= 50:
            gt[gt == label] = 0
            del_lab.append(label)
    assert len(list(np.unique(gt)) + del_lab) == len(list(label_list) + [0])

    if np.sum(gt) > 0: # has at least 1 object
        # gt: seperate each target into size (B, H, W) binary 0-1 uint8
        new_lab_list = list(np.unique(gt))[1:] # except bk
        new_lab_list.sort()
        gt_ = []
        for l in new_lab_list:
            gt_.append((gt == l) + 0)
        gt_ = np.array(gt_, dtype=np.uint8)

        image_ori_size = image_data.shape[:2]
        # Remove any alpha channel if present.
        if image_data.shape[-1] > 3 and len(image_data.shape) == 3:
            image_data = image_data[:, :, :3]
        # If image is grayscale, then repeat the last channel to convert to rgb
        if len(image_data.shape) == 2:
            image_data = np.repeat(image_data[:, :, None], 3, axis=-1)
        # nii preprocess start (clip the intensity)
        lower_bound, upper_bound = np.percentile(image_data, 0.95), np.percentile(
            image_data, 99.5 # Intensity of 0.95% pixels in image_data lower than lower_bound
                             # Intensity of 99.5% pixels in image_data lower than upper_bound
        )
        image_data_pre = np.clip(image_data, lower_bound, upper_bound)
        # min-max normalize and scale
        image_data_pre = (
            (image_data_pre - np.min(image_data_pre))
            / (np.max(image_data_pre) - np.min(image_data_pre))
            * 255.0
        )
        image_data_pre[image_data == 0] = 0 # ensure 0-255
        image_data_pre = np.uint8(image_data_pre)
        # print("image_data_pre shape:", image_data_pre.shape)
        ##############

        # Med4SAM preprocess
        # resize image to 3*1024*1024
        resize_img = sam_transform.apply_image(image_data_pre, interpolation=InterpolationMode.BILINEAR) # ResizeLong
        resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1))[None, :, :, :].to(device) # (1, 3, resized_h, 1024)
        resized_size_be_padding = tuple(resize_img_tensor.shape[-2:])
        input_image = sam_model.preprocess(resize_img_tensor) # padding to (1, 3, 1024, 1024)
        assert input_image.shape == (
            1,
            3,
            sam_model.image_encoder.img_size,
            sam_model.image_encoder.img_size,
        ), "input image should be resized to 1024*1024"
        assert input_image.shape[-2:] == (1024, 1024)
        # pre-compute the image embedding
        sam_model.eval()
        with torch.no_grad():
            embedding = sam_model.image_encoder(input_image)
            img_embedding = embedding.cpu().numpy()[0]
        return gt_, new_lab_list, img_embedding, resized_size_be_padding, image_ori_size, image_data_pre
    else:
        print("No any targets in the image")
        return None, None, None, None, None, None
    
def limit_rect(mask, box_ratio):
    """ check if the enlarged bounding box extends beyond the image. """
    height, width = mask.shape[0], mask.shape[1]
    # maximum bounding box
    box = find_box_from_mask(mask)
    w, h = box[2] - box[0], box[3] - box[1]
    w_ratio = w * box_ratio
    h_ratio = h * box_ratio
    x1 = box[0] - w_ratio/2 + w / 2
    y1 = box[1] - h_ratio/2 + h / 2
    x2 = x1 + w_ratio
    y2 = y1 + h_ratio
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2 >= width:
        x2 = width
    if y2 >= height:
        y2 = height
    return x1, y1, x2-x1, y2-y1
    
def find_center_from_mask_new(mask, box_ratio=2, n_fg=5, n_bg=5):
# def get_all_point_info(mask, box_ratio, n_fg, n_bg):
    """
    input:
        mask:     single mask
        bg_ratio: expand by a factor of bg_ratio based on the maximum bounding box
        n_fg:     foreground points number
        n_bg:     background points number
    Return:
        point_coords(ndarry): size=M*2, select M points(foreground or background)
        point_labels(ndarry): size=M 
    """
    # find barycenter
    M = cv2.moments(mask)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    center_point = np.array([cX, cY]).reshape(1, 2)

    # get foreground points
    indices_fg = np.where(mask == 1)
    points_fg = np.column_stack((indices_fg[1], indices_fg[0]))

    # uniformly sample n points
    step_fg = int(len(points_fg) / n_fg)
    # print(len(points_fg))
    points_fg = points_fg[::step_fg, :]
    

    # find the maximum bounding box
    x, y, w, h = limit_rect(mask, box_ratio)
    box1 = (x, y, x+w, y+h)
    x, y, w, h = int(x), int(y), int(w), int(h)

    # get background points
    yy, xx = np.meshgrid(np.arange(x, x+w), np.arange(y, y+h))
    roi = mask[y:y+h, x:x+w]
    bool_mask = roi == 0
    points_bg = np.column_stack((yy[bool_mask], xx[bool_mask]))

    # uniformly sample n points
    step_bg = int(len(points_bg) / n_bg)
    points_bg = points_bg[::step_bg, :]

    # get point_coords
    points_fg = np.concatenate((center_point, points_fg[1:]), axis=0)
    point_coords = np.concatenate((points_fg, points_bg), axis=0)
    point_labels = np.concatenate((np.ones(n_fg), np.zeros(n_bg)), axis=0)

    return point_coords, point_labels, points_fg, points_bg, box1, (cX, cY)
    
def find_box_from_mask(mask):
    y, x = np.where(mask == 1)
    x0 = x.min()
    x1 = x.max()
    y0 = y.min()
    y1 = y.max()
    return [x0, y0, x1, y1]
    
# get box and points information
def find_all_info(mask, label_list):
    point_list = []
    point_label_list = []
    mask_list = []
    box_list = []
    # multi-object processing
    for current_label_id in range(len(label_list)):
        current_mask = mask[current_label_id]
        current_center_point_list, current_label_list,_,_,_,_=  find_center_from_mask_new(current_mask)
        current_box = find_box_from_mask(current_mask)
        point_list.append(current_center_point_list[0:10,:])
        point_label_list.append(current_label_list[0:10,])
        mask_list.append(current_mask)
        box_list.append(current_box)
    return point_list, point_label_list, box_list, mask_list

def dice_coefficient(y_true, y_pred):
    """
    y_true: GT, [N, W, H]
    Y_pred: target, [M, W, H]
    N, M: number
    W, H: weight and height of the masks
    Returns:
        dice_matrix [N, M]
        dice_max_index [N,] indexes of prediceted masks with the highest DICE between each N GTs 
        dice_max_value [N,] N values of the highest DICE
    """
    smooth = 0.1


    y_true_f = y_true.reshape(y_true.shape[0], -1) # N
    
    y_pred_f = y_pred.reshape(y_pred.shape[0], -1) # M
    intersection = np.matmul(y_true_f.astype(float), y_pred_f.T.astype(float))
    dice_matrix = (2. * intersection + smooth) / (y_true_f.sum(1).reshape(y_true_f.shape[0],-1) + y_pred_f.sum(1) + smooth)
    dice_max_index, dice_max_value = dice_matrix.argmax(1), dice_matrix.max(1)
    return dice_matrix, dice_max_index, dice_max_value

def preprocess(args):
    # set up the model
    # get the model from sam_model_registry using the model_type argument
    # and load it with checkpoint argument
    # download save the SAM checkpoint.
    # [https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth](VIT-B SAM model)
    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint, device = args.device).to(
        args.device
    )
    # ResizeLongestSide (1024), including image and gt
    sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)

    imgs = 0
    save_path = os.path.join(args.npz_path, 'train', 'acdc')
    os.makedirs(save_path, exist_ok=True)
    print("Saving to:", save_path)

    acdc_dataset = ACDCDataset(
        base_dir=args.path,
        split="train",
        transform=None,
    )
    print("Num. of all train images:", len(acdc_dataset.sample_list))

    for sample in tqdm(acdc_dataset):
        # sample = {'image': image, 'label': label, 'gt': gt, 'idx': idx}
        image = sample['image']
        gt = sample['gt']
        idx = sample['idx']
        for slice_idx in range(gt.shape[0]):
            gt_slice = gt[slice_idx, :, :]
            image_slice = image[slice_idx, :, :]
            gt_name = acdc_dataset.sample_list[idx].split('.')[0] + "_" + str(slice_idx)
            if os.path.exists(os.path.join(save_path, gt_name + ".npz")):
                print("Already processed:", gt_name)
                continue
            gt_, new_lab_list, img_embedding, resized_size_be_padding, image_ori_size, img = process(gt_slice, image_slice, args.device, sam_model, sam_transform)
            if gt_ is not None:
                _, _, box_list, gt_list = find_all_info(gt_, new_lab_list)
                imgs += 1
                np.savez_compressed(
                    os.path.join(save_path, gt_name + ".npz"),
                    label_except_bk=new_lab_list,
                    gts=gt_,
                    img_embeddings=img_embedding,
                    img=img,
                    image_shape=image_ori_size,
                    resized_size_before_padding=resized_size_be_padding,
                    gt_list=gt_list,
                    box_list=box_list,
                )
    print("Num. of processed train images (delete images with no any targets):", imgs)
    return save_path

def SAM4Med(npz_data, predictor: SamPredictor, resize_transform, device, gt_name):
    gt2D = npz_data['gts']
    img_embed = torch.tensor(npz_data['img_embeddings']).float()

    # prompt mode
    predictor.original_size = tuple(npz_data['image_shape'])
    predictor.input_size = tuple(npz_data['resized_size_before_padding'])
    predictor.features = img_embed.unsqueeze(0).to(device)
    box_list = npz_data['box_list']
        
    # pre_process
    box_list_tensor = torch.tensor(box_list).float().to(device)
    box_list_tensor = resize_transform.apply_boxes_torch(box_list_tensor, (gt2D.shape[-2], gt2D.shape[-1]))
    if os.path.basename(gt_name) == "patient018_frame01_2.npz":  
        print(f"Resized boxes shape: {box_list_tensor.shape}, values: {box_list_tensor}")
        
    '''box'''
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes = box_list_tensor,
        multimask_output=False, # we set to false to match LiteMedSAM. M = 1
        return_logits=True  # uses threshold=0.0 which is equal to prob=0.5
        ) # Mask -> N,M,H,W
    
    masks = masks.squeeze(1).cpu().numpy()
    return masks

def LiteMedSAM(npz_data, medsam_lite_model, gt_name):
    img_3c = npz_data['img'] # (H, W, 3)
    assert np.max(img_3c)<256, f'input data should be in range [0, 255], but got {np.unique(img_3c)}'
    H, W = img_3c.shape[:2]
    boxes = npz_data['box_list']
    # print(f'{npz_name}, boxes: {boxes}')    # [[ 84  55 203 193]]
    segs = np.zeros((len(boxes), H, W), dtype=np.float32)

    ## preprocessing
    img_256 = resize_longest_side(img_3c, 256)
    newh, neww = img_256.shape[:2]
    # min-max normalize and scale
    img_256_norm = (img_256 - img_256.min()) / np.clip(
        img_256.max() - img_256.min(), a_min=1e-8, a_max=None
    )
    img_256_padded = pad_image(img_256_norm, 256)
    img_256_tensor = torch.tensor(img_256_padded).float().permute(2, 0, 1).unsqueeze(0).to(args.device)
    # save img_256_tensor for debugging
    if os.path.basename(gt_name) == "patient018_frame01_2.npz":  
        np.savez_compressed(f'test_img_256_tensor.npz', img_256_tensor=img_256_tensor.cpu().numpy())
        # load previously saved tensor
        img_256_tensor = np.load('test_img_256.npz')['img_256']
        img_256_tensor = torch.tensor(img_256_tensor).float().to(args.device)
        print(img_256_tensor.shape, img_256_tensor.dtype)

        # exit()
    else:
        return segs
    with torch.no_grad():
        image_embedding = medsam_lite_model.image_encoder(img_256_tensor)

    for idx, box in enumerate(boxes):
        box256 = resize_box_to_256(box, original_size=(H, W))
        box256 = box256[None, ...] # (1, 4)
        medsam_mask, iou_pred = medsam_inference(medsam_lite_model, image_embedding, box256, (newh, neww), (H, W), return_logits=True)
        segs[idx] = medsam_mask
        # print(f'{npz_name}, box: {box}, predicted iou: {np.round(iou_pred.item(), 4)}')
    
    # save segs for debugging
    if os.path.basename(gt_name) == "patient018_frame01_2.npz":
        save_path = "/Users/i/ICL/fusion/code/data/debug"
        for j, mask_filename in enumerate(range(len(boxes))):
            print(f"Saving mask logits for {mask_filename}...")
            mask_save_path = os.path.join(save_path, "ensemble", f"{mask_filename}_mask_logits.png")
            print(f"Saving mask logits to: {mask_save_path}")
            # np.save(mask_save_path, mask_logits[j].cpu().numpy())
            cv2.imwrite(mask_save_path, (segs[j]).astype(np.uint8))
    return segs

def ensemble_logits(sam4med_logits, medsam_lite_logits, method='mean'):
    """
    Ensembles raw logits from two models.

    Args:
        sam4med_logits (np.ndarray): Logits from SAM4Med, shape (num_boxes, H, W).
        medsam_lite_logits (np.ndarray): Logits from MedSAM-Lite, shape (num_boxes, H, W).
        method (str): Ensembling method ('mean', 'max', 'min').

    Returns:
        np.ndarray: Ensembled logits, shape (num_boxes, H, W).
    """
    if sam4med_logits.shape != medsam_lite_logits.shape:
        raise ValueError(f"Logit shapes must match for ensembling: {sam4med_logits.shape} vs {medsam_lite_logits.shape}")

    if method == 'mean':
        ensembled_logits = (sam4med_logits + medsam_lite_logits) / 2
    elif method == 'max':
        ensembled_logits = np.maximum(sam4med_logits, medsam_lite_logits)
    elif method == 'min':
        ensembled_logits = np.minimum(sam4med_logits, medsam_lite_logits)
    else:
        raise ValueError(f"Unsupported ensembling method: {method}. Choose 'mean', 'max', or 'min'.")

    return ensembled_logits

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preprocess grey and RGB images")
    parser.add_argument(
        "--path",
        type=str,
        default="/Users/i/ICL/fusion/code/data/ACDC/ACDC_preprocessed",
        help="path to the folder containing the images",
    )
    parser.add_argument(
        "-o",
        "--npz_path",
        type=str,
        default=f"data",
        help="path to save the npz files",
    )
    parser.add_argument(
        "--img_name_suffix", type=str, default=".png", help="image name suffix"
    )
    parser.add_argument("--model_type", type=str, default="vit_b", help="model type")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="SAM4Med/sam_vit_b_01ec64.pth",
        help="original sam checkpoint",
    )
    parser.add_argument("--device", type=str, default="mps:0", help="device")
    args = parser.parse_args()

    # PREPROCESS
    print("Preprocessing the data...")
    save_path = preprocess(args)
    print(save_path)

    # ensemble mode: the data contains all the info for both models
    
    # task_name
    task = "acdc"
    print("Current processing task: ", task)
    # model size
    size = "b"
    sam4med_checkpoint = f"SAM4Med/model/medsam_box_best_vitb.pth"
    lite_medsam_checkpoint = "MedSAM/work_dir/LiteMedSAM/lite_medsam.pth"
    json_info_path = f"SAM4Med/data_infor_json/{task}.json"
    model_type = f"vit_{size}"

    # SAM4Med
    sam = sam_model_registry[model_type](checkpoint=sam4med_checkpoint, device=args.device)
    sam.to(args.device)
    sam.eval()
    predictor = SamPredictor(sam)
    resize_transform = ResizeLongestSide(sam.image_encoder.img_size)

    info = json.load(open(json_info_path))
    label_info = info["info"]
    color_info = info["color"]
    all_dices = {}

    # save infer results (png: final mask, npy: pred_all_masks)
    # save selected prompts (npz)
    all_dices["box"] = {}
    dice_targets = [[] for _ in range(len(label_info))] # 某个方法中,记录不同结构的dice

    # LiteMedSAM
    medsam_lite_image_encoder = TinyViT(
    img_size=256,
    in_chans=3,
    embed_dims=[
        64, ## (64, 256, 256)
        128, ## (128, 128, 128)
        160, ## (160, 64, 64)
        320 ## (320, 64, 64) 
    ],
    depths=[2, 2, 6, 2],
    num_heads=[2, 4, 5, 10],
    window_sizes=[7, 7, 14, 7],
    mlp_ratio=4.,
    drop_rate=0.,
    drop_path_rate=0.0,
    use_checkpoint=False,
    mbconv_expand_ratio=4.0,
    local_conv_size=3,
    layer_lr_decay=0.8
    )

    medsam_lite_prompt_encoder = PromptEncoder(
        embed_dim=256,
        image_embedding_size=(64, 64),
        input_image_size=(256, 256),
        mask_in_chans=16
    )

    medsam_lite_mask_decoder = MaskDecoder(
        num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
    )

    medsam_lite_model = MedSAM_Lite(
        image_encoder = medsam_lite_image_encoder,
        mask_decoder = medsam_lite_mask_decoder,
        prompt_encoder = medsam_lite_prompt_encoder
    )

    # lite_medsam_checkpoint = torch.load("MedSAM/work_dir/LiteMedSAM/lite_medsam.pth", map_location='cpu')
    medsam_lite_model.load_state_dict(torch.load(lite_medsam_checkpoint, map_location='cpu'))
    medsam_lite_model.to(args.device)
    medsam_lite_model.eval()
    
    for ori_path in tqdm(glob.glob(os.path.join(save_path, "*.npz"))):
        name = ori_path.split('/')[-1].split('.')[0]
        npz_data = np.load(ori_path)

        # SAM4Med:
        sam4med_masks = SAM4Med(npz_data, predictor, resize_transform, args.device, ori_path)
        # print(f"sam4med_masks shape: {sam4med_masks.shape}")        # (3, 256, 216)
        # print(f"sam4med_masks: {sam4med_masks}")
        gt2D = npz_data['gts']
        label_list = npz_data['label_except_bk'].tolist()
        # img_embed = torch.tensor(npz_data['img_embeddings']).float()

        # # prompt mode
        # predictor.original_size = tuple(npz_data['image_shape'])
        # predictor.input_size = tuple(npz_data['resized_size_before_padding'])
        # predictor.features = img_embed.unsqueeze(0).to(args.device)
        # box_list = npz_data['box_list']
        gt_list = npz_data['gt_list']
            
        # # pre_process
        # print("box_list: ", box_list)
        # box_list_tensor = torch.tensor(box_list).float().to(args.device)
        # box_list_tensor = resize_transform.apply_boxes_torch(box_list_tensor, (gt2D.shape[-2], gt2D.shape[-1]))
            
        # '''box'''
        # masks, scores, logits = predictor.predict_torch(
        #     point_coords=None,
        #     point_labels=None,
        #     boxes = box_list_tensor,
        #     multimask_output=True,
        #     ) # Mask -> N,M,H,W
        
        # masks = masks.cpu().numpy()

        # LiteMedSAM:
        medsam_masks = LiteMedSAM(npz_data, medsam_lite_model, ori_path)
        # img_3c = npz_data['img'] # (H, W, 3)
        # assert np.max(img_3c)<256, f'input data should be in range [0, 255], but got {np.unique(img_3c)}'
        # H, W = img_3c.shape[:2]
        # boxes = npz_data['box_list']
        # # print(f'{npz_name}, boxes: {boxes}')    # [[ 84  55 203 193]]
        # segs = np.zeros(img_3c.shape[:2], dtype=np.uint8)

        # ## preprocessing
        # img_256 = resize_longest_side(img_3c, 256)
        # newh, neww = img_256.shape[:2]
        # # min-max normalize and scale
        # img_256_norm = (img_256 - img_256.min()) / np.clip(
        #     img_256.max() - img_256.min(), a_min=1e-8, a_max=None
        # )
        # img_256_padded = pad_image(img_256_norm, 256)
        # img_256_tensor = torch.tensor(img_256_padded).float().permute(2, 0, 1).unsqueeze(0).to(args.device)
        # with torch.no_grad():
        #     image_embedding = medsam_lite_model.image_encoder(img_256_tensor)

        # for idx, box in enumerate(boxes, start=1):
        #     box256 = resize_box_to_256(box, original_size=(H, W))
        #     box256 = box256[None, ...] # (1, 4)
        #     medsam_mask, iou_pred = medsam_inference(medsam_lite_model, image_embedding, box256, (newh, neww), (H, W))
        #     segs[medsam_mask>0] = idx
        #     # print(f'{npz_name}, box: {box}, predicted iou: {np.round(iou_pred.item(), 4)}')
        
        # combine the two models
        # masks = np.zeros((sam4med_masks.shape[0], gt2D.shape[1], gt2D.shape[2]), dtype=np.uint8)
        ensembled_logits = ensemble_logits(sam4med_masks, medsam_masks, method='min')
        masks = (ensembled_logits > 0).astype(np.uint8)
        # print(masks.shape) # (3, 256, 216)

        # compute dice
        current_method_res = np.zeros(gt2D.shape[1:]) # all target in a single image
        for idx in range(len(gt_list)): # mask list for a single image
            current_gt = gt_list[idx]
            dice_matrix, dice_max_index, dice_max_value = dice_coefficient(y_true=current_gt[None, :, :], y_pred=masks[idx][None, :, :])
            final_mask = masks[idx]#[dice_max_index.squeeze(0)]
            
            # index mapping for matching DICE with different target structures
            try:
                id_dice = int(
                    list(color_info.keys())[list(color_info.values()).index(label_list[idx])]
                    )
                dice_targets[id_dice - 1].append(dice_max_value.squeeze(0))
                
                current_method_res[final_mask == 1] = label_list[idx] # one target
            except IndexError as e:
                print(e)
                print(f"error: {ori_path}")
                continue

        # for visulize (save infer results)
        if not os.path.exists(os.path.join(args.npz_path, "seg", "acdc", ori_path.split('/')[-1])):
            cv2.imwrite(os.path.join(args.npz_path, "seg", "acdc", ori_path.split('/')[-1].split(".")[0] + ".png"), current_method_res.astype(np.uint8))

    # print
    for id in range(len(dice_targets)):
        all_dices["box"][label_info[str(id + 1)]] = f'{round(np.array(dice_targets[id]).mean() * 100, 2)}({round((100 * np.array(dice_targets[id])).std(), 2)})'
    
    print("======following is the dice results======")
    print(json.dumps(all_dices, indent=4, ensure_ascii=False))


