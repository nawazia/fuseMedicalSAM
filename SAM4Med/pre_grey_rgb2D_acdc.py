# %% import packages
import numpy as np
import os

join = os.path.join
from skimage import io
import cv2
from tqdm import tqdm
import torch
from torchvision.transforms.functional import InterpolationMode
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import argparse

import re
import h5py
import numpy as np
from torch.utils.data import Dataset
from random import sample

class ACDCDataset(Dataset):
    def __init__(self, base_dir=None, split='train', transform=None, fold="fold1", sup_type="label"):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.sup_type = sup_type
        self.transform = transform
        if self.split == "train" or self.split == "val":
            train_ids, test_ids = self._get_fold_ids(fold)
        if self.split == 'train':
            self.all_slices = os.listdir(
                self._base_dir + "/ACDC_training_slices")
            self.sample_list = []
            for ids in train_ids:
                new_data_list = list(filter(lambda x: re.match(
                    '{}.*'.format(ids), x) != None, self.all_slices))
                self.sample_list.extend(new_data_list)
        elif self.split == 'val':
            self.all_volumes = os.listdir(
                self._base_dir + "/ACDC_training_volumes")
            self.sample_list = []
            for ids in test_ids:
                new_data_list = list(filter(lambda x: re.match(
                    '{}.*'.format(ids), x) != None, self.all_volumes))
                self.sample_list.extend(new_data_list)
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
            training_set = sample(["patient{:0>3}".format(i) for i in [37, 50, 53, 100, 38, 19, 61, 74, 97, 31, 91, 35, 56, 94, 26, 69, 46, 59, 4, 89,
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
                            "/ACDC_training_slices/{}".format(case), 'r')
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
        if self.split == "train":
            image = h5f['image'][:]
            if self.sup_type == "random_walker":
                label = pseudo_label_generator_acdc(image, h5f["scribble"][:])
            else:
                label = h5f[self.sup_type][:]
            sample = {'image': image, 'label': label, 'gt': gt}
            if self.transform is not None:
                sample = self.transform(sample)
        else:
            image = h5f['image'][:]
            label = h5f['label'][:]
            sample = {'image': image, 'label': label, 'gt': gt}
        sample["idx"] = idx
        return sample

def process(gt_data, image_data, device):
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
        return gt_, new_lab_list, img_embedding, resized_size_be_padding, image_ori_size
    else:
        print("No any targets in the image")
        return None, None, None, None, None

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
        default="sam_vit_b_01ec64.pth",
        help="original sam checkpoint",
    )
    parser.add_argument("--device", type=str, default="mps:0", help="device")
    args = parser.parse_args()

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

    # create a directory to save the npz files
    save_base = args.npz_path + "/precompute_" + args.model_type
    imgs = 0
    # get all the names of the images in the ground truth folder
    acdc_dataset = ACDCDataset(
        base_dir=args.path,
        split="test",
        transform=None,
    )
    save_path = join(save_base, 'test', 'acdc')
    print("Saving to:", save_path)
    print("Num. of all train images:", len(acdc_dataset.sample_list))
    
    os.makedirs(save_path, exist_ok=True)
    for sample in tqdm(acdc_dataset):
        # sample = {'image': image, 'label': label, 'gt': gt, 'idx': idx}
        image = sample['image']
        gt = sample['gt']
        idx = sample['idx']
        for slice_idx in range(gt.shape[0]):
            gt_slice = gt[slice_idx, :, :]
            image_slice = image[slice_idx, :, :]
            gt_name = acdc_dataset.sample_list[idx].split('.')[0] + "_" + str(slice_idx)
            if os.path.exists(join(save_path, gt_name + ".npz")):
                print("Already processed:", gt_name)
                continue
            cv2.imwrite(join(save_path, gt_name + ".png"), ((image_slice * 255).astype(np.uint8)))
            gt_, new_lab_list, img_embedding, resized_size_be_padding, image_ori_size,  = process(gt_slice, image_slice, args.device)
            if gt_ is not None:
                imgs += 1
                np.savez_compressed(
                    join(save_path, gt_name + ".npz"),
                    label_except_bk=new_lab_list,
                    gts=gt_,
                    img_embeddings=img_embedding,
                    image_shape=image_ori_size,
                    resized_size_before_padding=resized_size_be_padding
                )
    print("Num. of processed train images (delete images with no any targets):", imgs)
