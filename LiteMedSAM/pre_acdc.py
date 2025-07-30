import argparse
import os
import re
from random import sample
from tqdm import tqdm
import numpy as np
import cv2

from torch.utils.data import Dataset
import h5py

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
    
def process(gt_data, image_data):
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
        return gt_, new_lab_list, image_data_pre, image_ori_size
    else:
        print("No any targets in the image")
        return None, None, None, None
    
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
    args = parser.parse_args()

    # create a directory to save the npz files
    # get all the names of the images in the ground truth folder
    imgs = 0
    save_path = os.path.join(args.npz_path, 'test', 'acdc')
    os.makedirs(save_path, exist_ok=True)
    print("Saving to:", save_path)

    acdc_dataset = ACDCDataset(
        base_dir=args.path,
        split="test",
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
            gt_, new_lab_list, img, image_ori_size = process(gt_slice, image_slice)
            if gt_ is not None:
                _, _, box_list, gt_list = find_all_info(gt_, new_lab_list)
                imgs += 1
                np.savez_compressed(
                    os.path.join(save_path, gt_name + ".npz"),
                    label_except_bk=new_lab_list,
                    gts=gt_,
                    img=img,
                    image_shape=image_ori_size,
                    gt_list=gt_list,
                    box_list=box_list,
                )
    print("Num. of processed train images (delete images with no any targets):", imgs)

