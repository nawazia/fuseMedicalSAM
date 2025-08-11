import json
import os
import copy
import io

import numpy as np
import cv2
from torch.utils.data import Dataset
import torch
import albumentations as A
from google.cloud import storage
from albumentations.core.transforms_interface import ImageOnlyTransform


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
    return (x0, y0, x1, y1)

class ClipNorm(ImageOnlyTransform):
    """
    Custom intensity normalization that:
    1. Clips intensities between 0.95 and 99.5 percentiles
    2. Scales to 0-255 range
    3. Ensures proper uint8 dtype
    """
    def __init__(self, p=1.0):
        super().__init__(p=p)
    
    def apply(self, img, **params):
        # Calculate percentiles
        lower_bound, upper_bound = np.percentile(img, 0.95), np.percentile(img, 99.5)
        
        # Clip and normalize
        img = np.clip(img, lower_bound, upper_bound)
        img = ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255.0)
        return img.astype(np.uint8)
    
    def get_transform_init_args_names(self):
        return ()
    
class PadBottomRightOnly(A.DualTransform):
    """
    Custom Albumentations transform to pad an image/mask only on the bottom and right
    to reach a specified target square size. If a dimension is already larger than
    the target size, no padding will be applied to that dimension.
    This transform explicitly does NOT resize images.
    """
    def __init__(self, target_size: int, border_mode: int = cv2.BORDER_CONSTANT, value: int = 0, mask_value: int = 0, p: float = 1.0):
        """
        Args:
            target_size (int): The target minimum square size (height and width) for the output.
                               If a dimension is smaller than target_size, it will be padded.
                               If a dimension is larger, it will remain unchanged.
            border_mode (int): Border mode for padding (e.g., cv2.BORDER_CONSTANT for constant border).
            value (int/float): Pixel value for padding images (e.g., 0 for black).
            mask_value (int/float): Pixel value for padding masks (e.g., 0 for background label).
            p (float): Probability of applying the transform. Default: 1.0.
        """
        super().__init__(p=p)
        self.target_size = target_size
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value

    @property
    def targets_as_dict(self):
        # This tells Albumentations how to apply this transform to different data types
        return {"image": self.apply, "mask": self.apply_to_mask, "bboxes": self.apply_to_bboxes}

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """Applies the padding transformation to the image."""
        h, w = img.shape[:2]

        # Calculate padding needed. Use max(0, ...) to ensure no negative padding
        # if the image dimension is already larger than target_size.
        pad_bottom = max(0, self.target_size - h)
        pad_right = max(0, self.target_size - w)

        if pad_bottom > 0 or pad_right > 0:
            # Only apply padding if any padding is actually needed
            return cv2.copyMakeBorder(
                img, 0, pad_bottom, 0, pad_right, # top, bottom, left, right
                self.border_mode, value=self.value
            )
        else:
            # If no padding is needed (dimensions are >= target_size), return original image
            return img

    def apply_to_mask(self, mask: np.ndarray, **params) -> np.ndarray:
        """Applies the padding transformation to the mask."""
        h, w = mask.shape[:2]

        pad_bottom = max(0, self.target_size - h)
        pad_right = max(0, self.target_size - w)

        if pad_bottom > 0 or pad_right > 0:
            return cv2.copyMakeBorder(
                mask, 0, pad_bottom, 0, pad_right,
                self.border_mode, value=self.mask_value # Use mask_value for masks
            )
        else:
            return mask
        
    def apply_to_bboxes(self, bbox: tuple, **params) -> tuple:
        # For padding added to the bottom and right, where the top-left corner (0,0) is fixed,
        # the absolute pixel coordinates of the bounding box do not change.
        # Albumentations' internal mechanism will handle the context of the new image dimensions
        # for subsequent transforms.
        return bbox
        

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        # Required for serialization of the transform
        return ("target_size", "border_mode", "value", "mask_value")
    
class CustomNormalize(ImageOnlyTransform):
    def __init__(self, mean, std, max_pixel_value=255.0, p=1.0):
        super().__init__(p)
        self.mean = mean
        self.std = std

    @property
    def targets(self):
        """
        Defines the mapping from target names (e.g., 'image') to the method
        that applies the transform to that target.
        For BasicTransform, it typically only includes 'image'.
        """
        return {"image": self.apply}

    def apply(self, img, **params):
        normalized_img = (img - self.mean) / self.std

        return normalized_img.astype(np.float32)

    def get_transform_init_args_names(self):
        return ("mean", "std", "max_pixel_value")

class ConditionalPadOrResize(A.DualTransform):
    def __init__(self, target_size: int, resize_interpolation: int = cv2.INTER_NEAREST,
                 pad_border_mode: int = cv2.BORDER_CONSTANT, pad_value: int = 0,
                 pad_mask_value: int = 0, p: float = 1.0):
        super().__init__(p=p)
        self.target_size = target_size
        self.resize_interpolation = resize_interpolation
        self.pad_border_mode = pad_border_mode
        self.pad_value = pad_value
        self.pad_mask_value = pad_mask_value

        # Initialize the transforms once
        self.pad_transform = A.PadIfNeeded(
            min_height=target_size,
            min_width=target_size,
            border_mode=pad_border_mode,
            fill=pad_value,
            fill_mask=pad_value,
            p=1.0
        )
        self.resize_transform = A.Resize(
            height=target_size,
            width=target_size,
            interpolation=resize_interpolation,
            p=1.0
        )
        self.mask_resize_transform = A.Resize(
            height=target_size,
            width=target_size,
            interpolation=cv2.INTER_NEAREST,
            p=1.0
        )

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        h, w = img.shape[:2]
        if h < self.target_size and w < self.target_size:
            return self.pad_transform(image=img)['image']
        else:
            return self.resize_transform(image=img)['image']

    def apply_to_mask(self, mask: np.ndarray, **params) -> np.ndarray:
        h, w = mask.shape[:2]
        if h < self.target_size and w < self.target_size:
            # For masks, we need to use the mask_value parameter
            pad_params = {
                'min_height': self.target_size,
                'min_width': self.target_size,
                'border_mode': self.pad_border_mode,
                'fill': self.pad_mask_value,
                'fill_mask': self.pad_mask_value,
                'p': 1.0
            }
            return A.PadIfNeeded(**pad_params)(image=mask)['image']
        else:
            return self.mask_resize_transform(image=mask)['image']

    def get_transform_init_args_names(self):
        return ("target_size", "resize_interpolation", "pad_border_mode", "pad_value", "pad_mask_value")

class MiniMSAMDataset(Dataset):
    def __init__(self, data_path, json_path, split="train"):
        """
        Args:
            data_path (str): Path to the directory containing images and masks folder.
            json_path (str): Path to the JSON file containing image-mask pairs.
        """
        self.data_path = data_path
        self.json_path = json_path
        with open(self.json_path, 'r') as file:
            data = json.load(file)
        self.data = data[split]
        self.image_paths = list(self.data.keys())
        self.num_masks = 0 # add num of masks from data dict
        for _, masks in self.data.items():
            self.num_masks += len(masks)
        self.simple = False
        self.fused_path = None
        self.debug = False

    def set_transforms(self, model : str = None):
        """ Set the transformation pipeline based on the model type.
        Args:
            model (str): Model type to determine the transformation pipeline.
        """
        self.model = model

        if model is not None and model not in ["MedSAM", "SAM4Med", "SAM-Med2D"]:#, "Med-SA"]:
            raise ValueError(f"Default model type '{model}' not in model_types")
        
        transforms = []
        
        if model == "MedSAM":
            transforms.extend([
                A.Resize(1024, 1024, cv2.INTER_CUBIC),
                A.Normalize(normalization="min_max", p=1.0),
            ])
        if model == "LiteMedSAM":
            transforms.extend([ClipNorm(p=1.0),
                A.Normalize(normalization="min_max", p=1.0),
                A.LongestMaxSize(256),
                PadBottomRightOnly(target_size=256, p=1.0)])
        elif model == "SAM4Med":
            transforms.extend(
                [ClipNorm(p=1.0),
                A.LongestMaxSize(1024),
                CustomNormalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], max_pixel_value=255.0, p=1.0),
                PadBottomRightOnly(target_size=1024, p=1.0)])
        elif model == "SAM-Med2D":
            transforms.extend([
                CustomNormalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], max_pixel_value=255.0, p=1.0),
                # train_transforms() here
                ConditionalPadOrResize(
                    target_size=256,
                    resize_interpolation=cv2.INTER_NEAREST,
                    pad_border_mode=cv2.BORDER_CONSTANT,
                    pad_value=0,
                    pad_mask_value=0,
                    p=1.0
                ),
                ])

        transforms.append(A.ToTensorV2(p=1.0))  # Always convert to tensor at the end
        self.transform = A.Compose(
            transforms,
            p=1.0)

    def __len__(self):
        return len(self.data)

    def get_num_masks(self):
        return self.num_masks
    
    def set_simple(self, mode : bool):
        self.simple = mode

    def set_fused(self, path : str):
        self.fused_path = path
    
    def set_debug(self, mode : bool):
        self.debug = mode

    def __getitem__(self, idx):
        image_filename = self.image_paths[idx]
        mask_filenames = self.data[image_filename]
        if self.debug:
            print("image_filename:", image_filename)
        if self.simple:
            sample = {
                'image_filename': image_filename,
                'mask_filenames': mask_filenames
            }
            return sample
        image_path_full = os.path.join(self.data_path, image_filename)
        mask_paths_full = [os.path.join(self.data_path, mask_filename) for mask_filename in mask_filenames]

        # Load the image
        img = cv2.imread(image_path_full, cv2.IMREAD_UNCHANGED)
        original_size = img.shape[:2]  # (height, width)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        assert img.shape[2] == 3, "Image must have 3 channels (RGB)"

        # Load the masks, assuming they're grayscale
        masks = []
        for mask_path in mask_paths_full:
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            mask = (mask > 0).astype(np.uint8)
            masks.append(mask)
            assert len(mask.shape) == 2, "Mask must be a single-channel image"
            assert mask.shape[:2] == img.shape[:2], "Mask shape {} does not match image shape: {}".format(mask.shape, img.shape)
        
        original_masks = np.array(masks)
        transformed_data = self.transform(image=img, masks=masks)
        img = transformed_data['image']
        # print(img.shape, img.min(), img.max(), img.mean(), img.std())
        new_size = list(img.shape[-2:])

        # find pre-pad size
        scale = max(new_size) * 1.0 / max(original_size)
        newh, neww = original_size[0] * scale, original_size[1] * scale
        newh, neww = int(newh + 0.5), int(neww + 0.5)

        masks = transformed_data['masks']

        boxes = []
        for mask in masks:
            boxes.append(find_box_from_mask(mask.numpy()))
        boxes = torch.tensor(boxes, dtype=torch.float32)

        sample = {
            'image': img,
            'boxes': boxes,
            'masks': masks,
            'original_masks': original_masks,
            'original_size': original_size,
            'prepad_size': (newh, neww),
            'image_filename': image_filename,
            'mask_filenames': mask_filenames
        }

        if self.fused_path:
            fused_paths_full = [os.path.join(self.fused_path, os.path.basename(mask_filename)[:-4]+"_mask_logits.npz") for mask_filename in mask_filenames]
            all_mask_logits = []
            for npz_path in fused_paths_full:
                with np.load(npz_path) as data:
                    mask_logits = data['mask_logits']
                    all_mask_logits.append(mask_logits)
            # The final shape will be (num_masks, H, W)
            stacked_logits = torch.from_numpy(np.stack(all_mask_logits, axis=0)).float()
            sample['teacher_logits'] = stacked_logits
        return sample
    
class MiniMSAMDatasetGCS(Dataset):
    def __init__(self, gcs_bucket_name, data_path, json_path, split="train"):
        """
        Args:
            gcs_bucket_name (str): The name of your GCS bucket.
            json_path (str): Path to the JSON file containing image-mask pairs.
        """
        # --- GCS client setup ---
        # The client will use the credentials authenticated in Colab
        self.client = storage.Client()
        self.bucket = self.client.bucket(gcs_bucket_name)
        self.data_path = data_path
        self.json_path = json_path
        blob = self.bucket.blob(json_path)
        json_data_bytes = blob.download_as_bytes()
        json_data = json.loads(json_data_bytes)
        
        self.data = json_data[split]
        self.image_paths = list(self.data.keys())
        self.num_masks = sum(len(masks) for masks in self.data.values())
        self.simple = False
        self.fused_path = None  # Note: fused_path would also need to be a GCS path
        self.debug = False

    def set_transforms(self, model : str = None):
        """ Set the transformation pipeline based on the model type.
        Args:
            model (str): Model type to determine the transformation pipeline.
        """
        self.model = model

        if model is not None and model not in ["MedSAM", "SAM4Med", "SAM-Med2D"]:#, "Med-SA"]:
            raise ValueError(f"Default model type '{model}' not in model_types")
        
        transforms = []
        
        if model == "MedSAM":
            transforms.extend([
                A.Resize(1024, 1024, cv2.INTER_CUBIC),
                A.Normalize(normalization="min_max", p=1.0),
            ])
        if model == "LiteMedSAM":
            transforms.extend([ClipNorm(p=1.0),
                A.Normalize(normalization="min_max", p=1.0),
                A.LongestMaxSize(256),
                PadBottomRightOnly(target_size=256, p=1.0)])
        elif model == "SAM4Med":
            transforms.extend(
                [ClipNorm(p=1.0),
                A.LongestMaxSize(1024),
                CustomNormalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], max_pixel_value=255.0, p=1.0),
                PadBottomRightOnly(target_size=1024, p=1.0)])
        elif model == "SAM-Med2D":
            transforms.extend([
                CustomNormalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], max_pixel_value=255.0, p=1.0),
                # train_transforms() here
                ConditionalPadOrResize(
                    target_size=256,
                    resize_interpolation=cv2.INTER_NEAREST,
                    pad_border_mode=cv2.BORDER_CONSTANT,
                    pad_value=0,
                    pad_mask_value=0,
                    p=1.0
                ),
                ])

        transforms.append(A.ToTensorV2(p=1.0))  # Always convert to tensor at the end
        self.transform = A.Compose(
            transforms,
            p=1.0)

    def __len__(self):
        return len(self.data)

    def get_num_masks(self):
        return self.num_masks
    
    def set_simple(self, mode : bool):
        self.simple = mode

    def set_fused(self, path : str):
        self.fused_path = path

    def set_debug(self, mode : bool):
        self.debug = mode

    def _read_image_from_gcs(self, blob_name):
        """Helper to read an image from GCS and return it as a numpy array."""
        blob = self.bucket.blob(blob_name)
        image_bytes = blob.download_as_bytes()
        # Decode the bytes into a numpy array
        image_array = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)
        return img

    def __getitem__(self, idx):
        image_filename = self.image_paths[idx]
        mask_filenames = self.data[image_filename]
        if self.debug:
            print("image_filename:", image_filename)
        if self.simple:
            sample = {
                'image_filename': image_filename,
                'mask_filenames': mask_filenames
            }
            return sample
        image_path_full = os.path.join(self.data_path, image_filename)
        mask_paths_full = [os.path.join(self.data_path, mask_filename) for mask_filename in mask_filenames]

        # Load the image
        # img = cv2.imread(image_path_full, cv2.IMREAD_UNCHANGED)
        img = self._read_image_from_gcs(image_path_full)
        original_size = img.shape[:2]  # (height, width)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        assert img.shape[2] == 3, "Image must have 3 channels (RGB)"

        # Load the masks, assuming they're grayscale
        masks = []
        for mask_path in mask_paths_full:
            # mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            mask = self._read_image_from_gcs(mask_path)
            mask = (mask > 0).astype(np.uint8)
            masks.append(mask)
            assert len(mask.shape) == 2, "Mask must be a single-channel image"
            assert mask.shape[:2] == img.shape[:2], "Mask shape {} does not match image shape: {}".format(mask.shape, img.shape)
        
        original_masks = np.array(masks)
        transformed_data = self.transform(image=img, masks=masks)
        img = transformed_data['image']
        # print(img.shape, img.min(), img.max(), img.mean(), img.std())
        new_size = list(img.shape[-2:])

        # find pre-pad size
        scale = max(new_size) * 1.0 / max(original_size)
        newh, neww = original_size[0] * scale, original_size[1] * scale
        newh, neww = int(newh + 0.5), int(neww + 0.5)

        masks = transformed_data['masks']

        boxes = []
        for mask in masks:
            boxes.append(find_box_from_mask(mask.numpy()))
        boxes = torch.tensor(boxes, dtype=torch.float32)

        sample = {
            'image': img,
            'boxes': boxes,
            'masks': masks,
            'original_masks': original_masks,
            'original_size': original_size,
            'prepad_size': (newh, neww),
            'image_filename': image_filename,
            'mask_filenames': mask_filenames
        }

        if self.fused_path:
            all_mask_logits = []
            for mask_filename in mask_filenames:
                npz_blob_name = os.path.join(self.data_path, self.fused_path, os.path.basename(mask_filename)[:-4]+"_mask_logits.npz")
                blob = self.bucket.blob(npz_blob_name)
                with blob.open("rb") as f:
                    npz_data = np.load(io.BytesIO(f.read()))
                    mask_logits = npz_data['mask_logits']
                    all_mask_logits.append(mask_logits)
            # The final shape will be (num_masks, H, W)
            try:
                stacked_logits = torch.from_numpy(np.stack(all_mask_logits, axis=0)).float()
            except:
                print(image_filename)
                stacked_logits = None
            sample['teacher_logits'] = stacked_logits
        return sample

if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader
    parser = argparse.ArgumentParser(description="Custom dataset loader.")
    parser.add_argument("--data_path", required=True, help="Path to dir containing images and masks folders")
    parser.add_argument("--json_path", required=True, help="Path to JSON files containing image-mask pairs.")
    
    args = parser.parse_args()
    
    dataset = MiniMSAMDataset(data_path=args.data_path, json_path=args.json_path)
    print(f"Dataset length: {len(dataset)}")
    dataset.set_transforms(model="MedSAM")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    for data in dataloader:
        image = data['image']
        print(image.shape)
        boxes = data['boxes']
        [print(box) for box in boxes]
        masks = data['masks']
        [print(mask.shape) for mask in masks]
        image_filename = data['image_filename']
        print(image_filename)
        mask_filenames = data['mask_filenames']
        print(mask_filenames)
        original_size = data["original_size"]
        print(original_size)
        original_masks = data["original_masks"]
        # print(original_masks)
        for mask in original_masks:
            print(mask.shape)
            # print(find_box_from_mask(mask.squeeze().numpy()))
        break
