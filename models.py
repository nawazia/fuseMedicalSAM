import numpy as np
import torch
from torch import nn
from torch.nn.functional import interpolate

from LiteMedSAM.segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from LiteMedSAM.tiny_vit_sam import TinyViT
from LiteMedSAM.medsam_lite import _MedSAM_Lite, litemedsam_inference

from MedSAM.segment_anything import sam_model_registry_medsam
from MedSAM.medsam import medsam_inference

from SAM4Med.segment_anything import sam_model_registry_sam4med, SamPredictor

from SAM_Med2D.segment_anything import sam_model_registry_sam_med2d
from SAM_Med2D.segment_anything.modeling.sam_model import Sam

class LiteMedSAM(nn.Module):
    def __init__(self, medsam_lite : _MedSAM_Lite):
        super().__init__()
        self.medsam_lite = medsam_lite
        self.inference = litemedsam_inference

    def forward(self, data):
        img = data["image"]
        boxes = data["boxes"][0]
        H, W = data["original_size"]
        newh, neww = data["prepad_size"]
        segs = np.zeros((len(boxes), H, W), dtype=np.float32)

        with torch.no_grad():
            image_embedding = self.medsam_lite.image_encoder(img.float())

        for idx, box in enumerate(boxes):
            medsam_mask, iou_pred = self.inference(self.medsam_lite, image_embedding, box, (newh, neww), (H, W), return_logits=True)
            segs[idx] = medsam_mask

        return segs, iou_pred
    
class MedSAM(nn.Module):
    def __init__(self, model_type: str = "vit_b", image_size : int = 1024, sam_checkpoint: str = None):
        super().__init__()
        self.inference = medsam_inference
        self.image_size = image_size
        self.medsam = sam_model_registry_medsam[model_type](checkpoint=sam_checkpoint)
        # self.medsam.eval()

    def forward(self, data):
        img = data["image"]
        boxes = data["boxes"][0]
        H, W = data["original_size"]
        # newh, neww = data["prepad_size"]
        # segs = np.zeros((len(boxes), H, W), dtype=np.float32)

        with torch.no_grad():
            image_embedding = self.medsam.image_encoder(img.float())

        if len(boxes.shape) == 2:
            boxes = boxes[:, None, :]  # (B, 1, 4)

        medsam_mask, iou_pred = self.inference(self.medsam, image_embedding, boxes, H, W, return_logits=True)
        return medsam_mask, iou_pred
    
    
class SAM4Med(nn.Module):
    def __init__(self, predictor : SamPredictor, model_type: str = "vit_b", checkpoint: str = None):
        super().__init__()
        self.sam4med = predictor
        self.sam_model = sam_model_registry_sam4med[model_type](checkpoint=checkpoint, device="cpu").to(predictor.device)
        # self.sam_model.eval()

    def forward(self, data):
        original_size = data["original_size"]
        prepad_size = data["prepad_size"]
        input_image = data["image"]
        boxes = data["boxes"].squeeze().float()
        
        assert input_image.shape == (
            1,
            3,
            self.sam_model.image_encoder.img_size,
            self.sam_model.image_encoder.img_size,
        ), "input image should be resized to 1024*1024"
        assert input_image.shape[-2:] == (1024, 1024)
        # pre-compute the image embedding
        # with torch.no_grad():
        embedding = self.sam_model.image_encoder(input_image)
        
        self.sam4med.original_size = tuple(original_size)  # Set original size for the predictor
        self.sam4med.input_size = prepad_size  # Set input size for the prepad
        self.sam4med.features = embedding.float()

        if len(boxes.shape) == 1:
            boxes = boxes.unsqueeze(0)
            
        '''box'''
        masks, iou_predictions, _ = self.sam4med.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes = boxes,
            multimask_output=False, # we set to false to match LiteMedSAM. M = 1
            return_logits=True  # if False, uses threshold=0.0 which is equal to prob=0.5
            ) # Mask -> N,M,H,W
        
        # masks = masks.squeeze(1).cpu().numpy()
        return masks, iou_predictions
    
class SAM_Med2D(nn.Module):
    def __init__(self, model_type: str = "vit_b", image_size : int = 256, sam_checkpoint: str = None, encoder_adapter: bool = True):
        super().__init__()
        self.image_size = image_size
        self.model : Sam = sam_model_registry_sam_med2d[model_type](image_size, sam_checkpoint, encoder_adapter)
        # self.model.eval()
    
    def forward(self, data):
        image = data["image"].float()
        boxes = data["boxes"][0].float()
        # add one to x2, y2 to replicate og code
        # lazy fix, needs to revisited
        boxes[:, -2:] = boxes[:, -2:] + 1
        original_size = data["original_size"]
        
        # with torch.no_grad():
        image_embeddings = self.model.image_encoder(image)

        # masks, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings)
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=None,
            boxes=boxes,
            masks=None,
        )

        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings = image_embeddings,
            image_pe = self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        
        # interp masks to 256
        masks = interpolate(low_res_masks, (self.image_size, self.image_size), mode="bilinear", align_corners=False,)
        # interp masks to original size
        # masks, pad = postprocess_masks(low_res_masks, self.image_size, original_size)
        ori_h, ori_w = original_size
        if ori_h < self.image_size and ori_w < self.image_size:
            top = torch.div((self.image_size - ori_h), 2, rounding_mode='trunc')  #(self.image_size - ori_h) // 2
            left = torch.div((self.image_size - ori_w), 2, rounding_mode='trunc') #(self.image_size - ori_w) // 2
            masks = masks[..., top : ori_h + top, left : ori_w + left]
        else:
            masks = interpolate(masks, original_size, mode="bilinear", align_corners=False)

        # masks = masks.squeeze(1).cpu().numpy()
        return masks, iou_predictions

def load_model(model_name: str, device='cpu', colab=False):
    """
    Loads a model weights and architecture by its name.
    """
    if model_name not in ["MedSAM", "LiteMedSAM", "SAM4Med", "SAM-Med2D", "Med-SA"]:
        raise ValueError(f"Model {model_name} is not recognized.")
    
    if model_name == "MedSAM":
        print("\nLoading MedSAM model...")
        medsam_checkpoint = "MedSAM/work_dir/MedSAM/medsam_vit_b.pth"
        if colab:
            medsam_checkpoint = "/content/drive/My Drive/fuseMedicalSAM/" + medsam_checkpoint
        medsam = MedSAM("vit_b", 1024, medsam_checkpoint)
        medsam.to(device)
        # medsam.eval()
        return medsam

    if model_name == "LiteMedSAM":
        # Load LiteMedSAM model
        print("\nLoading LiteMedSAM model...")
        lite_medsam_checkpoint = "LiteMedSAM/work_dir/LiteMedSAM/lite_medsam.pth"
        if colab:
            lite_medsam_checkpoint = "/content/drive/My Drive/fuseMedicalSAM/" + lite_medsam_checkpoint
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

        medsam_lite_model = _MedSAM_Lite(
            image_encoder = medsam_lite_image_encoder,
            mask_decoder = medsam_lite_mask_decoder,
            prompt_encoder = medsam_lite_prompt_encoder
        )

        medsam_lite_model.load_state_dict(torch.load(lite_medsam_checkpoint, map_location='cpu'))

        litemedsam = LiteMedSAM(medsam_lite_model)
        litemedsam.to(device)
        # litemedsam.eval()
        return litemedsam
    elif model_name == "SAM4Med":
        # Load SAM4Med model
        print("\nLoading SAM4Med model...")
        og_checkpoint = "SAM4Med/sam_vit_b_01ec64.pth"
        sam4med_checkpoint = "SAM4Med/model/medsam_box_best_vitb.pth"
        if colab:
            og_checkpoint = "/content/drive/My Drive/fuseMedicalSAM/" + og_checkpoint
            sam4med_checkpoint = "/content/drive/My Drive/fuseMedicalSAM/" + sam4med_checkpoint
        model_type = "vit_b"

        # SAM4Med
        sam = sam_model_registry_sam4med[model_type](checkpoint=sam4med_checkpoint, device='cpu')
        sam.to(device)
        predictor = SamPredictor(sam)

        sam4med = SAM4Med(predictor, model_type=model_type, checkpoint=og_checkpoint)
        sam4med.to(device)
        # sam4med.eval()
        return sam4med
    elif model_name == "SAM-Med2D":
        # Load SAM-Med2D model
        print("\nLoading SAM-Med2D model...")
        sam_med2d_checkpoint = "SAM_Med2D/pretrain_model/sam-med2d_b.pth"
        if colab:
            sam_med2d_checkpoint = "/content/drive/My Drive/fuseMedicalSAM/" + sam_med2d_checkpoint
        model_type = "vit_b"
        sam_med2d = SAM_Med2D(model_type=model_type, image_size=256, sam_checkpoint=sam_med2d_checkpoint, encoder_adapter=True)
        sam_med2d.to(device)
        # sam_med2d.eval()
        return sam_med2d
    else:
        raise NotImplementedError(f"Loading for {model_name} is not implemented yet.")


def calculate_segmentation_losses(gt_masks, mask_logits):
    """
    Calculates Dice Loss, BCE Loss, and IoU Loss for each individual mask/channel.

    Args:
        gt_masks (torch.Tensor): Ground truth masks, shape [B, C, H, W].
                                 Expected to be binary (0 or 1).
        mask_logits (torch.Tensor): Raw model output (logits), shape [B, C, H, W].

    Returns:
        tuple: A tuple containing (dice_losses_per_mask, bce_losses_per_mask, iou_losses_per_mask).
               Each element in the tuple will be a list of scalar loss values,
               one for each mask/channel.
    """

    # Ensure gt_masks and mask_logits are float for calculations
    gt_masks = gt_masks.float()
    mask_logits = mask_logits.float() # Ensure mask_logits is float for BCEWithLogitsLoss

    # --- 1. Calculate BCE Loss ---
    # Use reduction='none' to get per-element (per-pixel) loss
    # The output shape will be [B, C, H, W]
    bce_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    bce_pixel_losses = bce_loss_fn(mask_logits, gt_masks)
    bce_pixel_losses_numpy = bce_pixel_losses.squeeze().detach().cpu().numpy()

    # # To get a single BCE loss value per mask (channel), average over H and W
    # # The result will be shape [B, C]
    # bce_losses_per_mask_tensor = bce_pixel_losses.mean(dim=(-1, -2)) # Average over H and W

    # # Convert to a list of individual loss values (flattened Bx_C_ list)
    # bce_losses_list = bce_losses_per_mask_tensor.flatten().tolist()


    # --- 2. Calculate Dice & IoU Loss ---
    mask_probs = torch.sigmoid(mask_logits)
    epsilon = 1e-6

    batch_size = gt_masks.shape[0]
    num_channels = gt_masks.shape[1]

    dice_losses_list = []
    iou_losses_list = []

    # Iterate over each item in the batch AND each channel
    for b in range(batch_size):
        for c in range(num_channels):
            # Extract current mask/channel for this batch item
            # Shape will be [H, W] for the current mask
            gt_mask_slice = gt_masks[b, c, :, :]
            prob_mask_slice = mask_probs[b, c, :, :]

            # Flatten these slices for element-wise operations
            gt_flat = gt_mask_slice.reshape(-1)
            prob_flat = prob_mask_slice.reshape(-1)

            # Calculate intersection
            intersection = (prob_flat * gt_flat).sum()

            # Calculate union for Dice (P + G)
            union_dice = prob_flat.sum() + gt_flat.sum()

            # Calculate union for IoU (P + G - Intersection)
            union_iou = prob_flat.sum() + gt_flat.sum() - intersection

            # Special handling for perfectly empty ground truth and prediction
            # This ensures that if both are truly empty, the loss is 0.0,
            # which is the correct behavior for metric calculation.
            if gt_flat.sum() == 0 and prob_flat.sum() < epsilon:
                dice_loss_val = torch.tensor(0.0, device=gt_masks.device)
                iou_loss_val = torch.tensor(0.0, device=gt_masks.device)
            else:
                dice_coefficient = (2. * intersection + epsilon) / (union_dice + epsilon)
                dice_loss_val = 1 - dice_coefficient

                iou_coefficient = (intersection + epsilon) / (union_iou + epsilon)
                iou_loss_val = 1 - iou_coefficient

            dice_losses_list.append(dice_loss_val.item())
            iou_losses_list.append(iou_loss_val.item())

    # Return lists of individual scalar losses
    return dice_losses_list, bce_pixel_losses_numpy, iou_losses_list
