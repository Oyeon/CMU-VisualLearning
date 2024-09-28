# Credit to Justin Johnsons' EECS-598 course at the University of Michigan,
# from which this assignment is heavily drawn.
import math
from typing import Dict, List, Optional, Union

import torch
from detection_utils import *
from torch import nn
from torch.nn import functional as F
from torch.utils.data._utils.collate import default_collate
from torchvision import models
from torchvision.models import feature_extraction
from torchvision.ops import nms

from torchvision.ops import sigmoid_focal_loss

# def convert_to_one_hot(cls_targets: torch.Tensor, num_classes: int) -> torch.Tensor:
#     """
#     Converts class labels to one-hot encoded targets suitable for sigmoid focal loss.

#     Args:
#         cls_targets (torch.Tensor): Tensor of shape [N] with class labels.
#         num_classes (int): Number of classes.

#     Returns:
#         torch.Tensor: One-hot encoded tensor of shape [N, C].
#     """
#     # Ensure cls_targets is of integer type
#     cls_targets = cls_targets.long()

#     # Initialize a tensor of zeros
#     one_hot_targets = torch.zeros(
#         (cls_targets.size(0), num_classes),
#         device=cls_targets.device,
#         dtype=torch.float32
#     )

#     # Create a mask for positive samples (labels >= 0)
#     pos_mask = cls_targets >= 0

#     if pos_mask.any():
#         # Assign 1 to the corresponding class indices
#         one_hot_targets[pos_mask, cls_targets[pos_mask]] = 1.0

#     # Background samples remain all zeros
#     return one_hot_targets

def convert_to_one_hot(cls_targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Converts class labels to one-hot encoded targets suitable for sigmoid focal loss.

    Args:
        cls_targets (torch.Tensor): Tensor of shape [N] with class labels.
        num_classes (int): Number of classes.

    Returns:
        torch.Tensor: One-hot encoded tensor of shape [N, C].
    """
    # Ensure cls_targets is of integer type
    cls_targets = cls_targets.long()

    # Initialize a tensor of zeros
    one_hot_targets = torch.zeros(
        (cls_targets.size(0), num_classes),
        device=cls_targets.device,
        dtype=torch.float32
    )

    # Create mask for valid class labels (i.e., not background)
    valid_mask = cls_targets >= 0

    # For valid class labels, set the corresponding position to 1
    one_hot_targets[valid_mask, cls_targets[valid_mask]] = 1.0

    return one_hot_targets


class DetectorBackboneWithFPN(nn.Module):
    """
    Detection backbone network: A tiny RegNet model coupled with a Feature
    Pyramid Network (FPN). This model takes in batches of input images with
    shape `(B, 3, H, W)` and gives features from three different FPN levels
    with shapes and total strides upto that level:

        - level p3: (out_channels, H /  8, W /  8)      stride =  8
        - level p4: (out_channels, H / 16, W / 16)      stride = 16
        - level p5: (out_channels, H / 32, W / 32)      stride = 32

    NOTE: We could use any convolutional network architecture that progressively
    downsamples the input image and couple it with FPN. We use a small enough
    backbone that can work with Colab GPU and get decent enough performance.
    """

    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels
        _cnn = models.regnet_x_400mf(pretrained=True)
        self.backbone = feature_extraction.create_feature_extractor(
            _cnn,
            return_nodes={
                "trunk_output.block2": "c3",
                "trunk_output.block3": "c4",
                "trunk_output.block4": "c5",
            },
        )

        # Pass a dummy batch of input images to infer shapes of (c3, c4, c5).
        dummy_out = self.backbone(torch.randn(2, 3, 224, 224))
        dummy_out_shapes = [(key, value.shape) for key, value in dummy_out.items()]

        self.fpn_params = nn.ModuleDict()

        # Initialize lateral 1x1 convolutions for c3, c4, c5
        # Create THREE "lateral" 1x1 conv layers to transform (c3, c4, c5)   #
        # such that they all end up with the same `out_channels`.            #        
        for level in ['c3', 'c4', 'c5']:
            in_channels = dummy_out[level].shape[1]  # Extract input channels
            self.fpn_params[f'lateral_{level}'] = nn.Conv2d(
                in_channels, self.out_channels, kernel_size=1, stride=1, padding=0
            )

        # Initialize output 3x3 convolutions for p3, p4, p5
        # Then create THREE "output" 3x3 conv layers to transform the merged #
        # FPN features to output (p3, p4, p5) features.                      #
        for level in ['p3', 'p4', 'p5']:
            self.fpn_params[f'output_{level}'] = nn.Conv2d(
                self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1
            )

    @property
    def fpn_strides(self):
        """
        Total stride up to the FPN level. For a fixed ConvNet, these values
        are invariant to input image size. You may access these values freely
        to implement your logic in FCOS / Faster R-CNN.
        """
        return {"p3": 8, "p4": 16, "p5": 32}

    def forward(self, images: torch.Tensor):
        backbone_feats = self.backbone(images)

        fpn_feats = {"p3": None, "p4": None, "p5": None}
        
        # Apply lateral 1x1 convolutions
        c3_lateral = self.fpn_params['lateral_c3'](backbone_feats['c3'])
        c4_lateral = self.fpn_params['lateral_c4'](backbone_feats['c4'])
        c5_lateral = self.fpn_params['lateral_c5'](backbone_feats['c5'])

        # Padding in Convolutions: 3x3 conv padding=1 to preserve spatial dimensions. To align feature maps during the addition operation in the top-down pathway
        # Nearest-Neighbor: nearest is efficient and simpler (maybe we can use others)

        # p5: Start top-down from p5
            # Directly applies the output_p5 3x3 convolution to the c5_lateral feature.
        fpn_feats['p5'] = self.fpn_params['output_p5'](c5_lateral)

        # p4: Upsample p5 and add to c4 lateral
            # Upsampling: The c5_lateral feature is upsampled by a factor of 2 using nearest-neighbor interpolation.
            # Merging: The upsampled c5_lateral is added to the c4_lateral feature.
            # Convolution: The merged feature is passed through the output_p4 3x3 convolution to produce p4.
        fpn_feats['p4'] = self.fpn_params['output_p4'](c4_lateral + F.interpolate(c5_lateral, scale_factor=2, mode='nearest'))

        # p3: Upsample p4 and add to c3 lateral
            # First Upsampling: The c5_lateral feature is upsampled by a factor of 2.
            # First Merging: The upsampled c5_lateral is added to the c4_lateral feature.
            # Second Upsampling: The merged c4_lateral feature is further upsampled by a factor of 2.
            # Second Merging: The twice-upsampled feature is added to the c3_lateral feature.
            # Convolution: The twice-merged feature is passed through the output_p3 3x3 convolution to produce p3.        
        fpn_feats['p3'] = self.fpn_params['output_p3'](c3_lateral + F.interpolate(c4_lateral + F.interpolate(c5_lateral, scale_factor=2, mode='nearest'), scale_factor=2, mode='nearest'))

        return fpn_feats


class FCOSPredictionNetwork(nn.Module):
    """
    FCOS prediction network that accepts FPN feature maps from different levels
    and makes three predictions at every location: bounding boxes, class ID and
    centerness. This module contains a "stem" of convolution layers, along with
    one final layer per prediction. For a visual depiction, see Figure 2 (right
    side) in FCOS paper: https://arxiv.org/abs/1904.01355

    We will use feature maps from FPN levels (P3, P4, P5) and exclude (P6, P7).
    """

    def __init__(
        self, num_classes: int, in_channels: int, stem_channels: List[int]
    ):
        """
        Args:
            num_classes: Number of object classes for classification.
            in_channels: Number of channels in input feature maps. This value
                is same as the output channels of FPN, since the head directly
                operates on them.
            stem_channels: List of integers giving the number of output channels
                in each convolution layer of stem layers.
        """
        super().__init__()
        
        stem_cls = []
        stem_box = []
        current_in_channels = in_channels

        # Create stems layers for class prediction
        for out_channels in stem_channels:
            stem_cls.append(
                nn.Conv2d(
                    in_channels=current_in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            stem_cls.append(nn.ReLU(inplace=True))
            current_in_channels = out_channels

        # Reset for stem_box
        current_in_channels = in_channels

        # Create stem layers for box prediction
        for out_channels in stem_channels:
            stem_box.append(
                nn.Conv2d(
                    in_channels=current_in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            stem_box.append(nn.ReLU(inplace=True))
            current_in_channels = out_channels

        self.stem_cls = nn.Sequential(*stem_cls)
        self.stem_box = nn.Sequential(*stem_box)

        # Initialize all layers.
        for stems in (self.stem_cls, self.stem_box):
            for layer in stems:
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)

        #1. object class prediction conv (`num_classes` outputs)
        self.pred_cls = nn.Conv2d(
            in_channels=stem_channels[-1],
            out_channels=num_classes,
            kernel_size=3,
            stride=1,
            padding=1,
        )  

        #2. box regression prediction conv (4 outputs: LTRB deltas from locations)  
        self.pred_box = nn.Conv2d(
            in_channels=stem_channels[-1],
            out_channels=4,
            kernel_size=3,
            stride=1,
            padding=1,
        )  

        #3. centerness prediction conv (1 output)
        self.pred_ctr = nn.Conv2d(
            in_channels=stem_channels[-1],
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1,
        )  

        # OVERRIDE: Use a negative bias in `pred_cls` to improve training
        # stability. Without this, the training will most likely diverge.
        # STUDENTS: You do not need to get into details of why this is needed.
        if self.pred_cls is not None:
            torch.nn.init.constant_(self.pred_cls.bias, -math.log(99))

    def forward(self, feats_per_fpn_level: TensorDict) -> List[TensorDict]:
        """
        Accept FPN feature maps and predict the desired outputs at every location
        (as described above). Format them such that channels are placed at the
        last dimension, and (H, W) are flattened (having channels at last is
        convenient for computing loss as well as perforning inference).

        Args:
            feats_per_fpn_level: Features from FPN, keys {"p3", "p4", "p5"}. Each
                tensor will have shape `(batch_size, fpn_channels, H, W)`. For an
                input (224, 224) image, H = W are (28, 14, 7) for (p3, p4, p5).

        Returns:
            List of dictionaries, each having keys {"p3", "p4", "p5"}:
            1. Classification logits: `(batch_size, H * W, num_classes)`.
            2. Box regression deltas: `(batch_size, H * W, 4)`
            3. Centerness logits:     `(batch_size, H * W, 1)`
        """

        class_logits = {}
        boxreg_deltas = {}
        centerness_logits = {}

        for level_name, feature in feats_per_fpn_level.items():
            # Classification branch
            cls_feat = self.stem_cls(feature)
            cls_logit = self.pred_cls(cls_feat)  # (B, num_classes, H, W)
            # Reshape to (B, H*W, num_classes)
            B, C, H, W = cls_logit.shape
            cls_logit = cls_logit.permute(0, 2, 3, 1).reshape(B, H * W, C)
            class_logits[level_name] = cls_logit

            # Box regression branch
            box_feat = self.stem_box(feature)
            box_delta = self.pred_box(box_feat)  # (B, 4, H, W)
            # Reshape to (B, H*W, 4)
            box_delta = box_delta.permute(0, 2, 3, 1).reshape(B, H * W, 4)
            boxreg_deltas[level_name] = box_delta

            # Centerness branch
            ctr_logit = self.pred_ctr(box_feat)  # (B, 1, H, W)
            # Reshape to (B, H*W, 1)
            ctr_logit = ctr_logit.permute(0, 2, 3, 1).reshape(B, H * W, 1)
            centerness_logits[level_name] = ctr_logit

        return [class_logits, boxreg_deltas, centerness_logits]


class FCOS(nn.Module):
    """
    FCOS: Fully-Convolutional One-Stage Detector

    This class integrates the backbone with FPN, prediction heads, loss computations,
    and inference mechanisms. It handles both training and inference workflows.
    """
    def __init__(
        self,
        num_classes: int,
        fpn_channels: int = 256,
        stem_channels: List[int] = [256, 256, 256, 256],
    ):
        super(FCOS, self).__init__()
        self.num_classes = num_classes

        ######################################################################
        # Initialize backbone and FPN using arguments.
        ######################################################################

        self.backbone_fpn = DetectorBackboneWithFPN(out_channels=fpn_channels)
        self.prediction_heads = FCOSPredictionNetwork(
            num_classes=num_classes,
            in_channels=fpn_channels,
            stem_channels=stem_channels
        )
        self._normalizer = 1000  # Initial value for EMA normalizer

    def forward(self, images: torch.Tensor, gt_boxes: torch.Tensor = None, **kwargs) -> Dict[str, torch.Tensor]:
        """
        FCOS Forward Pass

        Args:
            images (torch.Tensor): Batch of images, shape (B, C, H, W).
            gt_boxes (torch.Tensor, optional): Ground-truth boxes, shape (B, N, 5).

        Returns:
            Dict[str, torch.Tensor]: Losses during training.
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Predictions during inference.
        """
        # Extract features using backbone and FPN
        features = self.backbone_fpn(images)  # features is a dict

        # Generate shapes for each FPN level
        shape_per_fpn_level = {level: feat.shape for level, feat in features.items()}

        # Generate location coordinates for each FPN level
        locations_per_fpn_level = get_fpn_location_coords(
            shape_per_fpn_level,
            self.backbone_fpn.fpn_strides,
            dtype=features[next(iter(features))].dtype,
            device=features[next(iter(features))].device
        )

        # Pass features through prediction heads
        class_logits, boxreg_deltas, centerness_logits = self.prediction_heads(features)

        if self.training:
            # Initialize accumulators for targets
            all_class_targets = []
            all_boxreg_targets_levels = []
            all_centerness_targets_levels = []
            
            # Initialize accumulators for predictions
            all_class_logits_pred_levels = []
            all_boxreg_deltas_pred_levels = []
            all_centerness_logits_pred_levels = []

            B = images.shape[0]
            for b in range(B):
                # Extract ground truth boxes for image b
                gt_b = gt_boxes[b]  # Shape: (Num_GT, 5)

                # Assign GT boxes to locations
                matched_gt_boxes = fcos_match_locations_to_gt(
                    {k: v.clone().detach() for k, v in locations_per_fpn_level.items()},
                    self.backbone_fpn.fpn_strides,
                    gt_b  # Shape: (Num_GT, 5)
                )

                for level_name in self.backbone_fpn.fpn_strides.keys():
                    # Extract class targets
                    cls_target = matched_gt_boxes[level_name][:, 4]  # Shape: (N,)
                    all_class_targets.append(cls_target)

                    # Compute box deltas using utility function
                    deltas = fcos_get_deltas_from_locations(
                        locations_per_fpn_level[level_name],
                        matched_gt_boxes[level_name],
                        self.backbone_fpn.fpn_strides[level_name]
                    )  # Shape: (N, 4)
                    all_boxreg_targets_levels.append(deltas)

                    # Compute centerness targets using utility function
                    ctr_target = fcos_make_centerness_targets(deltas)  # Shape: (N,)
                    all_centerness_targets_levels.append(ctr_target)
            
            # Concatenate all targets
            cls_targets_tensor = torch.cat(all_class_targets, dim=0)       # Shape: [N]
            box_targets_tensor = torch.cat(all_boxreg_targets_levels, dim=0)   # Shape: [N, 4]
            ctr_targets_tensor = torch.cat(all_centerness_targets_levels, dim=0)  # Shape: [N]

            # Convert class targets to one-hot encoding
            class_targets_binary = convert_to_one_hot(cls_targets_tensor, self.num_classes)  # Shape: [N, C]

            # Accumulate predictions from all FPN levels
            for level in self.backbone_fpn.fpn_strides.keys():
                # Class logits
                all_class_logits_pred_levels.append(class_logits[level].reshape(-1, self.num_classes))  # [L, C]
                # Box regression deltas
                all_boxreg_deltas_pred_levels.append(boxreg_deltas[level].reshape(-1, 4))                 # [L, 4]
                # Centerness logits
                all_centerness_logits_pred_levels.append(centerness_logits[level].reshape(-1, 1))        # [L, 1]

            # Concatenate all predictions
            class_logits_all = torch.cat(all_class_logits_pred_levels, dim=0)        # Shape: [N, C]
            boxreg_deltas_all = torch.cat(all_boxreg_deltas_pred_levels, dim=0)      # Shape: [N, 4]
            centerness_logits_all = torch.cat(all_centerness_logits_pred_levels, dim=0)  # Shape: [N, 1]

            # Debugging: Verify shapes
            # print(f"class_logits_all shape: {class_logits_all.shape}")             # Expected: [N, C]
            # print(f"class_targets_binary shape: {class_targets_binary.shape}")     # Expected: [N, C]
            # print(f"boxreg_deltas_all shape: {boxreg_deltas_all.shape}")           # Expected: [N, 4]
            # print(f"ctr_targets_tensor shape: {ctr_targets_tensor.shape}")         # Expected: [N]
            # print(f"centerness_logits_all shape: {centerness_logits_all.shape}") # Expected: [N, 1]

            # Compute classification loss with reduction='sum'
            loss_cls = sigmoid_focal_loss(
                class_logits_all,        # [N, C]
                class_targets_binary,    # [N, C]
                reduction='sum'
            )

            # Compute Box Regression Loss (Only for positive samples)
            fg_mask = cls_targets_tensor != -1  # Background samples have label -1, Shape: [N]

            # Ensure fg_mask is boolean
            fg_mask = fg_mask.bool()

            if fg_mask.sum() > 0:
                box_pred = boxreg_deltas_all[fg_mask]           # Shape: (num_fg, 4)
                box_targets_fg = box_targets_tensor[fg_mask]    # Shape: (num_fg, 4)
                loss_box = F.l1_loss(box_pred, box_targets_fg, reduction='sum')

                # Centerness Loss (Only for positive samples)
                ctr_pred = centerness_logits_all[fg_mask].squeeze(1)  # Shape: (num_fg,)
                ctr_targets_fg = ctr_targets_tensor[fg_mask]          # Shape: (num_fg,)
                loss_ctr = F.binary_cross_entropy_with_logits(
                    ctr_pred, 
                    ctr_targets_fg, 
                    reduction='sum'
                )
            else:
                loss_box = torch.tensor(0.0, device=images.device)
                loss_ctr = torch.tensor(0.0, device=images.device)

            # Debugging: Verify that all losses are scalar
            # print(f"loss_cls shape: {loss_cls.shape}")  # Expected: torch.Size([])
            # print(f"loss_box shape: {loss_box.shape}")  # Expected: torch.Size([])
            # print(f"loss_ctr shape: {loss_ctr.shape}")  # Expected: torch.Size([])

            # Compute normalization
            loss_normalizer = self._normalizer * B
            losses = {
                "loss_cls": loss_cls / loss_normalizer,
                "loss_box": loss_box / loss_normalizer,
                "loss_ctr": loss_ctr / loss_normalizer,
            }

            return losses
        else:
            # Inference mode: Handle predictions
            # Retrieve locations and predictions
            locations_per_fpn_level = get_fpn_location_coords(
                {level: feat.shape for level, feat in self.backbone_fpn(images).items()},
                self.backbone_fpn.fpn_strides,
                dtype=images.dtype,
                device=images.device
            )

            pred_cls_logits, pred_boxreg_deltas, pred_ctr_logits = self.prediction_heads(features)

            # Call the inference method with the provided thresholds
            return self.inference(
                images,
                locations_per_fpn_level,
                pred_cls_logits,
                pred_boxreg_deltas,
                pred_ctr_logits,
                test_score_thresh=kwargs.get('test_score_thresh', 0.1),
                test_nms_thresh=kwargs.get('test_nms_thresh', 0.3),
            )

    #
    def inference(
        self,
        images: torch.Tensor,
        locations_per_fpn_level: Dict[str, torch.Tensor],
        pred_cls_logits: Dict[str, torch.Tensor],
        pred_boxreg_deltas: Dict[str, torch.Tensor],
        pred_ctr_logits: Dict[str, torch.Tensor],
        test_score_thresh: float = 0.3,
        test_nms_thresh: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform inference on the input images and return predictions.

        Args:
            images: Batch of images, tensors of shape `(B, C, H, W)`.
            locations_per_fpn_level: Dictionary of location coordinates per FPN level.
            pred_cls_logits: Dictionary of predicted class logits per FPN level.
            pred_boxreg_deltas: Dictionary of predicted box regression deltas per FPN level.
            pred_ctr_logits: Dictionary of predicted centerness logits per FPN level.
            test_score_thresh: Threshold to filter out low-confidence predictions.
            test_nms_thresh: IoU threshold for Non-Maximum Suppression (NMS).

        Returns:
            Tuple containing:
                - Predicted bounding boxes
                - Predicted class labels
                - Predicted confidence scores
        """

        pred_boxes_all_levels = []
        pred_classes_all_levels = []
        pred_scores_all_levels = []

        for level_name in pred_cls_logits.keys():
            level_cls_logits = pred_cls_logits[level_name].view(
                images.shape[0], -1, self.num_classes
            )  # (B, H*W, num_classes)
            level_deltas = pred_boxreg_deltas[level_name].view(
                images.shape[0], -1, 4
            )  # (B, H*W, 4)
            level_ctr_logits = pred_ctr_logits[level_name].view(
                images.shape[0], -1, 1
            )  # (B, H*W, 1)

            # Debugging: Uncomment the following lines to trace shapes
            # print(f"Level: {level_name}")
            # print(f"Class Logits Shape: {level_cls_logits.shape}")
            # print(f"Box Deltas Shape: {level_deltas.shape}")
            # print(f"Centerness Logits Shape: {level_ctr_logits.shape}")

            # Iterate over the batch
            for i in range(images.shape[0]):
                # Get per-image predictions
                cls_logits = level_cls_logits[i]  # (H*W, num_classes)
                deltas = level_deltas[i]          # (H*W, 4)
                ctr_logits = level_ctr_logits[i]  # (H*W, 1)

                # Compute sigmoid probabilities
                cls_probs = torch.sigmoid(cls_logits)  # (H*W, num_classes)
                ctr_probs = torch.sigmoid(ctr_logits)  # (H*W, 1)

                # Compute confidence scores by multiplying class probs with centerness probs
                confidence_scores = cls_probs * ctr_probs  # (H*W, num_classes)

                # Get the maximum confidence score and corresponding class for each location
                scores, classes = confidence_scores.max(dim=1)  # (H*W,), (H*W,)

                # Apply score threshold
                score_mask = scores > test_score_thresh
                scores = scores[score_mask]
                classes = classes[score_mask]
                selected_deltas = deltas[score_mask]

                if scores.numel() == 0:
                    # No detections for this image and level
                    continue

                # Get corresponding locations
                selected_locations = locations_per_fpn_level[level_name][score_mask]  # (N, 2)

                # Apply deltas to locations to get predicted boxes
                pred_boxes = fcos_apply_deltas_to_locations(
                    deltas=selected_deltas,
                    locations=selected_locations,
                    stride=self.backbone_fpn.fpn_strides[level_name],
                )  # (N, 4)

                # Clip XYXY box-coordinates that go beyond the height and width of input image.
                img_h, img_w = images.shape[2], images.shape[3]
                pred_boxes[:, 0::2].clamp_(min=0, max=img_w - 1)  # x1, x2
                pred_boxes[:, 1::2].clamp_(min=0, max=img_h - 1)  # y1, y2

                pred_boxes_all_levels.append(pred_boxes)
                pred_classes_all_levels.append(classes)
                pred_scores_all_levels.append(scores)

        ######################################################################
        # Combine predictions from all levels and perform Class-Specific NMS.
        ######################################################################
        if pred_boxes_all_levels:
            # Concatenate all predictions from different FPN levels
            pred_boxes_all_levels = torch.cat(pred_boxes_all_levels, dim=0)        # (Total_preds, 4)
            pred_classes_all_levels = torch.cat(pred_classes_all_levels, dim=0)    # (Total_preds,)
            pred_scores_all_levels = torch.cat(pred_scores_all_levels, dim=0)      # (Total_preds,)

            # Perform Class-Specific Non-Maximum Suppression (NMS)
            keep = class_spec_nms(
                boxes=pred_boxes_all_levels,
                scores=pred_scores_all_levels,
                class_ids=pred_classes_all_levels,
                iou_threshold=test_nms_thresh,
            )

            # Filter the predictions
            pred_boxes_all_levels = pred_boxes_all_levels[keep]
            pred_classes_all_levels = pred_classes_all_levels[keep]
            pred_scores_all_levels = pred_scores_all_levels[keep]

            return (
                pred_boxes_all_levels,
                pred_classes_all_levels,
                pred_scores_all_levels,
            )
        else:
            # No predictions were made
            return (
                torch.empty((0, 4), device=images.device),
                torch.empty((0,), dtype=torch.int64, device=images.device),
                torch.empty((0,), dtype=torch.float32, device=images.device),
            )



# class FCOS(nn.Module):
#     """
#     FCOS: Fully-Convolutional One-Stage Detector

#     This class puts together everything you implemented so far. It contains a
#     backbone with FPN, and prediction layers (head). It computes loss during
#     training and predicts boxes during inference.
#     """

#     def __init__(
#         self, num_classes: int, fpn_channels: int, stem_channels: List[int]
#     ):
#         super().__init__()
#         self.num_classes = num_classes

#         ######################################################################
#         # Initialize backbone and prediction network using arguments.        #
#         ######################################################################
#         self.backbone = DetectorBackboneWithFPN(out_channels=fpn_channels)
#         self.pred_net = FCOSPredictionNetwork(
#             num_classes=num_classes,
#             in_channels=fpn_channels,
#             stem_channels=stem_channels
#         )
#         ######################################################################
#         #                           END OF YOUR CODE                         #
#         ######################################################################

#         # Averaging factor for training loss; EMA of foreground locations.
#         # STUDENTS: See its use in `forward` when you implement losses.
#         self._normalizer = 150  # per image

#     @staticmethod
#     def _cat_across_fpn_levels(
#         dict_with_fpn_levels: Dict[str, torch.Tensor], dim: int = 1
#     ):
#         """
#         Convert a dict of tensors across FPN levels {"p3", "p4", "p5"} to a
#         single tensor. Values could be anything - batches of image features,
#         GT targets, etc.
#         """
#         return torch.cat(list(dict_with_fpn_levels.values()), dim=dim)

#     def forward(
#         self,
#         images: torch.Tensor,
#         gt_boxes: Optional[torch.Tensor] = None,
#         test_score_thresh: Optional[float] = None,
#         test_nms_thresh: Optional[float] = None,
#     ):
#         """
#         Args:
#             images: Batch of images, tensors of shape `(B, C, H, W)`.
#             gt_boxes: Batch of training boxes, tensors of shape `(B, N, 5)`.
#                 `gt_boxes[i, j] = (x1, y1, x2, y2, C)` gives information about
#                 the `j`th object in `images[i]`. The position of the top-left
#                 corner of the box is `(x1, y1)` and the position of bottom-right
#                 corner of the box is `(x2, y2)`. These coordinates are
#                 real-valued in `[H, W]`. `C` is an integer giving the category
#                 label for this bounding box. Not provided during inference.
#             test_score_thresh: During inference, discard predictions with a
#                 confidence score less than this value. Ignored during training.
#             test_nms_thresh: IoU threshold for NMS during inference. Ignored
#                 during training.

#         Returns:
#             Losses during training and predictions during inference.
#         """

#         ######################################################################
#         # Process the image through backbone, FPN, and prediction head       #
#         # to obtain model predictions at every FPN location.                 #
#         # Get dictionaries of keys {"p3", "p4", "p5"} giving predicted class #
#         # logits, deltas, and centerness.                                    #
#         ######################################################################
#         backbone_feats = self.backbone(images)
#         pred_cls_logits, pred_boxreg_deltas, pred_ctr_logits = self.pred_net(backbone_feats)

#         ######################################################################
#         # Get absolute co-ordinates `(xc, yc)` for every location in         #
#         # FPN levels.                                                        #
#         ######################################################################
#         locations_per_fpn_level = get_fpn_location_coords(
#             {level: feat.shape for level, feat in backbone_feats.items()},
#             self.backbone.fpn_strides,
#             dtype=images.dtype,
#             device=images.device
#         )

#         ######################################################################
#         #                           END OF YOUR CODE                         #
#         ######################################################################

#         if not self.training:
#             # During inference, just go to this method and skip rest of the
#             # forward pass.
#             # fmt: off
#             return self.inference(
#                 images, locations_per_fpn_level,
#                 pred_cls_logits, pred_boxreg_deltas, pred_ctr_logits,
#                 test_score_thresh=test_score_thresh,
#                 test_nms_thresh=test_nms_thresh,
#             )
#             # fmt: on

#         ######################################################################
#         # Assign ground-truth boxes to feature locations. We have this       #
#         # implemented in a `fcos_match_locations_to_gt`. This operation is   #
#         # NOT BATCHED so call it separately per GT boxes in batch.           #
#         ######################################################################
#         matched_gt_boxes = []
#         matched_gt_deltas = []
#         for gt_boxes_per_image in gt_boxes:
#             matched_boxes = fcos_match_locations_to_gt(locations_per_fpn_level, self.backbone.fpn_strides, gt_boxes_per_image)
#             matched_gt_boxes.append(matched_boxes)
#             deltas_per_image = {}
#             for level_name in matched_boxes.keys():
#                 deltas = fcos_get_deltas_from_locations(
#                     locations_per_fpn_level[level_name],
#                     matched_boxes[level_name],
#                     self.backbone.fpn_strides[level_name]
#                 )
#                 deltas_per_image[level_name] = deltas
#             matched_gt_deltas.append(deltas_per_image)

#         ######################################################################
#         #                           END OF YOUR CODE                         #
#         ######################################################################

#         # Collate lists of dictionaries into dictionaries of batched tensors.
#         # These are dictionaries with keys {"p3", "p4", "p5"} and values as
#         # tensors of shape (batch_size, locations_per_fpn_level, 5 or 4)
#         matched_gt_boxes = default_collate(matched_gt_boxes)
#         matched_gt_deltas = default_collate(matched_gt_deltas)

#         # Combine predictions and GT from across all FPN levels.
#         # shape: (batch_size, num_locations_across_fpn_levels, ...)
#         matched_gt_boxes = self._cat_across_fpn_levels(matched_gt_boxes)
#         matched_gt_deltas = self._cat_across_fpn_levels(matched_gt_deltas)
#         pred_cls_logits = self._cat_across_fpn_levels(pred_cls_logits)
#         pred_boxreg_deltas = self._cat_across_fpn_levels(pred_boxreg_deltas)
#         pred_ctr_logits = self._cat_across_fpn_levels(pred_ctr_logits)

#         # Perform EMA update of normalizer by number of positive locations.
#         num_pos_locations = (matched_gt_boxes[:, :, 4] != -1).sum()
#         pos_loc_per_image = num_pos_locations.item() / images.shape[0]
#         self._normalizer = 0.9 * self._normalizer + 0.1 * pos_loc_per_image

#         ######################################################################
#         # Calculate losses per location for classification, box reg and      #
#         # centerness. Remember to set box/centerness losses for "background" #
#         # positions to zero.                                                 #
#         ######################################################################
#         # Classification loss
#         cls_targets = matched_gt_boxes[:, :, 4].long().view(-1)  # [B * num_locations]

#         # Validate class labels to prevent indexing errors
#         if (cls_targets < -1).any() or (cls_targets >= self.num_classes).any():
#             raise ValueError("Found class labels outside the valid range [-1, num_classes - 1].")

#         cls_targets_one_hot = convert_to_one_hot(cls_targets, self.num_classes)  # [B * num_locations, C]

#         # Flatten pred_cls_logits to [B * num_locations, C]
#         pred_cls_logits_flat = pred_cls_logits.view(-1, self.num_classes)  # [B * num_locations, C]

#         # Debugging: Verify shapes and sample values
#         # print(f"pred_cls_logits_flat shape: {pred_cls_logits_flat.shape}")          # Expected: [16464, 20]
#         # print(f"cls_targets_one_hot shape: {cls_targets_one_hot.shape}")            # Expected: [16464, 20]

#         loss_cls = sigmoid_focal_loss(pred_cls_logits_flat, cls_targets_one_hot, reduction='sum')

#         # Box regression loss
#         fg_mask = matched_gt_boxes[:, :, 4] != -1  # [B, num_locations]
#         loss_box = F.l1_loss(pred_boxreg_deltas[fg_mask], matched_gt_deltas[fg_mask], reduction='sum')

#         # Centerness loss
#         # Reshape fg_mask to match the deltas tensor for masking
#         fg_mask_flat = fg_mask.view(-1)  # [B * num_locations]

#         # Flatten deltas to [B * num_locations, 4]
#         matched_gt_deltas_flat = matched_gt_deltas.view(-1, 4)  # [B * num_locations, 4]

#         # Call fcos_make_centerness_targets on the flattened deltas
#         ctr_targets_flat = fcos_make_centerness_targets(matched_gt_deltas_flat)  # Should return [B * num_locations]

#         # Reshape ctr_targets_flat to [B * num_locations]
#         ctr_targets_flat = ctr_targets_flat.view(-1)

#         # Apply the foreground mask
#         ctr_targets = ctr_targets_flat[fg_mask_flat]  # [num_fg]
#         pred_ctr_logits_flat = pred_ctr_logits.view(-1)[fg_mask_flat]  # [num_fg]

#         loss_ctr = F.binary_cross_entropy_with_logits(pred_ctr_logits_flat, ctr_targets, reduction='sum')

#         ######################################################################
#         #                            END OF YOUR CODE                        #
#         ######################################################################
#         # Sum all locations and average by the EMA of foreground locations.
#         # In training code, we simply add these three and call `.backward()`
#         return {
#             "loss_cls": loss_cls / (self._normalizer * images.shape[0]),
#             "loss_box": loss_box / (self._normalizer * images.shape[0]),
#             "loss_ctr": loss_ctr / (self._normalizer * images.shape[0]),
#         }

#     def inference(
#         self,
#         images: torch.Tensor,
#         locations_per_fpn_level: Dict[str, torch.Tensor],
#         pred_cls_logits: Dict[str, torch.Tensor],
#         pred_boxreg_deltas: Dict[str, torch.Tensor],
#         pred_ctr_logits: Dict[str, torch.Tensor],
#         test_score_thresh: Optional[float] = 0.3,
#         test_nms_thresh: Optional[float] = 0.5,
#     ):
#         """
#         Run inference on a single input image (batch size = 1). Other input
#         arguments are same as those computed in `forward` method. This method
#         should not be called from anywhere except from inside `forward`.

#         Returns:
#             Three tensors:
#                 - pred_boxes: Tensor of shape `(N, 4)` giving *absolute* XYXY
#                 co-ordinates of predicted boxes.

#                 - pred_classes: Tensor of shape `(N, )` giving predicted class
#                 labels for these boxes (one of `num_classes` labels). Make
#                 sure there are no background predictions (-1).

#                 - pred_scores: Tensor of shape `(N, )` giving confidence scores
#                 for predictions: these values are `sqrt(class_prob * ctrness)`
#                 where class_prob and ctrness are obtained by applying sigmoid
#                 to corresponding logits.
#         """

#         # Ensure batch size = 1
#         if images.size(0) != 1:
#             raise ValueError("Inference method is designed for batch size = 1.")

#         # Set default thresholds if None
#         if test_score_thresh is None:
#             test_score_thresh = 0.3
#             print("test_score_thresh was None. Using default value 0.3.")
#         if test_nms_thresh is None:
#             test_nms_thresh = 0.5
#             print("test_nms_thresh was None. Using default value 0.5.")

#         # Gather scores and boxes from all FPN levels in this list. Once
#         # gathered, we will perform NMS to filter highly overlapping predictions.
#         pred_boxes_all_levels = []
#         pred_classes_all_levels = []
#         pred_scores_all_levels = []

#         for level_name in locations_per_fpn_level.keys():

#             # Get locations and predictions from a single level.
#             # We index predictions by `[0]` to remove batch dimension.
#             level_locations = locations_per_fpn_level[level_name]
#             level_cls_logits = pred_cls_logits[level_name][0]
#             level_deltas = pred_boxreg_deltas[level_name][0]
#             level_ctr_logits = pred_ctr_logits[level_name][0]

#             # Compute geometric mean of class logits and centerness:
#             # Use out-of-place sigmoid to avoid modifying original tensors
#             level_pred_scores = torch.sqrt(
#                 level_cls_logits.sigmoid() * level_ctr_logits.sigmoid()
#             )

#             # Step 1: Get the most confidently predicted class and its score for every box.
#             scores, classes = level_pred_scores.max(dim=1)

#             # Step 2: Only retain predictions that have a confidence score higher than the provided threshold.
#             score_mask = scores > test_score_thresh
#             scores = scores[score_mask]
#             classes = classes[score_mask]
#             selected_deltas = level_deltas[score_mask]
#             selected_locations = level_locations[score_mask]

#             if scores.numel() == 0:
#                 continue

#             # Step 3: Obtain predicted boxes using predicted deltas and locations.
#             pred_boxes = fcos_apply_deltas_to_locations(
#                 selected_deltas, selected_locations, self.backbone.fpn_strides[level_name]
#             )

#             # Step 4: Clip XYXY box-coordinates that go beyond the height and width of input image.
#             img_h, img_w = images.shape[2], images.shape[3]
#             pred_boxes[:, 0::2].clamp_(min=0, max=img_w - 1)  # x1, x2
#             pred_boxes[:, 1::2].clamp_(min=0, max=img_h - 1)  # y1, y2

#             pred_boxes_all_levels.append(pred_boxes)
#             pred_classes_all_levels.append(classes)
#             pred_scores_all_levels.append(scores)

#         ######################################################################
#         # Combine predictions from all levels and perform NMS.
#         if not pred_boxes_all_levels:
#             # Return empty tensors if no predictions
#             pred_boxes_all_levels = torch.empty((0, 4), device=images.device, dtype=images.dtype)
#             pred_classes_all_levels = torch.empty((0,), dtype=torch.long, device=images.device)
#             pred_scores_all_levels = torch.empty((0,), device=images.device)
#         else:
#             pred_boxes_all_levels = torch.cat(pred_boxes_all_levels)
#             pred_classes_all_levels = torch.cat(pred_classes_all_levels)
#             pred_scores_all_levels = torch.cat(pred_scores_all_levels)

#         # Apply class-specific NMS
#         keep = class_spec_nms(
#             pred_boxes_all_levels,
#             pred_scores_all_levels,
#             pred_classes_all_levels,
#             iou_threshold=test_nms_thresh,
#         )

#         # Handle case where NMS returns no boxes
#         if keep.numel() == 0:
#             pred_boxes_all_levels = torch.empty((0, 4), device=images.device, dtype=images.dtype)
#             pred_classes_all_levels = torch.empty((0,), dtype=torch.long, device=images.device)
#             pred_scores_all_levels = torch.empty((0,), device=images.device)
#         else:
#             pred_boxes_all_levels = pred_boxes_all_levels[keep]
#             pred_classes_all_levels = pred_classes_all_levels[keep]
#             pred_scores_all_levels = pred_scores_all_levels[keep]

#         # Ensure no background predictions (-1)
#         valid_class_mask = pred_classes_all_levels >= 0
#         pred_boxes_all_levels = pred_boxes_all_levels[valid_class_mask]
#         pred_classes_all_levels = pred_classes_all_levels[valid_class_mask]
#         pred_scores_all_levels = pred_scores_all_levels[valid_class_mask]

#         return (
#             pred_boxes_all_levels,
#             pred_classes_all_levels,
#             pred_scores_all_levels,
#         )

