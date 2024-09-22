# Credit to Justin Johnsons' EECS-598 course at the University of Michigan,
# from which this assignment is heavily drawn.
import math
from typing import Dict, List, Optional

import torch
from detection_utils import *
from torch import nn
from torch.nn import functional as F
from torch.utils.data._utils.collate import default_collate
from torchvision.ops import sigmoid_focal_loss
from torchvision import models
from torchvision.models import feature_extraction
from torchvision.ops import nms


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
#         # TODO: Initialize backbone and prediction network using arguments.  #
#         ######################################################################
#         # Feel free to delete these two lines: (but keep variable names same)
#         self.backbone = None
#         self.pred_net = None
#         # Replace "pass" statement with your code
#         pass
#         ######################################################################
#         #                           END OF YOUR CODE                         #
#         ######################################################################

#         # Averaging factor for training loss; EMA of foreground locations.
#         # STUDENTS: See its use in `forward` when you implement losses.
#         self._normalizer = 150  # per image

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
#                 corner of the box is `(x2, x2)`. These coordinates are
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
#         # TODO: Process the image through backbone, FPN, and prediction head #
#         # to obtain model predictions at every FPN location.                 #
#         # Get dictionaries of keys {"p3", "p4", "p5"} giving predicted class #
#         # logits, deltas, and centerness.                                    #
#         ######################################################################
#         # Feel free to delete this line: (but keep variable names same)
#         backbone_feats = None
#         pred_cls_logits, pred_boxreg_deltas, pred_ctr_logits = None, None, None

#         ######################################################################
#         # TODO: Get absolute co-ordinates `(xc, yc)` for every location in
#         # FPN levels.
#         #
#         # HINT: You have already implemented everything, just have to
#         # call the functions properly.
#         ######################################################################
#         # Feel free to delete this line: (but keep variable names same)
#         locations_per_fpn_level = None

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
#         # TODO: Assign ground-truth boxes to feature locations. We have this
#         # implemented in a `fcos_match_locations_to_gt`. This operation is NOT
#         # batched so call it separately per GT boxes in batch.
#         ######################################################################
#         # List of dictionaries with keys {"p3", "p4", "p5"} giving matched
#         # boxes for locations per FPN level, per image. Fill this list:
#         matched_gt_boxes = []
#         pass

#         # Calculate GT deltas for these matched boxes. Similar structure
#         # as `matched_gt_boxes` above. Fill this list:
#         matched_gt_deltas = []
#         # Replace "pass" statement with your code
#         pass
#         ######################################################################
#         #                           END OF YOUR CODE                         #
#         ######################################################################

#         # Collate lists of dictionaries, to dictionaries of batched tensors.
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
#         # TODO: Calculate losses per location for classification, box reg and
#         # centerness. Remember to set box/centerness losses for "background"
#         # positions to zero.
#         ######################################################################
#         # Feel free to delete this line: (but keep variable names same)
#         loss_cls, loss_box, loss_ctr = None, None, None


#         ######################################################################
#         #                            END OF YOUR CODE                        #
#         ######################################################################
#         # Sum all locations and average by the EMA of foreground locations.
#         # In training code, we simply add these three and call `.backward()`
#         return {
#             "loss_cls": loss_cls.sum() / (self._normalizer * images.shape[0]),
#             "loss_box": loss_box.sum() / (self._normalizer * images.shape[0]),
#             "loss_ctr": loss_ctr.sum() / (self._normalizer * images.shape[0]),
#         }

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

#     def inference(
#         self,
#         images: torch.Tensor,
#         locations_per_fpn_level: Dict[str, torch.Tensor],
#         pred_cls_logits: Dict[str, torch.Tensor],
#         pred_boxreg_deltas: Dict[str, torch.Tensor],
#         pred_ctr_logits: Dict[str, torch.Tensor],
#         test_score_thresh: float = 0.3,
#         test_nms_thresh: float = 0.5,
#     ):
#         """
#         Run inference on a single input image (batch size = 1). Other input
#         arguments are same as those computed in `forward` method. This method
#         should not be called from anywhere except from inside `forward`.

#         Returns:
#             Three tensors:
#                 - pred_boxes: Tensor of shape `(N, 4)` giving *absolute* XYXY
#                   co-ordinates of predicted boxes.

#                 - pred_classes: Tensor of shape `(N, )` giving predicted class
#                   labels for these boxes (one of `num_classes` labels). Make
#                   sure there are no background predictions (-1).

#                 - pred_scores: Tensor of shape `(N, )` giving confidence scores
#                   for predictions: these values are `sqrt(class_prob * ctrness)`
#                   where class_prob and ctrness are obtained by applying sigmoid
#                   to corresponding logits.
#         """

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

#             ##################################################################
#             # TODO: FCOS uses the geometric mean of class probability and
#             # centerness as the final confidence score. This helps in getting
#             # rid of excessive amount of boxes far away from object centers.
#             # Compute this value here (recall sigmoid(logits) = probabilities)
#             #
#             # Then perform the following steps in order:
#             #   1. Get the most confidently predicted class and its score for
#             #      every box. Use level_pred_scores: (N, num_classes) => (N, )
#             #   2. Only retain prediction that have a confidence score higher
#             #      than provided threshold in arguments.
#             #   3. Obtain predicted boxes using predicted deltas and locations
#             #   4. Clip XYXY box-cordinates that go beyond the height and
#             #      and width of input image.
#             ##################################################################
#             # Feel free to delete this line: (but keep variable names same)
#             level_pred_boxes, level_pred_classes, level_pred_scores = (
#                 None,
#                 None,
#                 None,  # Need tensors of shape: (N, 4) (N, ) (N, )
#             )

#             # Compute geometric mean of class logits and centerness:
#             level_pred_scores = torch.sqrt(
#                 level_cls_logits.sigmoid_() * level_ctr_logits.sigmoid_()
#             )
#             # Step 1:
#             # Replace "pass" statement with your code
#             pass
            
#             # Step 2:
#             # Replace "pass" statement with your code
#             pass

#             # Step 3:
#             # Replace "pass" statement with your code
#             pass

#             # Step 4: Use `images` to get (height, width) for clipping.
#             # Replace "pass" statement with your code
#             pass

#             ##################################################################
#             #                          END OF YOUR CODE                      #
#             ##################################################################

#             pred_boxes_all_levels.append(level_pred_boxes)
#             pred_classes_all_levels.append(level_pred_classes)
#             pred_scores_all_levels.append(level_pred_scores)

#         ######################################################################
#         # Combine predictions from all levels and perform NMS.
#         pred_boxes_all_levels = torch.cat(pred_boxes_all_levels)
#         pred_classes_all_levels = torch.cat(pred_classes_all_levels)
#         pred_scores_all_levels = torch.cat(pred_scores_all_levels)

#         keep = class_spec_nms(
#             pred_boxes_all_levels,
#             pred_scores_all_levels,
#             pred_classes_all_levels,
#             iou_threshold=test_nms_thresh,
#         )
#         pred_boxes_all_levels = pred_boxes_all_levels[keep]
#         pred_classes_all_levels = pred_classes_all_levels[keep]
#         pred_scores_all_levels = pred_scores_all_levels[keep]
#         return (
#             pred_boxes_all_levels,
#             pred_classes_all_levels,
#             pred_scores_all_levels,
#         )

# one_stage_detector.py

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

    def forward(self, images: torch.Tensor, gt_boxes: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        # Extract features using backbone and FPN
        features = self.backbone_fpn(images)  # features is a dict

        # Generate shapes for each FPN level
        shape_per_fpn_level = {level: feat.shape[-2:] for level, feat in features.items()}

        # Generate location coordinates for each FPN level
        locations_per_fpn_level = get_fpn_location_coords(shape_per_fpn_level, self.backbone_fpn.fpn_strides, dtype=features[next(iter(features))].dtype, device=features[next(iter(features))].device)

        # Pass features through prediction heads
        class_logits, boxreg_deltas, centerness_logits = self.prediction_heads(features)

        if self.training:
            # Initialize loss accumulators
            loss_cls_total = 0.0
            loss_box_total = 0.0
            loss_ctr_total = 0.0
            num_pos_total = 0

            B = images.shape[0]
            for b in range(B):
                # Extract ground truth boxes for image b
                gt_b = gt_boxes[b]  # Shape: (Num_GT, 5)

                # Assign GT boxes to locations
                matched_gt_boxes = fcos_match_locations_to_gt(
                    {k: v.clone().detach() for k, v in locations_per_fpn_level.items()},
                    self.backbone_fpn.fpn_strides,
                    gt_b.unsqueeze(0)  # Shape: (1, Num_GT, 5)
                )

                # Should return (1, L, 5)
                matched_gt_boxes = matched_gt_boxes[0]

                # Compute deltas
                deltas = fcos_get_deltas_from_locations(
                    locations_per_fpn_level,
                    matched_gt_boxes,
                    self.backbone_fpn.fpn_strides
                )
                # Shape: (L, 4)

                # Compute centerness targets
                centerness_targets = fcos_make_centerness_targets(deltas)
                # Shape: (L, )

                # Compute classification targets
                cls_targets = matched_gt_boxes[:, 4]
                # Shape: (L, )

                # Compute classification loss
                cls_pred = class_logits['p3'].view(-1, self.num_classes)  # Adjust based on levels
                loss_cls = sigmoid_focal_loss(cls_pred, cls_targets)

                # Compute box regression loss (ignore background)
                fg_mask = cls_targets != -1
                if fg_mask.sum() > 0:
                    box_pred = boxreg_deltas['p3'][fg_mask]  # Shape: (num_fg, 4)
                    deltas_fg = deltas[fg_mask]               # Shape: (num_fg, 4)
                    loss_box = F.l1_loss(box_pred, deltas_fg, reduction='sum')

                    # Compute centerness loss
                    ctr_pred = centerness_logits['p3'][fg_mask].squeeze()         # Shape: (num_fg, )
                    ctr_targets_b = centerness_targets[fg_mask].unsqueeze(1)      # Shape: (num_fg, 1)
                    loss_ctr = F.binary_cross_entropy_with_logits(ctr_pred, ctr_targets_b, reduction='sum')
                else:
                    loss_box = torch.tensor(0.0, device=images.device)
                    loss_ctr = torch.tensor(0.0, device=images.device)

                # Accumulate losses
                loss_cls_total += loss_cls
                loss_box_total += loss_box
                loss_ctr_total += loss_ctr
                num_pos_total += fg_mask.sum().item()

            # Normalize losses
            loss_normalizer = num_pos_total if num_pos_total > 0 else 1.0
            losses = {
                "loss_cls": loss_cls_total / loss_normalizer,
                "loss_box": loss_box_total / loss_normalizer,
                "loss_ctr": loss_ctr_total / loss_normalizer,
            }
            return losses
        else:
            # Inference mode
            pred_boxes_all_levels = []
            pred_classes_all_levels = []
            pred_scores_all_levels = []

            for level_name, cls_logits in class_logits.items():
                # Get box deltas and centerness for the current level
                boxreg_level = boxreg_deltas[level_name].view(-1, 4)  # Shape: (L, 4)
                ctr_level = centerness_logits[level_name].view(-1, 1)  # Shape: (L, 1)
                cls_level = cls_logits.view(-1, self.num_classes)  # Shape: (L, C)

                # Apply sigmoid to centerness and class logits
                ctr_probs = torch.sigmoid(ctr_level)  # Shape: (L, 1)
                cls_probs = torch.sigmoid(cls_level)  # Shape: (L, C)

                # Combine class probabilities and centerness
                confidence_scores = torch.sqrt(cls_probs * ctr_probs)  # Shape: (L, C)

                # Get the top class and score
                scores, classes = confidence_scores.max(dim=1)  # Shape: (L,), (L,)

                # Apply score threshold
                score_mask = scores > self.test_score_thresh
                scores = scores[score_mask]
                classes = classes[score_mask]
                selected_deltas = boxreg_level[score_mask]  # Shape: (N, 4)

                # Get corresponding locations
                locations = locations_per_fpn_level[level_name]  # Shape: (L, 2)
                selected_locations = locations[score_mask]  # Shape: (N, 2)

                # Apply deltas to locations to get predicted boxes
                pred_boxes = fcos_apply_deltas_to_locations(
                    deltas=selected_deltas,
                    locations=selected_locations,
                    stride=self.backbone_fpn.fpn_strides[level_name]
                )  # Shape: (N, 4)

                # Clip XYXY box-coordinates to image boundaries
                img_h, img_w = images.shape[2], images.shape[3]
                pred_boxes[:, 0::2].clamp_(min=0, max=img_w - 1)  # x1, x2
                pred_boxes[:, 1::2].clamp_(min=0, max=img_h - 1)  # y1, y2

                pred_boxes_all_levels.append(pred_boxes)
                pred_classes_all_levels.append(classes)
                pred_scores_all_levels.append(scores)

            # Concatenate all predictions from different FPN levels
            pred_boxes_all_levels = torch.cat(pred_boxes_all_levels, dim=0)
            pred_classes_all_levels = torch.cat(pred_classes_all_levels, dim=0)
            pred_scores_all_levels = torch.cat(pred_scores_all_levels, dim=0)

            # Perform Non-Maximum Suppression (NMS)
            keep = nms(pred_boxes_all_levels, pred_scores_all_levels, iou_threshold=self.test_nms_thresh)

            # Filter the predictions
            pred_boxes_all_levels = pred_boxes_all_levels[keep]
            pred_classes_all_levels = pred_classes_all_levels[keep]
            pred_scores_all_levels = pred_scores_all_levels[keep]

            return (
                pred_boxes_all_levels,
                pred_classes_all_levels,
                pred_scores_all_levels,
            )

    def inference(
        self,
        images: torch.Tensor,
        locations_per_fpn_level: Dict[str, torch.Tensor],
        pred_cls_logits: Dict[str, torch.Tensor],
        pred_boxreg_deltas: Dict[str, torch.Tensor],
        pred_ctr_logits: Dict[str, torch.Tensor],
        test_score_thresh: float = 0.3,
        test_nms_thresh: float = 0.5,
    ):
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
            level_cls_logits = pred_cls_logits[level_name].squeeze(-1)  # (B, H*W, num_classes)
            level_deltas = pred_boxreg_deltas[level_name].squeeze(-1)  # (B, H*W, 4)
            level_ctr_logits = pred_ctr_logits[level_name].squeeze(-1)  # (B, H*W, 1)

            # Iterate over the batch
            for i in range(images.shape[0]):
                # Get per-image predictions
                cls_logits = level_cls_logits[i]  # (H*W, num_classes)
                deltas = level_deltas[i]  # (H*W, 4)
                ctr_logits = level_ctr_logits[i]  # (H*W, 1)

                # Compute sigmoid probabilities
                cls_probs = torch.sigmoid(cls_logits)  # (H*W, num_classes)
                ctr_probs = torch.sigmoid(ctr_logits)  # (H*W, 1)
                confidence_scores = torch.sqrt(cls_probs * ctr_probs)  # (H*W, num_classes)

                scores, classes = confidence_scores.max(dim=1)  # (H*W,), (H*W,)

                # Apply score threshold
                score_mask = scores > test_score_thresh
                scores = scores[score_mask]
                classes = classes[score_mask]
                selected_deltas = deltas[score_mask]

                # Get corresponding locations
                selected_locations = locations_per_fpn_level[level_name][score_mask]  # (N, 2)

                # Apply deltas to locations to get predicted boxes
                pred_boxes = fcos_apply_deltas_to_locations(
                    deltas=selected_deltas,
                    locations=selected_locations,
                    stride=self.backbone_fpn.fpn_strides[level_name]
                )  # (N, 4)

                # Clip XYXY box-coordinates that go beyond the height and width of input image.
                img_h, img_w = images.shape[2], images.shape[3]
                pred_boxes[:, 0::2].clamp_(min=0, max=img_w - 1)  # x1, x2
                pred_boxes[:, 1::2].clamp_(min=0, max=img_h - 1)  # y1, y2

                pred_boxes_all_levels.append(pred_boxes)
                pred_classes_all_levels.append(classes)
                pred_scores_all_levels.append(scores)

        ######################################################################
        # Combine predictions from all levels and perform NMS.
        ######################################################################
        # Concatenate all predictions from different FPN levels
        pred_boxes_all_levels = torch.cat(pred_boxes_all_levels, dim=0)
        pred_classes_all_levels = torch.cat(pred_classes_all_levels, dim=0)
        pred_scores_all_levels = torch.cat(pred_scores_all_levels, dim=0)

        # Perform Class-wise NMS
        keep = nms(pred_boxes_all_levels, pred_scores_all_levels, iou_threshold=test_nms_thresh)

        # Filter the predictions
        pred_boxes_all_levels = pred_boxes_all_levels[keep]
        pred_classes_all_levels = pred_classes_all_levels[keep]
        pred_scores_all_levels = pred_scores_all_levels[keep]

        return (
            pred_boxes_all_levels,
            pred_classes_all_levels,
            pred_scores_all_levels,
        )