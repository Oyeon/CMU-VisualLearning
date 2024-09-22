"""
This module contains classes and functions that are common across both, one-stage
and two-stage detector implementations. You have to implement some parts here -
walk through the notebooks and you will find instructions on *when* to implement
*what* in this module.
"""
from typing import Dict, Tuple, List, Optional

import torch
from torch import nn
from torch.nn import functional as F

from torchvision.ops import nms as nms_torch

# Short hand type notation:
TensorDict = Dict[str, torch.Tensor]

@torch.no_grad()
def fcos_match_locations_to_gt(
    locations_per_fpn_level: TensorDict,
    strides_per_fpn_level: Dict[str, int],
    gt_boxes: torch.Tensor,
) -> TensorDict:
    """
    Match centers of the locations of FPN feature with a set of GT bounding
    boxes of the input image. Since our model makes predictions at every FPN
    feature map location, we must supervise it with an appropriate GT box.
    There are multiple GT boxes in image, so FCOS has a set of heuristics to
    assign centers with GT, which we implement here.

    NOTE: This function is NOT BATCHED. Call separately for GT box batches.

    Args:
        locations_per_fpn_level: Centers at different levels of FPN (p3, p4, p5),
            that are already projected to absolute co-ordinates in input image
            dimension. Dictionary of three keys: (p3, p4, p5) giving tensors of
            shape `(H * W, 2)` where H = W is the size of feature map.
        strides_per_fpn_level: Dictionary of same keys as above, each with an
            integer value giving the stride of corresponding FPN level.
            See `common.py` for more details.
        gt_boxes: GT boxes of a single image, a batch of `(M, 5)` boxes with
            absolute co-ordinates and class ID `(x1, y1, x2, y2, C)`. In this
            codebase, this tensor is directly served by the dataloader.

    Returns:
        Dict[str, torch.Tensor]
            Dictionary with same keys as `shape_per_fpn_level` and values as
            tensors of shape `(N, 5)` GT boxes, one for each center. They are
            one of M input boxes, or a dummy box called "background" that is
            `(-1, -1, -1, -1, -1)`. Background indicates that the center does
            not belong to any object.
    """

    matched_gt_boxes = {
        level_name: None for level_name in locations_per_fpn_level.keys()
    }

    # Do this matching individually per FPN level.
    for level_name, centers in locations_per_fpn_level.items():

        # Get stride for this FPN level.
        stride = strides_per_fpn_level[level_name]

        x, y = centers.unsqueeze(dim=2).unbind(dim=1)
        x0, y0, x1, y1 = gt_boxes[:, :4].unsqueeze(dim=0).unbind(dim=2)
        pairwise_dist = torch.stack([x - x0, y - y0, x1 - x, y1 - y], dim=2)

        # Pairwise distance between every feature center and GT box edges:
        # shape: (num_gt_boxes, num_centers_this_level, 4)
        pairwise_dist = pairwise_dist.permute(1, 0, 2)

        # The original FCOS anchor matching rule: anchor point must be inside GT.
        match_matrix = pairwise_dist.min(dim=2).values > 0

        # Multilevel anchor matching in FCOS: each anchor is only responsible
        # for certain scale range.
        # Decide upper and lower bounds of limiting targets.
        pairwise_dist = pairwise_dist.max(dim=2).values

        lower_bound = stride * 4 if level_name != "p3" else 0
        upper_bound = stride * 8 if level_name != "p5" else float("inf")
        match_matrix &= (pairwise_dist > lower_bound) & (
            pairwise_dist < upper_bound
        )

        # Match the GT box with minimum area, if there are multiple GT matches.
        gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (
            gt_boxes[:, 3] - gt_boxes[:, 1]
        )

        # Get matches and their labels using match quality matrix.
        match_matrix = match_matrix.to(torch.float32)
        match_matrix *= 1e8 - gt_areas[:, None]

        # Find matched ground-truth instance per anchor (un-matched = -1).
        match_quality, matched_idxs = match_matrix.max(dim=0)
        matched_idxs[match_quality < 1e-5] = -1

        # Anchors with label 0 are treated as background.
        matched_boxes_this_level = gt_boxes[matched_idxs.clip(min=0)]
        matched_boxes_this_level[matched_idxs < 0, :] = -1

        matched_gt_boxes[level_name] = matched_boxes_this_level

    return matched_gt_boxes


def fcos_get_deltas_from_locations(
    locations: torch.Tensor, gt_boxes: torch.Tensor, stride: int
) -> torch.Tensor:
    """
    [Existing Docstring]
    """
    # Determine if gt_boxes include class labels
    if gt_boxes.shape[1] == 5:
        # Background boxes have class label -1
        background_mask = gt_boxes[:, 4] == -1
        boxes = gt_boxes[:, :4]
    else:
        # Assume all boxes are foreground unless all elements are -1
        background_mask = torch.all(gt_boxes == -1, dim=1)
        boxes = gt_boxes

    # Initialize deltas tensor
    deltas = torch.zeros((locations.size(0), 4), dtype=locations.dtype, device=locations.device)

    # Assign -1 to deltas for background boxes
    deltas[background_mask] = -1

    # Compute deltas for foreground boxes
    if (~background_mask).any():
        # Select foreground locations and boxes
        fg_locations = locations[~background_mask]
        fg_boxes = boxes[~background_mask]

        # Compute distances
        left = fg_locations[:, 0] - fg_boxes[:, 0]
        top = fg_locations[:, 1] - fg_boxes[:, 1]
        right = fg_boxes[:, 2] - fg_locations[:, 0]
        bottom = fg_boxes[:, 3] - fg_locations[:, 1]

        # Normalize by stride
        normalized_left = left / stride
        normalized_top = top / stride
        normalized_right = right / stride
        normalized_bottom = bottom / stride

        # Stack into deltas
        fg_deltas = torch.stack((normalized_left, normalized_top, normalized_right, normalized_bottom), dim=1)

        # Assign to deltas tensor
        deltas[~background_mask] = fg_deltas

    return deltas


def fcos_apply_deltas_to_locations(
    deltas: torch.Tensor, locations: torch.Tensor, stride: int
) -> torch.Tensor:
    """
    Implement the inverse of `fcos_get_deltas_from_locations` here:

    Given edge deltas (left, top, right, bottom) and feature locations of FPN, get
    the resulting bounding box co-ordinates by applying deltas on locations. This
    method is used for inference in FCOS: deltas are outputs from model, and
    applying them to anchors will give us final box predictions.

    Recall in above method, we were required to normalize the deltas by feature
    stride. Similarly, we have to un-normalize the input deltas with feature
    stride before applying them to locations, because the given input locations are
    already absolute co-ordinates in image dimensions.

    Args:
        deltas: Tensor of shape `(N, 4)` giving edge deltas to apply to locations.
        locations: Locations to apply deltas on. shape: `(N, 2)`
        stride: Stride of the FPN feature map.

    Returns:
        torch.Tensor
            Same shape as deltas and locations, giving co-ordinates of the
            resulting boxes `(x1, y1, x2, y2)`, absolute in image dimensions.
    """
    # Initialize output_boxes tensor
    output_boxes = torch.zeros_like(deltas)

    # Clip deltas to ensure feature center lies inside the box
    deltas_clipped = torch.clamp(deltas, min=0)

    # Un-normalize deltas
    deltas_unorm = deltas_clipped * stride

    # Extract center locations
    xc = locations[:, 0]
    yc = locations[:, 1]

    # Compute box coordinates
    x1 = xc - deltas_unorm[:, 0]
    y1 = yc - deltas_unorm[:, 1]
    x2 = xc + deltas_unorm[:, 2]
    y2 = yc + deltas_unorm[:, 3]

    # Stack coordinates into (x1, y1, x2, y2)
    output_boxes = torch.stack((x1, y1, x2, y2), dim=1)


    return output_boxes


def fcos_make_centerness_targets(deltas: torch.Tensor):
    """
    [Existing Docstring]
    """
    
    centerness = torch.zeros(deltas.size(0), dtype=deltas.dtype, device=deltas.device)

    # Identify background boxes where deltas are [-1, -1, -1, -1]
    background_mask = torch.all(deltas == -1, dim=1)

    # Compute centerness for foreground boxes
    if (~background_mask).any():
        fg_deltas = deltas[~background_mask]

        left = fg_deltas[:, 0]
        top = fg_deltas[:, 1]
        right = fg_deltas[:, 2]
        bottom = fg_deltas[:, 3]

        min_lr = torch.min(left, right)
        min_tb = torch.min(top, bottom)
        max_lr = torch.max(left, right)
        max_tb = torch.max(top, bottom)

        # Avoid division by zero -> infinity
        eps = 1e-6
        centerness_fg = torch.sqrt((min_lr * min_tb) / (max_lr * max_tb + eps))

        # Assign to centerness
        centerness[~background_mask] = centerness_fg

    # Assign -1 for background boxes
    centerness[background_mask] = -1.0

    # Add batch dimension
    centerness = centerness.unsqueeze(0)  # Shape: (1, N)

    return centerness

def get_fpn_location_coords(
    shape_per_fpn_level: Dict[str, Tuple],
    strides_per_fpn_level: Dict[str, int],
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    [Existing Docstring]
    """

    # Set these to `(N, 2)` Tensors giving absolute location co-ordinates.
    location_coords = {
        level_name: None for level_name, _ in shape_per_fpn_level.items()
    }

    for level_name, feat_shape in shape_per_fpn_level.items():
        level_stride = strides_per_fpn_level[level_name]
        B, C, H, W = feat_shape  # Unpack feature shape

        # Generate center coordinates
        # x coordinates: (0.5 * stride, 1.5 * stride, ..., (W - 0.5) * stride)
        x_coords = (torch.arange(W, dtype=dtype, device=device) + 0.5) * level_stride
        # y coordinates: (0.5 * stride, 1.5 * stride, ..., (H - 0.5) * stride)
        y_coords = (torch.arange(H, dtype=dtype, device=device) + 0.5) * level_stride

        # Repeat x_coords for each y and tile y_coords for each x
        x_repeat = x_coords.repeat_interleave(H)  # (W*H,)
        y_tile = y_coords.repeat(W)               # (W*H,)

        # Stack to get (W*H, 2) tensor of (xc, yc)
        coords = torch.stack((x_repeat, y_tile), dim=1)

        # Assign to the dictionary
        location_coords[level_name] = coords

    return location_coords

def class_spec_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    iou_threshold: float = 0.5,
):
    """
    Wrap `nms` to make it class-specific. Pass class IDs as `class_ids`.
    STUDENT: This depends on your `nms` implementation.

    Returns:
        keep: torch.long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores;
            of shape [num_kept_boxes]
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coordinate = boxes.max()
    offsets = class_ids.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
    boxes_for_nms = boxes + offsets[:, None]
    # Use torchvision NMS.
    keep = nms_torch(boxes_for_nms, scores, iou_threshold)
    return keep
