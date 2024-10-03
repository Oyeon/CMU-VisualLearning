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
    Match centers of the locations of FPN feature maps with the ground truth bounding
    boxes in the input image. This function assigns each location on the FPN feature maps
    to a ground truth box or marks it as background based on FCOS matching rules.

    Args:
        locations_per_fpn_level: A dictionary with keys as FPN levels ('p3', 'p4', 'p5')
            and values as tensors of shape `(num_locations, 2)` representing the (x, y)
            coordinates of feature map locations projected to the input image space.
        strides_per_fpn_level: A dictionary with the same keys as above, each with an
            integer value indicating the stride of the corresponding FPN level.
        gt_boxes: A tensor of shape `(num_gt_boxes, 5)` representing the ground truth
            boxes in the image, where each box is `(x1, y1, x2, y2, class_id)`.

    Returns:
        matched_gt_boxes: A dictionary with the same keys as `locations_per_fpn_level`.
            Each value is a tensor of shape `(num_locations, 5)` where each location
            is assigned a ground truth box or marked as background with `-1`.
    """

    matched_gt_boxes = {}

    # Iterate over each FPN level
    for level_name, centers in locations_per_fpn_level.items():
        stride = strides_per_fpn_level[level_name]

        # Prepare centers and ground truth boxes
        num_centers = centers.size(0)
        num_gt_boxes = gt_boxes.size(0)

        # Expand centers and gt_boxes to compute pairwise distances
        centers_expanded = centers.unsqueeze(1).expand(num_centers, num_gt_boxes, 2)
        gt_boxes_expanded = gt_boxes[:, :4].unsqueeze(0).expand(num_centers, num_gt_boxes, 4)

        # Compute l, t, r, b distances
        l = centers_expanded[:, :, 0] - gt_boxes_expanded[:, :, 0]  # x - x0
        t = centers_expanded[:, :, 1] - gt_boxes_expanded[:, :, 1]  # y - y0
        r = gt_boxes_expanded[:, :, 2] - centers_expanded[:, :, 0]  # x1 - x
        b = gt_boxes_expanded[:, :, 3] - centers_expanded[:, :, 1]  # y1 - y

        # Stack distances to shape (num_centers, num_gt_boxes, 4)
        distances = torch.stack([l, t, r, b], dim=2)

        # Determine if centers are inside gt_boxes
        inside_gt_boxes = distances.min(dim=2).values > 0  # Shape: (num_centers, num_gt_boxes)

        # Compute the maximum distance to the borders for each center-gt_box pair
        max_distances = distances.max(dim=2).values  # Shape: (num_centers, num_gt_boxes)

        # Define scale range for current FPN level
        if level_name == "p3":
            lower_bound = 0
            upper_bound = stride * 8
        elif level_name == "p4":
            lower_bound = strides_per_fpn_level["p3"] * 8
            upper_bound = stride * 8
        elif level_name == "p5":
            lower_bound = strides_per_fpn_level["p4"] * 8
            upper_bound = float("inf")
        else:
            raise ValueError(f"Unknown FPN level: {level_name}")

        # Determine if centers are within the scale range
        in_scale_range = (max_distances >= lower_bound) & (max_distances <= upper_bound)

        # Combine conditions to create match matrix
        match_matrix = inside_gt_boxes & in_scale_range  # Shape: (num_centers, num_gt_boxes)

        # Compute areas of gt_boxes for tie-breaking
        gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])  # Shape: (num_gt_boxes,)

        # Expand gt_areas to match the shape of match_matrix
        gt_areas_expanded = gt_areas.unsqueeze(0).expand(num_centers, num_gt_boxes)

        # Set unmatched positions to a high area value
        INF = 1e8
        match_matrix = match_matrix.to(torch.float32)
        match_matrix = match_matrix * (INF - gt_areas_expanded)

        # Find the gt_box with the smallest area for each center
        match_quality, matched_idxs = match_matrix.max(dim=1)  # Shape: (num_centers,)

        # Identify background positions
        matched_idxs[match_quality == 0] = -1  # Background positions

        # Prepare matched_gt_boxes for current level
        matched_boxes = gt_boxes.new_full((num_centers, 5), -1)  # Initialize as background
        pos_indices = matched_idxs >= 0  # Positions with positive matches
        if pos_indices.any():
            matched_boxes[pos_indices] = gt_boxes[matched_idxs[pos_indices]]

        # Save matched boxes for current level
        matched_gt_boxes[level_name] = matched_boxes

        # Logging matched positives
        num_matched = pos_indices.sum().item()
        # print(f"Level {level_name}, matched positives: {num_matched}")

    return matched_gt_boxes




def fcos_get_deltas_from_locations(
    locations: torch.Tensor, gt_boxes: torch.Tensor, stride: int
) -> torch.Tensor:
    """
    Computes the deltas (l, t, r, b) between feature locations and GT boxes,
    normalized by the stride.

    Args:
        locations (torch.Tensor): Tensor of shape `(N, 2)` representing feature locations.
        gt_boxes (torch.Tensor): Tensor of shape `(N, 5)` representing matched GT boxes `(x1, y1, x2, y2, C)`.
        stride (int): Stride of the FPN level.

    Returns:
        torch.Tensor: Tensor of shape `(N, 4)` containing normalized deltas.
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


def fcos_make_centerness_targets(deltas: torch.Tensor) -> torch.Tensor:
    """
    Computes the centerness targets based on the deltas.

    Args:
        deltas (torch.Tensor): Tensor of shape `(N, 4)` containing deltas `(l, t, r, b)`.

    Returns:
        torch.Tensor: Tensor of shape `(N,)` containing centerness scores.
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

    return centerness  # Shape: (N,)

def get_fpn_location_coords(
    shape_per_fpn_level: Dict[str, Tuple],
    strides_per_fpn_level: Dict[str, int],
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    [Existing Docstring]
    """
    
    # Set these to `(N, 2)` Tensors giving absolute location coordinates.
    location_coords = {
        level_name: None for level_name, _ in shape_per_fpn_level.items()
    }

    for level_name, feat_shape in shape_per_fpn_level.items():
        level_stride = strides_per_fpn_level[level_name]
        B, C, H, W = feat_shape  # Unpack feature shape

        # Generate center coordinates
        x_coords = (torch.arange(W, dtype=dtype, device=device) + 0.5) * level_stride
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
