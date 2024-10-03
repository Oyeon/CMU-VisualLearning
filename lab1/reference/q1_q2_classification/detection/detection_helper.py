
from typing import Optional

import pdb
import json
import os
import shutil
import time

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import optim
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

from detection.utils.utils import detection_visualizer
import wget

def visualize_detections(image, outputs, score_thresh=0.05, save_path=None):
    import matplotlib.patches as patches

    fig, ax = plt.subplots(1)
    # Normalize image to [0, 1] range
    image = (image - image.min()) / (image.max() - image.min())
    ax.imshow(image)

    boxes, labels, scores = outputs

    # Move tensors to CPU and convert to numpy
    boxes = boxes.cpu().numpy()
    labels = labels.cpu().numpy()
    scores = scores.cpu().numpy()

    for box, score, label in zip(boxes, scores, labels):
        if score < score_thresh:
            continue
        x_min, y_min, x_max, y_max = box
        width, height = x_max - x_min, y_max - y_min
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x_min, y_min - 10, f"{score:.2f}", 
                bbox=dict(facecolor='yellow', alpha=0.5), fontsize=8, color='black')
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    plt.close(fig)



class VOC2007DetectionTiny(torch.utils.data.Dataset):
    """
    A tiny version of the PASCAL VOC 2007 Detection dataset that includes images and
    annotations with small images and no difficult boxes.
    """

    CLASS_NAMES = [
        "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
        "car", "cat", "chair", "cow", "diningtable", "dog",
        "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"
    ]

    def __init__(
        self,
        dataset_dir: str,
        split: str = "train",
        download: bool = False,
        image_size: int = 224,
    ):
        """
        Args:
            dataset_dir (str): Directory where the dataset is stored or will be downloaded to.
            split (str): Dataset split - 'train', 'val', 'trainval', or 'test'.
            download (bool): Whether to download PASCAL VOC 2007 to `dataset_dir`.
            image_size (int): Size to resize the shorter edge of images before center cropping.
        """
        super().__init__()
        self.dataset_dir = dataset_dir
        self.split = split
        self.image_size = image_size

        if download:
            self._attempt_download()

        # Paths to JSON files
        if split == "train":
            json_path = os.path.join(dataset_dir, "voc07_trainval.json")
        elif split == "val":
            json_path = os.path.join(dataset_dir, "voc07_val.json")
        else:
            raise ValueError(f"Unknown split: {split}")

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file {json_path} does not exist. Ensure you've downloaded or merged the splits correctly.")

        # Load annotations
        with open(json_path, 'r') as f:
            self.annotations = json.load(f)

        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        image_path = os.path.join(self.dataset_dir, annotation['image'])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        gt_boxes = []
        gt_classes = []
        for box in annotation['boxes']:
            gt_boxes.append(box['xyxy'])
            gt_classes.append(self.CLASS_NAMES.index(box['name']))

        gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32)
        gt_classes = torch.tensor(gt_classes, dtype=torch.float32).unsqueeze(1)

        # Concatenate GT classes with GT boxes; shape: (N, 5)
        gt_boxes = torch.cat([gt_boxes, gt_classes], dim=1)  # Shape: (N, 5)

        # Handle invalid boxes where x1 > x2 or y1 > y2
        invalid = (gt_boxes[:, 0] > gt_boxes[:, 2]) | (gt_boxes[:, 1] > gt_boxes[:, 3])
        gt_boxes[invalid] = -1.0

        # Pad to a maximum of 40 boxes per image
        num_required = 40
        num_current = gt_boxes.shape[0]
        if num_current < num_required:
            padding = torch.full((num_required - num_current, 5), -1.0)
            gt_boxes = torch.cat([gt_boxes, padding], dim=0)
        else:
            gt_boxes = gt_boxes[:num_required]

        return image, gt_boxes

    def _attempt_download(self):
        """
        Attempt to download the dataset. Implementation depends on your data source.
        """
        os.makedirs(self.dataset_dir, exist_ok=True)
        # Correct URLs for the required JSON files
        train_val_url = "https://web.eecs.umich.edu/~justincj/data/voc07_trainval.json"
        val_url = "https://web.eecs.umich.edu/~justincj/data/voc07_val.json"

        train_val_json_path = os.path.join(self.dataset_dir, "voc07_trainval.json")
        val_json_path = os.path.join(self.dataset_dir, "voc07_val.json")

        print("Downloading voc07_trainval.json...")
        wget.download(train_val_url, out=train_val_json_path)

        print("\nDownloading voc07_val.json...")
        wget.download(val_url, out=val_json_path)

        # Merge train and val JSON files into trainval
        print("\nMerging voc07_train.json and voc07_val.json into voc07_trainval.json...")
        with open(train_val_json_path, 'r') as f:
            train_data = json.load(f)
        with open(val_json_path, 'r') as f:
            val_data = json.load(f)

        merged_data = train_data + val_data

        with open(train_val_json_path, 'w') as f:
            json.dump(merged_data, f)

        print(f"Merged {len(train_data)} training and {len(val_data)} validation annotations into voc07_trainval.json")