"""
You will use this script to train your object detection model.

Arguments:
--overfit: This argument is default on, and will run training on a small
           subset of images. This is to test whether the model can overfit
           to a small dataset.
--inference: This argument is whether to run model inference or not. You 
           use this flag to test model training before you implement the 
           inference component.
--test_inference: This argument is default on, and will run inference on a
           small subset of images. This is to test whether the model can
           successfully perform inference and to check whether predictions
           are reasonable. If this flag is off, inference will be run on the
           entire validation set, and mAP will be computed.
"""

import argparse
from dataclasses import dataclass
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torchvision.utils import make_grid

from detection_helper import train_detector, inference_with_detector
from detection_helper import VOC2007DetectionTiny
from one_stage_detector import FCOS
from utils.utils import detection_visualizer

if torch.cuda.is_available():
    print("Good to go!")
    DEVICE = torch.device("cuda")
else:
    print("Please check your GPU (if running on AWS). Using CPU instead.")
    DEVICE = torch.device("cpu")

NUM_CLASSES = 20
BATCH_SIZE = 512
IMAGE_SHAPE = (224, 224)
NUM_WORKERS = 12
DATASET_PATH = "../data"

@dataclass
class HyperParameters:
    """
    Hyperparameters for training.
    """
    num_classes: int = NUM_CLASSES
    batch_size: int = BATCH_SIZE
    num_workers: int = NUM_WORKERS
    image_shape: tuple = IMAGE_SHAPE
    lr: float = 1e-4
    log_period: int = 100
    max_iters: int = 9000
    device: str = DEVICE

def create_dataset_and_dataloaders(subset=False):
    train_dataset = VOC2007DetectionTiny(
        DATASET_PATH, "train", image_size=IMAGE_SHAPE[0],
        download=True# True (set to False after the first time)
    )
    print(f"Loaded {len(train_dataset)} training samples.")
    if subset:
        small_dataset = torch.utils.data.Subset(
            train_dataset,
            torch.linspace(0, len(train_dataset) - 1, steps=BATCH_SIZE * 10).long()
        )
        train_dataset = small_dataset
        print(f"Using a subset of {len(train_dataset)} training samples.")

    val_dataset = VOC2007DetectionTiny(DATASET_PATH, "val", image_size=IMAGE_SHAPE[0])
    print(f"Loaded {len(val_dataset)} validation samples.")
    # `pin_memory` speeds up CPU-GPU batch transfer, `num_workers=NUM_WORKERS` loads data
    # on the main CPU process, suitable for Colab.
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, pin_memory=True, num_workers=NUM_WORKERS
    )

    # Use batch_size = 1 during inference - during inference we do not center crop
    # the image to detect all objects, hence they may be of different size. It is
    # easier and less redundant to use batch_size=1 rather than zero-padding images.
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, pin_memory=True, num_workers=NUM_WORKERS
    )

    return train_loader, val_loader, train_dataset, val_dataset

def train_model(detector, train_loader, hyperparams, overfit=False):
    device = hyperparams.device
    detector = detector.to(device)

    print("Starting training...")
    train_detector(
        detector,
        train_loader,
        learning_rate=hyperparams.lr,
        max_iters=hyperparams.max_iters,
        log_period=hyperparams.log_period,
        device=hyperparams.device,
        overfit=overfit
    )
    print("Successfully finished training.")
    return

def visualize_gt(train_dataset, val_dataset):
    writer = SummaryWriter("detection_logs")
    inverse_norm = transforms.Compose(
        [
            transforms.Normalize(mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
        ]
    )
    gt_images = []
    for idx, (_, image, gt_boxes) in enumerate(train_dataset):
        # Stop after visualizing three images.
        if idx > 2:
            break
        # Un-normalize image to bring in [0, 1] RGB range.
        image = inverse_norm(image)
        # Remove padded boxes from visualization.
        is_valid = gt_boxes[:, 4] >= 0
        img = detection_visualizer(image, val_dataset.idx_to_class, gt_boxes[is_valid])
        gt_images.append(torch.from_numpy(img))
    
    img_grid = make_grid(gt_images, nrow=8)
    writer.add_image("train/gt_images", img_grid, global_step=idx)
    writer.close()

def main(args):
    print("Loading data...")
    
    if args.overfit:
        print("Loading a small subset for overfitting.")
    train_loader, val_loader, train_dataset, val_dataset = create_dataset_and_dataloaders(args.overfit)
    
    if args.overfit:
        hyperparams = HyperParameters(
            max_iters=250,
            lr=1e-3,  # Reduce from 1e-2 to 1e-3
            log_period=10,
        )
    else:
        hyperparams = HyperParameters(
            lr=1e-3,  # Reduce from 1e-2 to 1e-3
            max_iters=9000*3,
            log_period=100,
        )
    detector = FCOS(
        num_classes=NUM_CLASSES,
        fpn_channels=256,
        stem_channels=[256, 256],
    )
    print("Model initialized.")

    if args.visualize_gt:
        print("Visualizing GT...")
        visualize_gt(train_dataset, val_dataset)        
        return
    
    print("Training model...")
    if not args.visualize_gt and not args.inference:
        train_model(detector, train_loader, hyperparams, overfit=args.overfit)
    print("Training complete!")
    if not args.inference:
        return
    print("Running inference...")
    if args.test_inference:
        small_dataset = torch.utils.data.Subset(
            val_dataset,
            torch.linspace(0, len(val_dataset) - 1, steps=10).long()
        )
        small_val_loader = torch.utils.data.DataLoader(
            small_dataset, batch_size=1, pin_memory=True, num_workers=NUM_WORKERS
        )        
        # Modify this depending on where you save your weights.
        weights_path = os.path.join(".", "fcos_detector.pt")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found at {weights_path}")

        # Re-initialize so this cell is independent from prior cells.
        detector = FCOS(
            num_classes=NUM_CLASSES, fpn_channels=256, stem_channels=[256, 256]
        )
        detector.to(device=DEVICE)
        detector.load_state_dict(torch.load(weights_path, map_location="cpu"))
        print("Generating example inference images...")
        inference_with_detector(
            detector,
            small_val_loader,
            val_dataset.idx_to_class,
            score_thresh=0.05,
            nms_thresh=0.3,
            device=DEVICE,
            dtype=torch.float32,
        )
        print("Model loaded successfully.")
    else:
        print("Running inference and computing mAP...")

        ############################################################
        ### TODO can be removed after testing
        from PIL import Image

        small_dataset = torch.utils.data.Subset(
            val_dataset,
            torch.linspace(0, len(val_dataset) - 1, steps=10).long()
        )
        small_val_loader = torch.utils.data.DataLoader(
            small_dataset, batch_size=1, pin_memory=True, num_workers=NUM_WORKERS
        )

        # Sample image for inference
        # sample_image_path = "/home/fluent/assignment-visual/lab1/reference/data/VOCdevkit/VOC2007/JPEGImages/000033.jpg"
        # image = Image.open(sample_image_path).convert("RGB")
        # preprocessed_image = preprocess_image(image, DEVICE)
        # visualize_preprocessing(image, preprocessed_image)

        # Load and prepare the model
        weights_path = os.path.join(".", "fcos_detector.pt")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found at {weights_path}")

        detector = FCOS(
            num_classes=NUM_CLASSES, fpn_channels=256, stem_channels=[256, 256]
        ).to(device=DEVICE)

        detector.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        detector.eval()
        print("Model loaded successfully.")

        # Check device consistency
        check_model_device(detector, DEVICE)

        # Verify input tensor device
        # print(f"Preprocessed image is on device: {preprocessed_image.device}")

        # Proceed with inference
        # with torch.no_grad():
        #     outputs = detector(preprocessed_image)
        #     print("Model output:", outputs)

        # Visualize detections
        # visualize_detections(image, outputs, val_dataset.idx_to_class, score_thresh=0.05, save_path="test.png")
        ############################################################

        assert os.path.exists("mAP")
        # Modify this depending on where you save your weights.
        weights_path = os.path.join(".", "fcos_detector.pt")
        detector.to(device=DEVICE)
        detector.load_state_dict(torch.load(weights_path, map_location="cpu"))
        inference_with_detector(
            detector,
            val_loader,
            val_dataset.idx_to_class,
            score_thresh=0.01,
            nms_thresh=0.1,
            device=DEVICE,
            dtype=torch.float32,
            output_dir="mAP/input",
        )
        os.system("cd mAP && python main.py")
        print("Output file written to ./mAP/output/mAP.png")
        

def check_model_device(model, device):
    for name, param in model.named_parameters():
        if param.device != device:
            print(f"Parameter {name} is on {param.device}, expected {device}")


def visualize_detections(image, outputs, idx_to_class, score_thresh=0.05, save_path=None):
    import matplotlib.patches as patches

    fig, ax = plt.subplots(1)
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
        ax.text(x_min, y_min - 10, f"{idx_to_class[label]}: {score:.2f}", 
                bbox=dict(facecolor='yellow', alpha=0.5), fontsize=8, color='black')
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def inspect_model_outputs(detector, image, device):
    preprocessed_image = preprocess_image(image, device)
    with torch.no_grad():
        outputs = detector(preprocessed_image)
    
    # Example structure of outputs (adjust based on your model's actual output)
    # outputs = {
    #     'boxes': tensor([...]),
    #     'scores': tensor([...]),
    #     'labels': tensor([...])
    # }
    print("Raw outputs:", outputs)
    
    # Check if outputs contain expected keys
    for key in ['boxes', 'scores', 'labels']:
        if key not in outputs:
            print(f"Warning: Output does not contain key '{key}'.")
        else:
            print(f"{key}: {outputs[key].shape}")
    
    return outputs

def preprocess_image(image, device):
    # Example preprocessing steps
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SHAPE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Same as training
                             std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.to(device)

def visualize_preprocessing(image, preprocessed_image):
    # Convert tensor to numpy for visualization
    inv_norm = transforms.Compose(
        [
            transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]),
        ]
    )
    image_inv = inv_norm(preprocessed_image.squeeze()).permute(1, 2, 0).cpu().numpy()
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title("Preprocessed Image")
    plt.imshow(image_inv)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize_gt", action="store_true")
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--inference", action="store_true")
    parser.add_argument("--test_inference", action="store_true")
    args = parser.parse_args()
    print(args.visualize_gt)
    print(args.overfit)
    print(args.inference)
    print(args.test_inference)
    main(args)


