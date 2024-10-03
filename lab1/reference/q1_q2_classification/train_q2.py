import torch
import torch.nn as nn
import torchvision.models as models
from voc_dataset import VOCDataset

class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        # Get the number of input features for the last fully connected layer
        num_ftrs = self.resnet.fc.in_features
        # Replace the last fully connected layer with a new one with num_classes outputs
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)

if __name__ == "__main__":
    import trainer
    from utils import ARGS
    import numpy as np
    import random

    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)

    args = ARGS(
        epochs=50,
        inp_size=224,
        use_cuda=True,
        val_every=30,
        lr=0.001,
        batch_size=512,
        step_size=10,
        gamma=0.1
    )

    print(args)

    # Initializes the model
    model = ResNet(num_classes=len(VOCDataset.CLASS_NAMES)).to(args.device)
    # Initializes Adam optimizer and simple StepLR scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # Trains model using your training code and reports test map
    test_ap, test_map = trainer.train(args, model, optimizer, scheduler)
    print('test map:', test_map)

    # Save the final trained model
    FINAL_MODEL_PATH = './trained_model.pth'
    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    print(f"Final model saved at {FINAL_MODEL_PATH}")
    

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################

# import torch
# import torch.nn as nn
# from utils import get_data_loader, ARGS
# from detection.detection_helper import VOC2007DetectionTiny  # Updated import
# from detection.one_stage_detector import FCOS  # Ensure this path is correct

# class FCOSModel(nn.Module):
#     def __init__(self, num_classes: int, in_channels: int = 256, stem_channels: list = [256, 256, 256]):
#         super(FCOSModel, self).__init__()
#         self.fcos = FCOS(
#             num_classes=num_classes,
#             fpn_channels=in_channels,
#             stem_channels=stem_channels
#         )

#     def forward(self, images: torch.Tensor, gt_boxes: torch.Tensor = None, **kwargs) -> dict:
#         return self.fcos(images, gt_boxes, **kwargs)  # Pass gt_boxes as positional argument

# if __name__ == "__main__":
#     import trainer  # Ensure trainer.py is in the correct path
#     import numpy as np
#     import random

#     # Set random seeds for reproducibility
#     np.random.seed(0)
#     torch.manual_seed(0)
#     random.seed(0)

#     # Define training arguments
#     args = ARGS(
#         epochs=10,
#         inp_size=224,
#         use_cuda=torch.cuda.is_available(),  # Automatically detect CUDA
#         device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
#         val_every=100,  # Adjust as needed
#         lr=0.001,
#         batch_size=32,
#         test_batch_size=32,  # Ensure test_batch_size is defined
#         step_size=5,
#         gamma=0.1,
#         save_freq=5,
#         save_at_end=True,
#         log_every=100  # Define how often to log (adjust as needed)
#     )

#     print(args)

#     # Initialize the FCOS model
#     # Correctly determine the number of classes using VOC2007DetectionTiny.CLASS_NAMES
#     num_classes = len(VOC2007DetectionTiny.CLASS_NAMES)
#     model = FCOSModel(num_classes=num_classes).to(args.device)

#     # Initialize Adam optimizer and StepLR scheduler
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

#     # Train the model using the updated training code
#     test_ap, test_map = trainer.train(args, model, optimizer, scheduler)
#     print('Test mAP:', test_map)

#     # Save the final trained model
#     FINAL_MODEL_PATH = './trained_model_fcos.pth'
#     torch.save(model.state_dict(), FINAL_MODEL_PATH)
#     print(f"Final model saved at {FINAL_MODEL_PATH}")