import os
import torch
import numpy as np
import sklearn.metrics
from torch.utils.data import DataLoader
import torch.nn.functional as F


class ARGS(object):
    """
    Tracks hyper-parameters for trainer code 
        - Feel free to add your own hparams below (cannot have __ in name)
        - Constructor will automatically support overrding for non-default values
    
    Example::
        >>> args = ARGS(batch_size=23, use_cuda=True)
        >>> print(args)
        args.batch_size = 23
        args.device = cuda
        args.epochs = 14
        args.gamma = 0.7
        args.log_every = 100
        args.lr = 1.0
        args.save_model = False
        args.test_batch_size = 1000
        args.val_every = 100
    """
    # input batch size for training 
    batch_size = 64
    # input batch size for testing
    test_batch_size=1000
    # number of epochs to train for
    epochs = 14
    # learning rate
    lr = 1.0
    # Learning rate step gamma
    gamma = 0.7
    step_size = 1
    # how many batches to wait before logging training status
    log_every = 100
    # how many batches to wait before evaluating model
    val_every = 100
    # set flag to True if you wish to save the model after training
    save_at_end = False
    # set this to value >0 if you wish to save every x epochs
    save_freq=-1
    # set true if using GPU during training
    use_cuda = False
    # input size
    inp_size = 224

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            assert '__' not in k and hasattr(self, k), "invalid attribute!"
            assert k != 'device', "device property cannot be modified"
            setattr(self, k, v)
        
    def __repr__(self):
        repr_str = ''
        for attr in dir(self):
            if '__' not in attr and attr !='use_cuda':
                repr_str += 'args.{} = {}\n'.format(attr, getattr(self, attr))
        return repr_str
    
    @property
    def device(self):
        return torch.device("cuda" if self.use_cuda else "cpu")


def compute_ap(gt, pred, valid, average=None):
    # ! Do not modify the code in this function
    
    """
    Compute the multi-label classification accuracy.
    Args:
        gt (np.ndarray): Shape Nx20, 0 or 1, 1 if the object i is present in that
            image.
        pred (np.ndarray): Shape Nx20, probability of that object in the image
            (output probablitiy).
        valid (np.ndarray): Shape Nx20, 0 if you want to ignore that class for that
            image. Some objects are labeled as ambiguous.
    Returns:
        AP (list): average precision for all classes
    """
    nclasses = gt.shape[1]
    AP = []
    for cid in range(nclasses):
        gt_cls = gt[:, cid][valid[:, cid] > 0].astype('float32')
        pred_cls = pred[:, cid][valid[:, cid] > 0].astype('float32')
        # As per PhilK. code:
        # https://github.com/philkr/voc-classification/blob/master/src/train_cls.py
        pred_cls -= 1e-5 * gt_cls
        ap = sklearn.metrics.average_precision_score(
            gt_cls, pred_cls, average=average)
        AP.append(ap)
    return AP


def eval_dataset_map(model, device, test_loader):
    # ! Do not modify the code in this function

    """
    Evaluate the model with the given dataset
    Args:
         model (keras.Model): model to be evaluated
         dataset (tf.data.Dataset): evaluation dataset
    Returns:
         AP (list): Average Precision for all classes
         MAP (float): mean average precision
    """
    
    with torch.no_grad():
        AP_list = []

        for data, target, wgt in test_loader:
            data = data.to(device)
            output = model(data)
            
            pred = F.softmax(output, dim=-1).cpu().numpy()
            gt = target.cpu().numpy()
            valid = wgt.cpu().numpy()
            
            AP = compute_ap(gt, pred, valid)
            AP_list.extend(AP)
    
    AP = np.array(AP_list)
    mAP = np.mean(AP)

    return AP, mAP


def get_data_loader(name='voc', train=True, batch_size=64, split='train', inp_size=224):
    if name == 'voc':
        from voc_dataset import VOCDataset
        dataset = VOCDataset(split, inp_size)
    else:
        raise NotImplementedError

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=4,
    )
    return loader


################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################


# import os
# import torch
# from torch.utils.data import DataLoader
# from detection.detection_helper import VOC2007DetectionTiny  # Updated import
# from typing import Any

# class ARGS:
#     def __init__(
#         self,
#         epochs: int,
#         inp_size: int,
#         use_cuda: bool,
#         device: torch.device,
#         val_every: int,
#         lr: float,
#         batch_size: int,
#         test_batch_size: int,
#         step_size: int,
#         gamma: float,
#         save_freq: int,
#         save_at_end: bool,
#         log_every: int
#     ):
#         self.epochs = epochs
#         self.inp_size = inp_size
#         self.use_cuda = use_cuda
#         self.device = device
#         self.val_every = val_every
#         self.lr = lr
#         self.batch_size = batch_size
#         self.test_batch_size = test_batch_size
#         self.step_size = step_size
#         self.gamma = gamma
#         self.save_freq = save_freq
#         self.save_at_end = save_at_end
#         self.log_every = log_every

# def custom_collate_fn(batch):
#     """
#     Custom collate function to handle batching of images and ground truth boxes.

#     Args:
#         batch (list): List of tuples (image, gt_boxes)

#     Returns:
#         images (torch.Tensor): Batched images tensor.
#         gt_boxes (torch.Tensor): Batched ground truth boxes tensor.
#     """
#     images, gt_boxes = zip(*batch)
#     images = torch.stack(images, dim=0)  # Shape: (B, C, H, W)
#     gt_boxes = torch.stack(gt_boxes, dim=0)  # Shape: (B, 40, 5)
#     return images, gt_boxes

# def get_data_loader(dataset_dir: str, train: bool = True, batch_size: int = 32, split: str = 'trainval', inp_size: int = 224) -> DataLoader:
#     """ 
#     Args:
#         dataset_dir (str): Directory of the dataset.
#         train (bool): Whether to shuffle the data.
#         batch_size (int): Number of samples per batch.
#         split (str): Dataset split - 'train', 'val', 'trainval', or 'test'.
#         inp_size (int): Input image size.
#     Returns:
#         DataLoader object
#     """
#     download = True if split in ["train", "trainval"] and train else False
#     dataset = VOC2007DetectionTiny(
#         dataset_dir=dataset_dir, split=split, download=download, image_size=inp_size
#     )
#     return DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=train,
#         collate_fn=custom_collate_fn,
#         num_workers=4  # Adjust based on your system
#     )

# def eval_dataset_map(model: torch.nn.Module, device: torch.device, data_loader: DataLoader) -> (float, float):
#     """
#     Evaluate model on the dataset and compute mAP.

#     Args:
#         model (nn.Module): Trained model.
#         device (torch.device): Device to perform computation on.
#         data_loader (DataLoader): DataLoader for the evaluation dataset.

#     Returns:
#         ap (float): Average Precision for the current AP.
#         map (float): Mean Average Precision across all classes.
#     """
#     # Placeholder implementation:
#     # Implement actual evaluation logic here.
#     ap = 0.0
#     map = 0.0
#     return ap, map