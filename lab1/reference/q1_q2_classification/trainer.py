from __future__ import print_function

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import utils
from voc_dataset import VOCDataset


def save_this_epoch(args, epoch):
    if args.save_freq > 0 and (epoch+1) % args.save_freq == 0:
        return True
    if args.save_at_end and (epoch+1) == args.epochs:
        return True
    return False


def save_model(epoch, model_name, model, save_path='trained_model.pth'):
    filename = 'checkpoint-{}-epoch{}.pth'.format(
        model_name, epoch+1)
    print("Saving model checkpoint at ", filename)
    torch.save(model.state_dict(), filename)  # Save state_dict

    # Additionally, save the final model state_dict
    if (epoch + 1) == args.epochs:
        torch.save(model.state_dict(), save_path)
        print(f"Final model saved at {save_path}")


def train(args, model, optimizer, scheduler=None, model_name='model'):
    writer = SummaryWriter()
    train_loader = utils.get_data_loader(
        'voc', train=True, batch_size=args.batch_size, split='trainval', inp_size=args.inp_size)
    test_loader = utils.get_data_loader(
        'voc', train=False, batch_size=args.test_batch_size, split='test', inp_size=args.inp_size)

    # Ensure model is in correct mode and on right device
    model.train()
    model = model.to(args.device)

    cnt = 0

    for epoch in range(args.epochs):
        for batch_idx, (data, target, wgt) in enumerate(train_loader):
            data, target, wgt = data.to(args.device), target.to(args.device), wgt.to(args.device)

            optimizer.zero_grad()
            output = model(data)

            ##################################################################
            # TODO: Implement a suitable loss function for multi-label
            # classification. You are NOT allowed to use any pytorch built-in
            # functions. Remember to take care of underflows / overflows.
            # Function Inputs:
            #   - `output`: Outputs from the network (logits)
            #   - `target`: Ground truth labels, refer to voc_dataset.py
            #   - `wgt`: Weights (difficult or not), refer to voc_dataset.py
            # Function Outputs:
            #   - `output`: Computed loss, a single floating point number
            ##################################################################
            max_val = torch.clamp(output, min=0)
            loss = max_val - output * target + torch.log(1 + torch.exp(-torch.abs(output)))
            loss = loss * wgt
            loss = loss.mean()

            #### Add L2 regularization
            # l2_lambda = 0.01  # You can adjust this value
            # l2_loss = sum(param.pow(2).sum() for param in model.parameters())
            # loss += l2_lambda * l2_loss            
            ##################################################################
            #                          END OF YOUR CODE                      #
            ##################################################################

            loss.backward()

            if cnt % args.log_every == 0:
                writer.add_scalar("Loss/train", loss.item(), cnt)
                print('Train Epoch: {} [{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, cnt, 100. * batch_idx / len(train_loader), loss.item()))
                
                # Log gradients
                for tag, value in model.named_parameters():
                    if value.grad is not None:
                        grad_numpy = value.grad.cpu().numpy()
                        
                        # Check for NaN or Inf
                        if np.isnan(grad_numpy).any() or np.isinf(grad_numpy).any():
                            print(f"Invalid gradients detected in {tag}. Skipping logging.")
                            continue  # Skip logging for this parameter
                        
                        # Ensure gradients are of type float32
                        grad_numpy = grad_numpy.astype(np.float32)
                        # for tag, value in model.named_parameters():
                        #     if value.grad is not None:
                        #         writer.add_histogram(tag + "/grad", value.grad.cpu().numpy(), cnt)
                        
                        hist, bin_edges = np.histogram(grad_numpy, bins=5)
                        # Log histogram data as scalars
                        for i in range(len(hist)):
                            writer.add_scalar(f"{tag}/grad_bin_{i}", hist[i], cnt)                        
            optimizer.step()
            
            # Validation iteration
            if cnt % args.val_every == 0:
                model.eval()
                ap, map = utils.eval_dataset_map(model, args.device, test_loader)
                print("map: ", map)
                writer.add_scalar("map", map, cnt)
                model.train()
            
            cnt += 1

        if scheduler is not None:
            scheduler.step()
            writer.add_scalar("learning_rate", scheduler.get_last_lr()[0], cnt)

        # Save model
        if save_this_epoch(args, epoch):
            save_model(epoch, model_name, model)

    # Final Validation
    test_loader = utils.get_data_loader('voc', train=False, batch_size=args.test_batch_size, split='test', inp_size=args.inp_size)
    ap, map = utils.eval_dataset_map(model, args.device, test_loader)
    return ap, map

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################

# from typing import Any
# import torch
# from torch.utils.tensorboard import SummaryWriter
# import utils  # Assuming utils.py is in the same directory or adjust the import path
# import json
# import os
# from PIL import Image
# import torch
# from torch.utils.data import Dataset
# from torchvision import transforms

# def save_model(epoch: int, model_name: str, model: torch.nn.Module, args: Any):
#     """
#     Save the model state dictionary.

#     Args:
#         epoch (int): Current epoch number.
#         model_name (str): Name of the model.
#         model (nn.Module): The model to save.
#         args (Any): Training arguments.
#     """
#     save_path = f"./models/{model_name}_epoch_{epoch}.pth"
#     torch.save(model.state_dict(), save_path)
#     print(f"Model saved to {save_path}")

# def save_this_epoch(args: Any, epoch: int) -> bool:
#     """
#     Determine whether to save the model at the current epoch.

#     Args:
#         args (Any): Training arguments.
#         epoch (int): Current epoch number.

#     Returns:
#         bool: Whether to save the model.
#     """
#     if args.save_at_end and epoch == args.epochs - 1:
#         return True
#     if epoch % args.save_freq == 0:
#         return True
#     return False

# def train(args: Any, model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler):
#     """
#     Training loop.

#     Args:
#         args (Any): Training arguments.
#         model (nn.Module): The model to train.
#         optimizer (Optimizer): Optimizer.
#         scheduler (LR_Scheduler): Learning rate scheduler.

#     Returns:
#         ap (float): Average Precision on test dataset.
#         map (float): Mean Average Precision on test dataset.
#     """
#     writer = SummaryWriter()  # Initialize TensorBoard writer
#     train_loader = utils.get_data_loader(
#        dataset_dir='/home/fluentt/data/VOC2007',
#        train=True,
#        batch_size=args.batch_size,
#        split='train',
#        inp_size=args.inp_size
#    )

#     test = utils.get_data_loader(
#        dataset_dir='/home/fluentt/data/VOC2007',
#        train=False,
#        batch_size=args.batch_size,
#        split='val',
#        inp_size=args.inp_size
#    )

#     model.train()
#     cnt = 0

#     for epoch in range(args.epochs):
#         for batch_idx, (images, gt_boxes) in enumerate(train_loader):
#             images = images.to(args.device)
#             gt_boxes = gt_boxes.to(args.device)

#             optimizer.zero_grad()
#             outputs = model(images, gt_boxes)

#             # Assuming outputs is a dict with losses
#             loss_cls = outputs.get('loss_cls', torch.tensor(0.0, device=args.device))
#             loss_box = outputs.get('loss_box', torch.tensor(0.0, device=args.device))
#             loss_ctr = outputs.get('loss_ctr', torch.tensor(0.0, device=args.device))
            
#             loss = loss_cls + loss_box + loss_ctr
#             loss.backward()
#             optimizer.step()

#             # Logging
#             if cnt % args.log_every == 0:
#                 writer.add_scalar('Loss/total', loss.item(), cnt)
#                 writer.add_scalar('Loss/classification', loss_cls.item(), cnt)
#                 writer.add_scalar('Loss/box', loss_box.item(), cnt)
#                 writer.add_scalar('Loss/centerness', loss_ctr.item(), cnt)

#             # Validation
#             if cnt % args.val_every == 0 and cnt != 0:
#                 ap, map = utils.eval_dataset_map(model, args.device, test_loader)
#                 writer.add_scalar('Validation/mAP', map, cnt)
#                 print(f"Epoch [{epoch}/{args.epochs}], Step [{cnt}], mAP: {map}")

#             cnt += 1

#         scheduler.step()

#         if save_this_epoch(args, epoch):
#             save_model(epoch, "FCOS", model, args)

#     # Final Validation
#     ap, map = utils.eval_dataset_map(model, args.device, test_loader)
#     writer.add_scalar('Validation/mAP_final', map, cnt)
#     print(f"Final mAP: {map}")
#     writer.close()
#     return ap, map