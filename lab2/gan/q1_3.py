import argparse
import os
from utils import get_args

import torch
import torch.nn.functional as F

from networks import Discriminator, Generator
from train import train_model


def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp, interp, lamb
):
    ##################################################################
    # TODO 1.5: Implement WGAN-GP loss for discriminator.
    # Compute the gradient penalty using discrim_interp and interp.
    ##################################################################
    # Compute the basic WGAN loss
    loss = torch.mean(discrim_fake) - torch.mean(discrim_real)

    # Compute gradients of discriminator outputs w.r.t. interpolated inputs
    gradients = torch.autograd.grad(
        outputs=discrim_interp,
        inputs=interp,
        grad_outputs=torch.ones_like(discrim_interp),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # Reshape gradients to (batch_size, -1) and compute L2 norm per sample
    gradients = gradients.view(gradients.size(0), -1)
    grad_norms = gradients.norm(2, dim=1)

    # Compute gradient penalty
    gradient_penalty = ((grad_norms - 1) ** 2).mean()

    # Add gradient penalty to the loss
    loss += lamb * gradient_penalty
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    return loss


def compute_generator_loss(discrim_fake):
    ##################################################################
    # TODO 1.3: Implement GAN loss for the generator.
    ##################################################################
    # For WGAN, the generator loss is:
    loss = -torch.mean(discrim_fake)
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    return loss


if __name__ == "__main__":
    args = get_args()
    gen = Generator().cuda()
    disc = Discriminator().cuda()
    prefix = "data_gan/"
    os.makedirs(prefix, exist_ok=True)

    train_model(
        gen,
        disc,
        num_iterations=int(3e4),
        batch_size=256,
        prefix=prefix,
        gen_loss_fn=compute_generator_loss,
        disc_loss_fn=compute_discriminator_loss,
        log_period=1000,
        amp_enabled=not args.disable_amp,
    )
