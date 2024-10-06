import os

import torch
import torch.nn.functional as F
from utils import get_args

from networks import Discriminator, Generator
from train import train_model


def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp, interp, lamb
):
    ##################################################################
    # TODO: 1.4: Implement LSGAN loss for discriminator.
    # Do not use discrim_interp, interp, lamb. They are placeholders
    # for Q1.5.
    ##################################################################
    # LSGAN Discriminator Loss:
    # For real images: 0.5 * (D(x) - b)^2, where b = 1
    # For fake images: 0.5 * (D(G(z)) - a)^2, where a = 0
    # Total loss: sum of both losses

    # Labels for real and fake images
    real_label = 1.0
    fake_label = 0.0

    # Compute losses for real and fake images
    loss_real = 0.5 * torch.mean((discrim_real - real_label) ** 2)
    loss_fake = 0.5 * torch.mean((discrim_fake - fake_label) ** 2)

    # Total discriminator loss
    loss = loss_real + loss_fake
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    return loss


def compute_generator_loss(discrim_fake):
    ##################################################################
    # TODO: 1.4: Implement LSGAN loss for generator.
    ##################################################################
    # LSGAN Generator Loss:
    # For fake images: 0.5 * (D(G(z)) - c)^2, where c = 1
    # The generator wants the discriminator to output real labels for fake images

    # Label that the generator wants the discriminator to output
    real_label = 1.0

    # Compute generator loss
    loss = 0.5 * torch.mean((discrim_fake - real_label) ** 2)
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    return loss


if __name__ == "__main__":
    args = get_args()
    gen = Generator().cuda()
    disc = Discriminator().cuda()
    prefix = "data_ls_gan/"
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
