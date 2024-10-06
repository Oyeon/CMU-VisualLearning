import argparse
import torch
from cleanfid import fid
from matplotlib import pyplot as plt
import torchvision.utils


def save_plot(x, y, xlabel, ylabel, title, filename):
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename + ".png")


@torch.no_grad()
def get_fid(gen, dataset_name, dataset_resolution, z_dimension, batch_size, num_gen):
    gen_fn = lambda z: (gen.forward_given_samples(z) / 2 + 0.5) * 255
    score = fid.compute_fid(
        gen=gen_fn,
        dataset_name=dataset_name,
        dataset_res=dataset_resolution,
        num_gen=num_gen,
        z_dim=z_dimension,
        batch_size=batch_size,
        verbose=True,
        dataset_split="custom",
    )
    return score


@torch.no_grad()
def interpolate_latent_space(gen, path):
    ##################################################################
    # TODO: 1.2: Generate and save out latent space interpolations.
    # 1. Generate 100 samples of 128-dim vectors. Do so by linearly
    # interpolating for 10 steps across each of the first two
    # dimensions between -1 and 1. Keep the rest of the z vector for
    # the samples to be some fixed value (e.g. 0).
    # 2. Forward the samples through the generator.
    # 3. Save out an image holding all 100 samples.
    # Use torchvision.utils.save_image to save out the visualization.
    ##################################################################
    # Step 1: Generate 100 samples with interpolated z vectors
    device = next(gen.parameters()).device  # Get the device of the generator
    
    z = torch.zeros(100, 128, device=device)

    # Create interpolation values for the first two dimensions
    z1_values = torch.linspace(-1, 1, steps=10)
    z2_values = torch.linspace(-1, 1, steps=10)

    # Meshgrid to create combinations
    z1_grid, z2_grid = torch.meshgrid(z1_values, z2_values)
    z1_grid = z1_grid.flatten()
    z2_grid = z2_grid.flatten()

    # Set the first two dimensions of z
    z[:, 0] = z1_grid
    z[:, 1] = z2_grid
    # The rest of z is already zero

    # Step 2: Generate images
    samples = gen.forward_given_samples(z)
    # Rescale to [0, 1]
    samples = (samples + 1) / 2.0

    # Step 3: Save the images
    torchvision.utils.save_image(samples, path, nrow=10)
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--disable_amp", action="store_true")
    args = parser.parse_args()
    return args
