import torch
import torch.nn as nn
import torch.nn.functional as F


class UpSampleConv2D(torch.jit.ScriptModule):
    def __init__(
        self,
        input_channels: int,
        kernel_size: int = 3,
        n_filters: int = 128,
        upscale_factor: int = 2,
        padding: int = 0,
    ):
        super(UpSampleConv2D, self).__init__()
        self.conv = nn.Conv2d(
            input_channels, n_filters, kernel_size=kernel_size, padding=padding
        )
        self.upscale_factor: int = upscale_factor  # Explicitly type as int

    @torch.jit.script_method
    def forward(self, x):
        ##################################################################
        # Implement nearest neighbor upsampling
        # 1. Repeat x channel-wise upscale_factor^2 times
        # 2. Use torch.nn.PixelShuffle to form an output of dimension
        #    (batch, channel, height*upscale_factor, width*upscale_factor)
        # 3. Apply convolution and return output
        ##################################################################
        # Step 1: Repeat channels
        upscale_factor_squared = self.upscale_factor * self.upscale_factor
        x = x.repeat(1, upscale_factor_squared, 1, 1)
        # Step 2: PixelShuffle rearranges channels to spatial dimensions
        x = F.pixel_shuffle(x, self.upscale_factor)
        # Step 3: Apply convolution
        x = self.conv(x)
        return x
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################
        

class DownSampleConv2D(torch.jit.ScriptModule):
    def __init__(
        self, input_channels: int, kernel_size: int = 3, n_filters: int = 128, downscale_ratio: int = 2, padding: int = 0
    ):
        super(DownSampleConv2D, self).__init__()
        self.conv = nn.Conv2d(
            input_channels, n_filters, kernel_size=kernel_size, padding=padding
        )
        self.downscale_ratio: int = downscale_ratio  # Explicitly type as int

    @torch.jit.script_method
    def forward(self, x):
        ##################################################################
        # Implement spatial mean pooling
        # 1. Use torch.nn.PixelUnshuffle to form an output of dimension
        #    (batch, channel*downscale_factor^2, height/downscale_factor, width/downscale_factor)
        # 2. Then split channel-wise and reshape into
        #    (downscale_factor^2, batch, channel, height, width) images
        # 3. Take the average across dimension 0, apply convolution,
        #    and return the output
        ##################################################################
        # Step 1: Apply PixelUnshuffle
        x = F.pixel_unshuffle(x, self.downscale_ratio)
        # Step 2: Reshape
        batch_size, channels, height, width = x.shape
        factor_squared = self.downscale_ratio * self.downscale_ratio
        channels_per_split = channels // factor_squared
        x = x.view(batch_size, factor_squared, channels_per_split, height, width)
        # Step 3: Take average and apply convolution
        x = x.mean(dim=1)
        x = self.conv(x)
        return x
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################



class ResBlockUp(torch.jit.ScriptModule):
    """
    ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(in_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(n_filters, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
            (5): UpSampleConv2D(
                (conv): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (upsample_residual): UpSampleConv2D(
            (conv): Conv2d(input_channels, n_filters, kernel_size=(1, 1), stride=(1, 1))
        )
    )
    """

    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlockUp, self).__init__()
        ##################################################################
        # Setup the network layers
        ##################################################################
        self.layers = nn.Sequential(
            nn.BatchNorm2d(input_channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(
                input_channels, n_filters, kernel_size=kernel_size, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(n_filters, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            UpSampleConv2D(
                input_channels=n_filters,
                kernel_size=kernel_size,
                n_filters=n_filters,
                padding=1,
            ),
        )
        self.upsample_residual = UpSampleConv2D(
            input_channels=input_channels,
            kernel_size=1,
            n_filters=n_filters,
            padding=0,
        )
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    @torch.jit.script_method
    def forward(self, x):
        ##################################################################
        # Forward through the layers and implement a residual connection.
        # Make sure to upsample the residual before adding it to the layer output.
        ##################################################################
        residual = self.upsample_residual(x)
        out = self.layers(x)
        out += residual
        return out
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################


class ResBlockDown(torch.jit.ScriptModule):
    """
    ResBlockDown(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(in_channels, n_filters, kernel_size=3, padding=1)
            (2): ReLU()
            (3): DownSampleConv2D(...)
        )
        (downsample_residual): DownSampleConv2D(...)
    )
    """

    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlockDown, self).__init__()
        ##################################################################
        # Setup the network layers
        ##################################################################
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                input_channels, n_filters, kernel_size=kernel_size, padding=1
            ),
            nn.ReLU(),
            DownSampleConv2D(
                input_channels=n_filters,
                kernel_size=kernel_size,
                n_filters=n_filters,
                padding=1,
            ),
        )
        self.downsample_residual = DownSampleConv2D(
            input_channels=input_channels,
            kernel_size=1,
            n_filters=n_filters,
            padding=0,
        )
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    @torch.jit.script_method
    def forward(self, x):
        ##################################################################
        # Forward through the layers and implement a residual connection.
        # Make sure to downsample the residual before adding it to the layer output.
        ##################################################################
        residual = self.downsample_residual(x)
        out = self.layers(x)
        out += residual
        return out
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################


class ResBlock(torch.jit.ScriptModule):
    """
    ResBlock(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(in_channels, n_filters, kernel_size=3, padding=1)
            (2): ReLU()
            (3): Conv2d(n_filters, n_filters, kernel_size=3, padding=1)
        )
    )
    """

    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlock, self).__init__()
        ##################################################################
        # Setup the network layers
        ##################################################################
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                input_channels, n_filters, kernel_size=kernel_size, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                n_filters, n_filters, kernel_size=kernel_size, padding=1
            ),
        )
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    @torch.jit.script_method
    def forward(self, x):
        ##################################################################
        # Forward the conv layers. Don't forget the residual connection!
        ##################################################################
        residual = x
        out = self.layers(x)
        out += residual
        return out
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################


class Generator(torch.jit.ScriptModule):
    """
    Generator(
        (dense): Linear(128, 2048)
        (layers): Sequential(
            (0): ResBlockUp(...)
            (1): ResBlockUp(...)
            (2): ResBlockUp(...)
            (3): BatchNorm2d(128)
            (4): ReLU()
            (5): Conv2d(128, 3, kernel_size=3, padding=1)
            (6): Tanh()
        )
    )
    """

    def __init__(self, starting_image_size=4):
        super(Generator, self).__init__()
        ##################################################################
        # Set up the network layers
        ##################################################################
        self.dense = nn.Linear(
            128, 128 * starting_image_size * starting_image_size
        )
        self.layers = nn.Sequential(
            ResBlockUp(128),
            ResBlockUp(128),
            ResBlockUp(128),
            nn.BatchNorm2d(128, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    @torch.jit.script_method
    def forward_given_samples(self, z):
        ##################################################################
        # Forward the generator assuming a set of samples z has
        # been passed in. Don't forget to re-shape the output of the dense
        # layer into an image with the appropriate size!
        ##################################################################
        x = self.dense(z)
        x = x.view(-1, 128, 4, 4)  # Assuming starting_image_size=4
        x = self.layers(x)
        return x
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    # @torch.jit.script_method
    # def forward(self, n_samples: int = 1024):
    #     ##################################################################
    #     # Generate n_samples latents and forward through the network.
    #     ##################################################################
    #     # Get the device of the model parameters
    #     device = next(self.parameters()).device
    #     # Create z on the same device
    #     z = torch.randn(n_samples, 128, device=device)
    #     x = self.forward_given_samples(z)
    #     return x

    @torch.jit.script_method
    def forward(self, n_samples: int = 1024, device: torch.device = torch.device('cuda')):
        z = torch.randn(n_samples, 128, device=device)
        x = self.forward_given_samples(z)
        return x        
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################


class Discriminator(torch.jit.ScriptModule):
    """
    Discriminator(
        (layers): Sequential(
            (0): ResBlockDown(...)
            (1): ResBlockDown(...)
            (2): ResBlock(...)
            (3): ResBlock(...)
            (4): ReLU()
        )
        (dense): Linear(128, 1)
    )
    """

    def __init__(self):
        super(Discriminator, self).__init__()
        ##################################################################
        # Set up the network layers
        ##################################################################
        self.layers = nn.Sequential(
            ResBlockDown(3, n_filters=128),
            ResBlockDown(128, n_filters=128),
            ResBlock(128, n_filters=128),
            ResBlock(128, n_filters=128),
            nn.ReLU(),
        )
        self.dense = nn.Linear(128, 1)
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    @torch.jit.script_method
    def forward(self, x):
        ##################################################################
        # Forward the discriminator assuming a batch of images
        # have been passed in. Make sure to sum across the image
        # dimensions after passing x through self.layers.
        ##################################################################
        x = self.layers(x)
        x = x.sum(dim=[2, 3])  # Sum over spatial dimensions
        x = self.dense(x)
        return x
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################
