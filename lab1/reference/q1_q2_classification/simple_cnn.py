import torch
import torch.nn as nn
import torch.nn.functional as F


def get_fc(inp_dim, out_dim, non_linear='relu'):
    """
    Mid-level API. It is useful to customize your own for large code repo.
    :param inp_dim: int, intput dimension
    :param out_dim: int, output dimension
    :param non_linear: str, 'relu', 'softmax'
    :return: list of layers [FC(inp_dim, out_dim), (non linear layer)]
    """
    layers = []
    layers.append(nn.Linear(inp_dim, out_dim))
    if non_linear == 'relu':
        layers.append(nn.ReLU())
    elif non_linear == 'softmax':
        layers.append(nn.Softmax(dim=1))
    elif non_linear == 'none':
        pass
    else:
        raise NotImplementedError
    return layers


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, inp_size=28, c_dim=1):
        super().__init__()
        self.num_classes = num_classes
        self.inp_size = inp_size  # Store input size
        self.conv1 = nn.Conv2d(c_dim, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.nonlinear = nn.ReLU()
        self.pool1 = nn.AvgPool2d(2, 2)
        self.pool2 = nn.AvgPool2d(2, 2)

        # Calculate the correct flat_dim based on the input size
        self.flat_dim = self._get_flat_dim(c_dim, inp_size)

        self.fc1 = nn.Sequential(*get_fc(self.flat_dim, 128, 'none'))
        self.fc2 = nn.Sequential(*get_fc(128, num_classes, 'none'))

    def _get_flat_dim(self, c_dim, inp_size):
        """
        Calculate the flat_dim based on the input size and channels.
        """
        dummy_input = torch.zeros(1, c_dim, inp_size, inp_size)  # (batch size, channels, height, width)

        x = self.conv1(dummy_input)
        x = self.nonlinear(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.nonlinear(x)
        x = self.pool2(x)

        flat_dim = x.numel()  # Calculate the total number of elements
        print(f"Calculated flat_dim: {flat_dim}")  # Debugging output
        return flat_dim

    def forward(self, x):
        """
        :param x: input image in shape of (N, C, H, W)
        :return: out: classification logits in shape of (N, Nc)
        """

        N = x.size(0)
        x = self.conv1(x)
        x = self.nonlinear(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.nonlinear(x)
        x = self.pool2(x)

        flat_x = x.view(N, self.flat_dim)
        out = self.fc1(flat_x)
        out = self.fc2(out)
        return out