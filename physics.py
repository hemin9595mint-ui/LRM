"""Paper equations (1), (3), (9), (11); RGB inputs in [0, 1]."""
import torch
from torch import nn
from torch.nn import functional as F


class GaussianBlur(nn.Module):
    def __init__(self, sigma=3.0):
        super().__init__()
        if sigma <= 0:
            raise ValueError('sigma must be positive')
        radius = int(3 * sigma + 0.5)
        x = torch.arange(-radius, radius + 1).float()
        k = torch.exp(-x.square() / (2 * sigma * sigma))
        k /= k.sum()
        self.register_buffer('kernel', (k[:, None] * k[None, :])[None, None])
        self.radius = radius

    def forward(self, x):
        shape = x.shape
        z = x.reshape(-1, *shape[-3:])
        z = F.pad(z, [self.radius] * 4, mode='replicate')
        z = F.conv2d(z, self.kernel.to(z).expand(shape[-3], 1, -1, -1), groups=shape[-3])
        return z.reshape(shape)


def formation(j, a, t, n):
    return j * t + a * (1 - t) + n


def inverse(i, a, t, n, eps=1e-3):
    # Keep this unclipped to implement Eq. (9). Only the refinement input is bounded.
    return (i - a * (1 - t) - n) / (t + eps)


class RetinexPrior(nn.Module):
    def __init__(self, sigma=3., eps=1e-3):
        super().__init__()
        self.blur = GaussianBlur(sigma)
        self.eps = eps

    def forward(self, x):
        return self.blur(torch.log(x + self.eps)).exp()
