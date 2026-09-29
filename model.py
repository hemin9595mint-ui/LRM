"""A/T/N decomposition -> inverse initialization -> progressive Mamba refinement."""
import torch
from torch import nn
from torch.nn import functional as F
from .physics import inverse, formation
from .mamba import AxisMambaBlock


def conv(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Estimator(nn.Module):
    """Compact U-Net; widths/depths are disclosed engineering choices."""
    def __init__(self, cin, width):
        super().__init__()
        self.enc = nn.Sequential(conv(cin, width), conv(width, width))
        self.down = nn.Sequential(conv(width, width * 2, 2), conv(width * 2, width * 2))
        self.dec = nn.Sequential(conv(width * 3, width), nn.Conv2d(width, 3, 1))

    def forward(self, x):
        skip = self.enc(x)
        deep = F.interpolate(self.down(skip), size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.dec(torch.cat([skip, deep], dim=1))


class Decomposition(nn.Module):
    def __init__(self, width=32, noise_scale=.1):
        super().__init__()
        self.illumination = Estimator(3, width)
        self.transmission = Estimator(6, width)
        self.noise_features = nn.Sequential(conv(9, width), conv(width, width))
        self.noise_head = nn.Conv2d(width, 3, 1)
        self.noise_scale = noise_scale

    def forward(self, x):
        a = self.illumination(x).sigmoid()
        t = self.transmission(torch.cat([x, a], 1)).sigmoid()
        # Dense signed residual with L1 sparsity; not a claimed sparse-convolution kernel.
        n = self.noise_scale * self.noise_head(self.noise_features(torch.cat([x, a, t], 1))).tanh()
        return a, t, n


class AquaPhysicsMamba(nn.Module):
    def __init__(self, width=32, blocks=4, state=16, backend='reference',
                 bidirectional=True, noise_scale=.1, eps=.001,
                 use_inverse=True, use_mamba=True):
        super().__init__()
        self.eps, self.use_inverse = eps, use_inverse
        self.decomposition = Decomposition(width, noise_scale)
        self.embed = nn.Sequential(conv(6, width, 2), conv(width, width, 2))
        self.blocks = nn.ModuleList([
            AxisMambaBlock(width, state, backend, bidirectional) for _ in range(blocks)
        ]) if use_mamba else nn.ModuleList()
        self.reconstruct = nn.Sequential(conv(width, width), nn.Conv2d(width, 3, 3, padding=1))

    def forward(self, x):
        if x.ndim != 5 or x.shape[2] != 3:
            raise ValueError('Expected RGB B,T,3,H,W tensor')
        b, time, _, h, w = x.shape
        flat = x.reshape(-1, 3, h, w)
        a, t, n = self.decomposition(flat)
        coarse = inverse(flat, a, t, n, self.eps)
        init = coarse if self.use_inverse else flat
        # Raw Eq. (9) is returned; clamp only guards neural input/output conversion.
        embed = self.embed(torch.cat([flat, init.clamp(0, 1)], dim=1))
        z = embed.reshape(b, time, *embed.shape[1:])
        for block in self.blocks:
            z = block(z)
        z = z.reshape(b * time, *z.shape[2:])
        residual = F.interpolate(self.reconstruct(z), (h, w), mode='bilinear', align_corners=False)
        base = init.clamp(.001, .999)
        j = torch.sigmoid(torch.logit(base) + residual)
        pack = {'J': j, 'A': a, 'T': t, 'N': n, 'J0': coarse,
                'I_hat': formation(j, a, t, n)}
        return {k: v.reshape(b, time, 3, h, w) for k, v in pack.items()}


class PatchDiscriminator(nn.Module):
    """Unbounded PatchGAN predictions for LSGAN; no final sigmoid."""
    def __init__(self, width=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, width, 4, 2, 1), nn.LeakyReLU(.2, inplace=True),
            nn.Conv2d(width, width * 2, 4, 2, 1), nn.InstanceNorm2d(width * 2), nn.LeakyReLU(.2, inplace=True),
            nn.Conv2d(width * 2, width * 4, 4, 2, 1), nn.InstanceNorm2d(width * 4), nn.LeakyReLU(.2, inplace=True),
            nn.Conv2d(width * 4, width * 4, 3, 1, 1), nn.LeakyReLU(.2, inplace=True),
            nn.Conv2d(width * 4, 1, 3, 1, 1))

    def forward(self, x):
        return self.net(x.reshape(-1, *x.shape[-3:]))
