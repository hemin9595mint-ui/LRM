"""Equations (12)-(25); clean references are unpaired, never pixel targets."""
import torch
from torch import nn
from torch.nn import functional as F
from .physics import GaussianBlur, RetinexPrior


def rgb_to_lab(x):
    # Differentiable D65 CIELAB; channels normalized by 100,128,128.
    lin = torch.where(x > .04045, ((x + .055) / 1.055).clamp_min(1e-8).pow(2.4), x / 12.92)
    matrix = x.new_tensor([[.4124564, .3575761, .1804375],
                           [.2126729, .7151522, .0721750],
                           [.0193339, .1191920, .9503041]])
    xyz = torch.einsum('ij,bjhw->bihw', matrix, lin)
    xyz = xyz / x.new_tensor([.95047, 1., 1.08883])[None, :, None, None]
    d = 6 / 29
    f = torch.where(xyz > d ** 3, xyz.clamp_min(1e-8).pow(1 / 3), xyz / (3 * d * d) + 4 / 29)
    return torch.stack([(116 * f[:, 1] - 16) / 100,
                        500 * (f[:, 0] - f[:, 1]) / 128,
                        200 * (f[:, 1] - f[:, 2]) / 128], 1)


def luminance(x):
    return (x * x.new_tensor([.2126, .7152, .0722])[None, :, None, None]).sum(1, keepdim=True)


def structure_loss(j, i):
    j, i = luminance(j), luminance(i)
    return F.l1_loss(j[..., 1:, :] - j[..., :-1, :], i[..., 1:, :] - i[..., :-1, :]) + \
        F.l1_loss(j[..., :, 1:] - j[..., :, :-1], i[..., :, 1:] - i[..., :, :-1])


def gram(x):
    b, c, h, w = x.shape
    f = x.reshape(b, c, h * w)
    return f @ f.transpose(1, 2) / (c * h * w)


class VGGStyle(nn.Module):
    def __init__(self, local_weights=None):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights
        if local_weights:
            net = vgg16(weights=None)
            net.load_state_dict(torch.load(local_weights, map_location='cpu', weights_only=True))
        else:
            # Never substitute random weights if the download is unavailable.
            net = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.features = net.features[:16].eval().requires_grad_(False)
        self.register_buffer('mean', torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer('std', torch.tensor([.229, .224, .225])[None, :, None, None])

    def encode(self, x):
        x = (x - self.mean) / self.std
        results = []
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in (3, 8, 15):
                results.append(gram(x))
        return results

    def forward(self, j, clean):
        with torch.no_grad():
            ref = self.encode(clean)
        return sum(F.l1_loss(a, b) for a, b in zip(self.encode(j), ref))


class EnhancementLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.w = cfg
        self.prior = RetinexPrior(cfg['retinex_sigma'], cfg['eps'])
        self.blur = GaussianBlur(cfg['temporal_sigma'])
        self.style = VGGStyle(cfg.get('vgg_weights')) if cfg['sty'] > 0 and cfg['cdg'] > 0 else None

    def forward(self, output, x, clean, discriminator):
        j = output['J']
        jf, xf = j.flatten(0, 1), x.flatten(0, 1)
        # Independently sampled clean image per sequence, broadcast over its frames.
        ref = clean[:, None].expand(-1, x.shape[1], -1, -1, -1).flatten(0, 1)
        cj, cr = rgb_to_lab(jf), rgb_to_lab(ref)
        color = (cj.mean((-2, -1)) - cr.mean((-2, -1))).abs().mean()
        color = color + (cj.std((-2, -1), unbiased=False) - cr.std((-2, -1), unbiased=False)).abs().mean()
        vals = {'deg': F.l1_loss(output['I_hat'], x),
                'A': F.l1_loss(output['A'], self.prior(x)), 'N': output['N'].abs().mean(),
                'col': color, 'str': structure_loss(jf, xf),
                'sty': self.style(jf, ref) if self.style else jf.new_zeros(())}
        if x.shape[1] > 1:
            weight = torch.exp(-self.w['gamma'] * (self.blur(x[:, 1:]) - self.blur(x[:, :-1])).abs())
            e = j - x
            vals['temp'] = (weight * (e[:, 1:] - e[:, :-1]).abs()).mean()
        else:
            vals['temp'] = jf.new_zeros(())
        vals['adv'] = .5 * (discriminator(jf) - 1).square().mean() if self.w['adv'] else jf.new_zeros(())
        total = self.w['deg'] * vals['deg'] + self.w['A'] * vals['A'] + self.w['N'] * vals['N']
        total = total + self.w['cdg'] * sum(self.w[k] * vals[k] for k in ('col', 'sty', 'str'))
        total = total + self.w['temp'] * vals['temp'] + self.w['adv'] * vals['adv']
        return total, vals


def discriminator_loss(discriminator, generated, clean):
    return .5 * ((discriminator(clean) - 1).square().mean() + discriminator(generated.detach()).square().mean())
