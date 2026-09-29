from collections import deque
import numpy as np
import torch
from torch.nn import functional as F
from .model import AquaPhysicsMamba


def load_model(weights, device):
    checkpoint = torch.load(weights, map_location='cpu', weights_only=False)
    model = AquaPhysicsMamba(**checkpoint['config']['model']).to(device).eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    return model, checkpoint['config']


def centered_windows(iterable, length):
    """Bounded-memory centered windows, replicating first/last at boundaries."""
    if length < 1 or length % 2 == 0:
        raise ValueError('Window length must be a positive odd integer')
    it = iter(iterable)
    try:
        first = next(it)
    except StopIteration:
        return
    r = length // 2
    q = deque([first] * (r + 1), maxlen=length)
    real = 1
    exhausted = False
    for _ in range(r):
        try:
            q.append(next(it))
            real += 1
        except StopIteration:
            exhausted = True
            q.append(q[-1])
    produced = 0
    while True:
        yield list(q)
        produced += 1
        if exhausted and produced >= real:
            break
        try:
            if exhausted:
                raise StopIteration
            q.append(next(it))
            real += 1
        except StopIteration:
            exhausted = True
            if produced >= real:
                break
            q.append(q[-1])


@torch.inference_mode()
def enhance_window(model, frames, device, max_side=0, return_maps=False):
    if len({frame.shape for frame in frames}) != 1:
        raise ValueError('Sequence dimensions change')
    x = torch.from_numpy(np.stack(frames).transpose(0, 3, 1, 2).copy()).float()[None].to(device) / 255.
    h, w = x.shape[-2:]
    if max_side and max(h, w) > max_side:
        size = (max(16, round(h * max_side / max(h, w))), max(16, round(w * max_side / max(h, w))))
        x = F.interpolate(x.flatten(0, 1), size, mode='bilinear', align_corners=False)[None]
    nh, nw = x.shape[-2:]
    ph, pw = max(0, 16 - nh), max(0, 16 - nw)
    if ph or pw:
        x = F.pad(x.flatten(0, 1), (0, pw, 0, ph), mode='replicate')[None]
    out = model(x)
    center = len(frames) // 2
    j = out['J'][0, center, :, :nh, :nw][None]
    j = F.interpolate(j, (h, w), mode='bilinear', align_corners=False)[0]
    image = (j.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    maps = {key: value[0, center, :, :nh, :nw].cpu().numpy() for key, value in out.items()} if return_maps else {}
    return image, maps
