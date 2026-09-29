"""Full-reference image metrics and Eq. (28) with externally supplied official flow."""
import numpy as np
import torch
from torch.nn import functional as F
from skimage.metrics import structural_similarity


def paired_metrics(pred, ref):
    if pred.shape != ref.shape:
        raise ValueError('Paired metric requires identical aligned RGB image shapes')
    if min(pred.shape[:2]) < 11:
        raise ValueError('SSIM requires image dimensions >= 11')
    p, r = pred.astype(np.float64) / 255, ref.astype(np.float64) / 255
    mse = np.mean((p - r) ** 2)
    return {'PSNR': float(-10 * np.log10(mse)) if mse > 0 else None,
            'SSIM': float(structural_similarity(p, r, channel_axis=2, data_range=1.0,
                          gaussian_weights=True, sigma=1.5, use_sample_covariance=False)),
            'identical': bool(mse == 0)}


def warping_error(previous, current, flow, mask):
    """Flow H,W,2: previous pixel -> current pixel (dx,dy). Mask H,W in [0,1].
    RGB squared error summed, divided by valid pixel weights (paper Eq. 28).
    Invalid sampling boundaries are removed. No alternate flow estimator is used.
    """
    if previous.shape != current.shape:
        raise ValueError('Adjacent enhanced images differ in shape')
    h, w, c = previous.shape
    if flow.shape != (h, w, 2) or mask.shape != (h, w) or c != 3:
        raise ValueError('Flow/mask/image dimensions do not match')
    if not np.isfinite(flow).all() or not np.isfinite(mask).all() or np.any((mask < 0) | (mask > 1)):
        raise ValueError('Flow must be finite and mask must be finite in [0,1]')
    y, x = np.mgrid[:h, :w]
    sx, sy = x + flow[..., 0], y + flow[..., 1]
    valid = (sx >= 0) & (sx <= w - 1) & (sy >= 0) & (sy <= h - 1)
    m = torch.from_numpy((mask * valid).astype(np.float32))
    if m.sum() == 0:
        raise ValueError('No valid nonoccluded pixels')
    grid = np.stack([2 * sx / max(w - 1, 1) - 1, 2 * sy / max(h - 1, 1) - 1], -1)
    j = torch.from_numpy(current.transpose(2, 0, 1).copy()).float()[None] / 255
    prior = torch.from_numpy(previous.transpose(2, 0, 1).copy()).float() / 255
    warped = F.grid_sample(j, torch.from_numpy(grid).float()[None], align_corners=True,
                           mode='bilinear', padding_mode='zeros')[0]
    return float((((warped - prior) * m).square().sum() / m.sum()).item())
