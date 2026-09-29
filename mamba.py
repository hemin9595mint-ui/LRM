"""Portable input-selective SSM, plus optional upstream Mamba backend.

The recurrence is h_t=exp(delta_t*A)*h_(t-1)+delta_t*B_t*u_t;
y_t=C_t*h_t+D*u_t. Delta, B, C depend on the current token.
This is NOT a GRU/attention substitution. Axis factorization is an implementation
choice because the manuscript does not specify tokenization or scan directions.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class SelectiveMamba(nn.Module):
    def __init__(self, dim, state=16, expand=2, kernel=4):
        super().__init__()
        inner = dim * expand
        rank = max(1, math.ceil(dim / 16))
        self.state = state
        self.rank = rank
        self.inner = inner
        self.kernel = kernel
        self.in_proj = nn.Linear(dim, 2 * inner, bias=False)
        self.conv = nn.Conv1d(inner, inner, kernel, groups=inner, padding=kernel - 1)
        self.select = nn.Linear(inner, rank + 2 * state, bias=False)
        self.dt = nn.Linear(rank, inner)
        self.A_log = nn.Parameter(torch.arange(1, state + 1).float().log().repeat(inner, 1))
        self.D = nn.Parameter(torch.ones(inner))
        self.out_proj = nn.Linear(inner, dim, bias=False)
        nn.init.uniform_(self.dt.weight, -rank ** -0.5, rank ** -0.5)
        with torch.no_grad():
            delta = torch.exp(torch.empty(inner).uniform_(math.log(.001), math.log(.1)))
            self.dt.bias.copy_(delta + torch.log(-torch.expm1(-delta)))

    def forward(self, x):
        u, gate = self.in_proj(x).chunk(2, -1)
        u = F.silu(self.conv(u.transpose(1, 2))[..., :x.shape[1]].transpose(1, 2))
        dt, b, c = torch.split(self.select(u), [self.rank, self.state, self.state], dim=-1)
        dt = F.softplus(self.dt(dt)).float()
        a = -self.A_log.float().exp()
        u32, b, c = u.float(), b.float(), c.float()
        h = u32.new_zeros(x.shape[0], self.inner, self.state)
        ys = []
        # Explicit reference scan: portable, differentiable, slower than CUDA kernels.
        for k in range(x.shape[1]):
            decay = torch.exp(dt[:, k, :, None] * a)
            h = decay * h + dt[:, k, :, None] * b[:, k, None, :] * u32[:, k, :, None]
            ys.append((h * c[:, k, None, :]).sum(-1) + self.D.float() * u32[:, k])
        y = torch.stack(ys, dim=1).to(u.dtype) * F.silu(gate)
        return self.out_proj(y)


def mixer(dim, state, backend):
    if backend == 'reference':
        return SelectiveMamba(dim, state)
    if backend == 'official':
        try:
            from mamba_ssm import Mamba
        except ImportError as exc:
            raise RuntimeError('Install mamba-ssm in a compatible Linux/CUDA environment, '
                               'or select model.backend=reference.') from exc
        return Mamba(d_model=dim, d_state=state, d_conv=4, expand=2)
    raise ValueError(f'Unknown backend: {backend}')


class AxisMambaBlock(nn.Module):
    def __init__(self, dim, state=16, backend='reference', bidirectional=True):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(3)])
        self.mixers = nn.ModuleList([mixer(dim, state, backend) for _ in range(3)])
        self.bidirectional = bidirectional
        self.local = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
                                   nn.GELU(), nn.Conv2d(dim, dim, 1))

    def forward(self, x):
        # B,T,C,H,W -> B,T,H,W,C; independently scan T, H, W.
        z = x.permute(0, 1, 3, 4, 2)
        for axis, norm, ssm in zip((1, 2, 3), self.norms, self.mixers):
            q = z.movedim(axis, -2)
            shape = q.shape
            q = norm(q.reshape(-1, shape[-2], shape[-1]))
            y = ssm(q)
            if self.bidirectional:
                y = .5 * (y + ssm(q.flip(1)).flip(1))
            z = z + y.reshape(shape).movedim(-2, axis)
        z = z.permute(0, 1, 4, 2, 3).contiguous()
        b, t, c, h, w = z.shape
        return z + self.local(z.reshape(b * t, c, h, w)).reshape(b, t, c, h, w)
