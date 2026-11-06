import torch


def chebyshev_basis(r, order):
    lo, hi = r.min(), r.max()
    x = ((r - lo) / (hi - lo + 1e-8) * 2.0 - 1.0).clamp(-1 + 1e-7, 1 - 1e-7)
    n = torch.arange(order, dtype=r.dtype, device=r.device)
    return torch.cos(torch.outer(torch.arccos(x), n))
