"""Centred unitary FFTs and fixed-FOV changes of matrix size."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor


def fftc(x: Tensor, dim: Sequence[int]) -> Tensor:
    """Unitary centred FFT, with DC at index ``n // 2``."""
    dims = tuple(dim)
    shifted = torch.fft.ifftshift(x, dim=dims)
    out: Tensor = torch.fft.fftshift(torch.fft.fftn(shifted, dim=dims, norm="ortho"), dim=dims)
    return out


def ifftc(k: Tensor, dim: Sequence[int]) -> Tensor:
    """Inverse of `fftc`."""
    dims = tuple(dim)
    shifted = torch.fft.ifftshift(k, dim=dims)
    out: Tensor = torch.fft.fftshift(torch.fft.ifftn(shifted, dim=dims, norm="ortho"), dim=dims)
    return out


def _centre_offset(n: int) -> float:
    """Where index ``n // 2`` sits relative to the FOV centre, in voxels."""
    return n // 2 - (n - 1) / 2


def tukey(n: int, alpha: float, device: torch.device | None = None) -> Tensor:
    """Tukey taper over ``n`` k-space lines; ``alpha`` is the tapered fraction."""
    if alpha <= 0:
        return torch.ones(n, device=device)
    u = (torch.arange(n, device=device, dtype=torch.float32) + 0.5) / n
    edge = (1 - (2 * u - 1).abs()) / alpha
    return 0.5 * (1 - torch.cos(math.pi * edge.clamp(max=1)))


def crop(k: Tensor, dim: int, m: int, alpha: float = 0.0) -> Tensor:
    """Keep the central ``m`` lines of ``dim``: same FOV, coarser voxels.

    Amplitude is rescaled so a uniform object keeps its image intensity, which leaves unit-variance
    noise added afterwards at unit variance per acquired voxel. A linear phase keeps the two grids'
    FOVs aligned for even sizes, so cropping does not shift the image.
    """
    n = k.shape[dim]
    start = n // 2 - m // 2
    out = k.narrow(dim, start, m)
    view = [1] * k.ndim
    view[dim] = m
    line = torch.arange(m, device=k.device, dtype=torch.float32) - m // 2
    shift = _centre_offset(m) * n / m - _centre_offset(n)
    ramp = torch.polar(torch.ones(m, device=k.device), 2 * math.pi * line * shift / n)
    scaled: Tensor = out * (ramp * tukey(m, alpha, k.device)).view(view) * (m / n) ** 0.5
    return scaled


def thick_slices(
    x: Tensor, dim: int, spacing: float, step: float, thickness: float, offset: float = 0.0
) -> Tensor:
    """Box-average ``thickness`` mm slabs every ``step`` mm along ``dim``, centred in the FOV.

    ``offset`` moves the slab positions, in mm, as a repeat acquisition's slices would be displaced.
    """
    n = x.shape[dim]
    length = n * spacing
    m = max(round(length / step), 1)
    centres = (torch.arange(m, device=x.device, dtype=x.real.dtype) + 0.5) * length / m + offset
    half = thickness / 2
    moved = x.movedim(dim, -1)
    total = torch.cat([torch.zeros_like(moved[..., :1]), moved.cumsum(-1)], dim=-1)

    def integral(edges_mm: Tensor) -> Tensor:
        pos = (edges_mm / spacing).clamp(0, n)
        low = pos.floor().long().clamp(max=n - 1)
        return total[..., low] + (pos - low) * moved[..., low]

    width = ((centres + half).clamp(0, length) - (centres - half).clamp(0, length)) / spacing
    out = (integral(centres + half) - integral(centres - half)) / width.clamp_min(1e-6)
    moved_back: Tensor = out.movedim(-1, dim)
    return moved_back
