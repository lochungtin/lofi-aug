"""Foreground and air masks, and smooth fields, all on regular grids."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


def _frequency(shape: Sequence[int], spacing: Sequence[float], device: torch.device) -> Tensor:
    """Squared spatial frequency (cycles/mm) for an ``rfftn`` of ``shape``."""
    axes = [
        torch.fft.fftfreq(n, d, device=device)
        for n, d in zip(shape[:-1], spacing[:-1], strict=True)
    ]
    axes.append(torch.fft.rfftfreq(shape[-1], spacing[-1], device=device))
    grid = torch.meshgrid(*axes, indexing="ij")
    total = torch.zeros_like(grid[0])
    for g in grid:
        total = total + g**2
    return total


def gaussian(x: Tensor, spacing: Sequence[float], sigma_mm: float) -> Tensor:
    """Periodic Gaussian smoothing of a real image, by FFT."""
    freq = _frequency(x.shape, spacing, x.device)
    kernel = torch.exp(-2 * (torch.pi * sigma_mm) ** 2 * freq)
    out: Tensor = torch.fft.irfftn(torch.fft.rfftn(x) * kernel, s=x.shape)
    return out


def smooth_field(
    shape: Sequence[int],
    spacing: Sequence[float],
    scale_mm: float,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
) -> Tensor:
    """Zero-mean, unit-variance Gaussian random field with correlation length ``scale_mm``."""
    white = torch.randn(tuple(shape), generator=generator, device=device)
    field = gaussian(white, spacing, scale_mm)
    return (field - field.mean()) / field.std().clamp_min(1e-12)


def otsu(x: Tensor, bins: int = 256) -> float:
    """Otsu threshold of ``x`` (any shape)."""
    low, high = float(x.min()), float(x.max())
    if high <= low:
        return low
    hist = torch.histc(x.float(), bins, low, high).double()
    centres = torch.linspace(low, high, bins + 1, device=x.device, dtype=torch.float64)
    centres = (centres[1:] + centres[:-1]) / 2
    weight, moment = hist.cumsum(0), (hist * centres).cumsum(0)
    upper = weight[-1] - weight
    separation = (
        weight
        * upper
        * (moment / weight.clamp_min(1) - (moment[-1] - moment) / upper.clamp_min(1)) ** 2
    )
    return float(centres[separation.argmax()])


def foreground(x: Tensor, spacing: Sequence[float], sigma_mm: float = 3.0) -> Tensor:
    """Bright tissue: Otsu on the smoothed image (a plain threshold catches noise speckle)."""
    smooth = gaussian(x, spacing, sigma_mm)
    return smooth > otsu(smooth)


def erode(mask: Tensor, voxels: int, outside: bool = False) -> Tensor:
    """Binary erosion by a cube of half-width ``voxels``; ``outside`` fills beyond the grid."""
    if voxels < 1:
        return mask
    pool = {2: F.max_pool2d, 3: F.max_pool3d}[mask.ndim]
    data = F.pad((~mask).float()[None, None], [voxels] * (2 * mask.ndim), value=float(not outside))
    grown: Tensor = pool(data, 2 * voxels + 1, stride=1)
    return grown[0, 0] < 0.5


def dilate(mask: Tensor, voxels: int) -> Tensor:
    """Binary dilation by a cube of half-width ``voxels``."""
    return ~erode(~mask, voxels, outside=True)


def inside_fov(x: Tensor, voxels: int = 2) -> Tensor:
    """Everything except the exact-zero region that distortion correction leaves outside the FOV."""
    return ~dilate(erode(x == 0, voxels, outside=True), voxels)


def air(x: Tensor, spacing: Sequence[float], level: float = 0.1, margin: int = 3) -> Tensor:
    """Dark voxels inside the FOV and away from tissue, including the lungs."""
    fov = inside_fov(x)
    if not fov.any():
        return fov
    smooth = gaussian(x, spacing, 2.0)
    reference = torch.quantile(_subsample(x[fov]), 0.99)
    dark = erode(fov & (smooth < level * reference), margin)
    if dark.any():
        return dark
    return erode(fov & ~foreground(x, spacing), margin)  # noise floor above the fixed level


def _subsample(x: Tensor, limit: int = 2_000_000) -> Tensor:
    """``torch.quantile`` refuses very large inputs."""
    if x.numel() <= limit:
        return x
    return x[:: x.numel() // limit + 1]


def quantile(x: Tensor, q: float | Tensor) -> Tensor:
    """Quantile of a large tensor, on an evenly spaced subset."""
    return torch.quantile(_subsample(x.flatten()), q)
