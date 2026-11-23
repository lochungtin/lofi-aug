"""Image statistics that describe how a low-field scan differs from a good one.

Calibration fits a profile to these, and validation compares corrupted scans with real ones on them.
Everything is relative to the foreground median, so scanner units do not matter.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from lofi import masks

BANDS = (0.05, 0.10, 0.15, 0.20, 0.25)  # cycles/mm
QUANTILES = 21


def band_power(x: Tensor, spacing: Sequence[float], axis: int) -> list[float]:
    """log10 share of signal power around each frequency in `BANDS`, along one axis."""
    others = tuple(a for a in range(x.ndim) if a != axis)
    power = (torch.fft.fftn(x - x.mean()).abs() ** 2).sum(others)
    n = x.shape[axis]
    half = power[: n // 2 + 1]
    half = half / half.sum()
    frequency = torch.arange(half.numel(), device=x.device) / (n * spacing[axis])
    out = []
    for band in BANDS:
        near = (frequency - band).abs() <= 0.0125
        out.append(float(half[near].mean().log10()) if near.any() else float("nan"))
    return out


def grain_mm(x: Tensor, mask: Tensor, spacing: Sequence[float], axis: int) -> float:
    """Distance at which the high-pass autocorrelation of ``x`` in ``mask`` falls below one half."""
    detail = x - masks.gaussian(x, spacing, 12.0)
    variance = detail[mask].var().clamp_min(1e-12)
    for lag in range(1, 12):
        both = mask & mask.roll(lag, axis)
        if not both.any():
            break
        centred = detail - detail[mask].mean()
        if float((centred * centred.roll(lag, axis))[both].mean() / variance) < 0.5:
            return lag * float(spacing[axis])
    return 12 * float(spacing[axis])


def slice_step(x: Tensor, axis: int = -1) -> float:
    """How piecewise-linear the image is along ``axis``, from 0 (smooth) towards 1.

    Linear interpolation between thick slices makes the second difference vanish except at the
    slices, so its median is small next to its mean.
    """
    second = x.diff(n=2, dim=axis).abs()
    return 1 - float(masks.quantile(second, 0.5) / second.mean().clamp_min(1e-12))


def measure(image: Tensor, spacing: Sequence[float]) -> dict[str, float | list[float]]:
    """Statistics of one real image."""
    x = image.to(torch.float32).nan_to_num()
    tissue = masks.foreground(x, spacing)
    inner = masks.erode(tissue, 3)
    if not inner.any():
        inner = tissue
    median = masks.quantile(x[inner], 0.5).clamp_min(1e-6)
    air = masks.air(x, spacing)
    out: dict[str, float | list[float]] = {
        "median": float(median),
        "zero_fraction": float((x == 0).float().mean()),
        "negative_fraction": float((x < 0).float().mean()),
        "slice_step": slice_step(x),
    }
    scaled = x / median
    if air.any():
        values = scaled[air]
        out |= {
            "air_median": float(masks.quantile(values, 0.5)),
            "air_mean": float(values.mean()),
            "air_std": float(values.std()),
        }
        for a, name in enumerate("xyz"):
            out[f"grain_{name}_mm"] = grain_mm(scaled, air, spacing, a)
    smooth = masks.gaussian(scaled.clamp_min(1e-3).log(), spacing, 20.0)
    out["shading"] = float(smooth[tissue].std())
    for a, name in enumerate("xyz"):
        out[f"band_{name}"] = band_power(scaled, spacing, a)
    probs = torch.linspace(0, 1, QUANTILES, device=x.device)
    out["quantiles"] = masks.quantile(scaled[inner], probs).tolist()
    return out
