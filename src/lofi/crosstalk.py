"""Slice-to-slice contrast from crosstalk between the slices of a 2D multi-slice acquisition.

A model for spoiled gradient echo (from a research note on slice contrast patterns, kept with the
project's experiments). Each slice's excitation profile has wings that partly
saturate its neighbours; how much a slice has recovered by its own excitation depends on when its
neighbours were hit, which the slice order sets, and on T1, so the result is a gain per slice *and*
per tissue: slice-to-slice contrast changes, not just shading.

Longitudinal magnetisation is tracked on a z grid much finer than a slice, for a set of T1 values:

- an excitation of slice ``j`` multiplies M_z(z) by ``cos(alpha * P(z - z_j))`` (transverse
  magnetisation spoiled), with ``P`` the small-flip profile of a Hann-windowed sinc pulse;
- between excitations every z relaxes towards 1 with its T1;
- slices are excited back to back, ``slice_time_ms`` apart, then the TR's dead time passes (an
  even spread over the TR if ``slice_time_ms`` is None, when interleaving makes no pattern);
- one TR (every slice of a package once, in acquisition order) is an affine map per z,
  ``M -> a M + b``, whose fixed point ``b / (1 - a)`` is the steady state, with no iteration;
- slice ``i``'s signal is the integral of ``M_z`` just before its excitation times
  ``sin(alpha * P(z - z_i))`` (uniform receive weighting).

Packages are acquired one after another, so a package's pulses never reach another's steady
state; where one ends, the timing to neighbours resets, which gives the period-N pattern. Gains are
relative to the mean over slices for each T1, so the source's overall contrast stays and only the
slice-to-slice pattern is added.

Not modelled (the note's later sections): magnetisation transfer, multi-slab, inflow, refocusing
pulses (spin echo/TSE), simultaneous multi-slice.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor

T1_POINTS: int = 24  # T1 values the gain table is computed at (log spaced)
Z_PER_SLICE: int = 24  # fine z samples per slice thickness, at least
WING: float = 3.0  # profile computed out to this many thicknesses from the slice centre
RF_SAMPLES: int = 257


def slice_profile(offset_mm: Tensor, thickness_mm: float, tbw: float) -> Tensor:
    """Small-flip excitation profile of a Hann-windowed sinc pulse, 1 at the centre.

    ``tbw`` is the pulse's time-bandwidth product: low values give broad wings and a soft edge. The
    profile is the pulse's Fourier transform, scaled so its passband spans ``thickness_mm``.
    """
    t = torch.linspace(-0.5, 0.5, RF_SAMPLES, dtype=torch.float64)
    rf = torch.sinc(tbw * t) * 0.5 * (1 + torch.cos(2 * math.pi * t))
    frequency = offset_mm.to(torch.float64)[..., None] * (tbw / thickness_mm)
    profile = (rf * torch.cos(2 * math.pi * frequency * t)).sum(-1) / rf.sum()  # rf is even
    return profile.abs()


def acquisition_order(n: int, packages: int, layout: str, order: str) -> list[list[int]]:
    """Slice indices per package, each in the order it is excited within a TR.

    ``layout`` ``contiguous`` splits the stack into blocks; ``distributed`` gives package ``p``
    slices ``p, p + packages, ...``. ``order`` ``interleaved`` excites ``0, 2, 4, ...`` then
    ``1, 3, 5, ...`` of each package; ``sequential`` goes in order.
    """
    packages = max(min(packages, n), 1)
    if layout == "contiguous":
        bounds = [round(p * n / packages) for p in range(packages + 1)]
        groups = [list(range(bounds[p], bounds[p + 1])) for p in range(packages)]
    elif layout == "distributed":
        groups = [list(range(p, n, packages)) for p in range(packages)]
    else:
        raise ValueError(f"unknown package layout {layout!r}")
    if order == "interleaved":
        return [g[0::2] + g[1::2] for g in groups]
    if order == "sequential":
        return groups
    raise ValueError(f"unknown slice order {order!r}")


def gain_table(
    n: int,
    spacing_mm: float,
    thickness_mm: float,
    flip_deg: float,
    tr_ms: float,
    tbw: float,
    t1_ms: Tensor,
    order: str = "interleaved",
    packages: int = 1,
    layout: str = "contiguous",
    slice_time_ms: float | None = None,
) -> Tensor:
    """Relative signal of each slice for each T1: ``(n, len(t1_ms))``, mean 1 over slices.

    ``tr_ms`` is the time for one pass through a package (each of its slices excited once); its
    slices are excited ``slice_time_ms`` apart, and the rest of the TR is dead time. A TR shorter
    than the package's slices need is lengthened to fit them.
    """
    t1 = t1_ms.to(torch.float64).reshape(-1, 1)
    if n < 2:
        return torch.ones((n, t1.shape[0]), dtype=torch.float64)
    per_spacing = max(math.ceil(Z_PER_SLICE * spacing_mm / thickness_mm), 1)
    dz = spacing_mm / per_spacing  # slice centres fall on the grid
    half = math.ceil(WING * thickness_mm / dz)
    table = slice_profile(torch.arange(-half, half + 1) * dz, thickness_mm, tbw)
    nz = (n - 1) * per_spacing + 2 * half + 1
    alpha = math.radians(flip_deg)

    profiles = torch.zeros((n, nz), dtype=torch.float64)
    for i in range(n):
        start = i * per_spacing
        profiles[i, start : start + 2 * half + 1] = table
    cos, sin = torch.cos(alpha * profiles), torch.sin(alpha * profiles)

    signal = torch.zeros((n, t1.shape[0]), dtype=torch.float64)
    for group in acquisition_order(n, packages, layout, order):
        step = tr_ms / len(group) if slice_time_ms is None else slice_time_ms
        gaps = [step] * (len(group) - 1) + [max(tr_ms - step * (len(group) - 1), step)]
        relax = [torch.exp(-gap / t1) for gap in gaps]  # (T1, 1) after each excitation
        a = torch.ones((t1.shape[0], nz), dtype=torch.float64)
        b = torch.zeros_like(a)
        for j, e in zip(group, relax, strict=True):
            a, b = e * cos[j] * a, e * cos[j] * b + (1 - e)
        mz = b / (1 - a)  # steady state just before the package's first excitation
        for j, e in zip(group, relax, strict=True):
            signal[j] = (mz * sin[j]).sum(-1) * dz
            mz = e * cos[j] * mz + (1 - e)
    return signal / signal.mean(0, keepdim=True)


def t1_from_intensity(
    x: Tensor, dark: float, bright: float, t1_short_ms: float, t1_long_ms: float
) -> Tensor:
    """A T1 surrogate for a T1-weighted source: log T1 falls linearly with intensity.

    Intensities at or below ``dark`` get ``t1_long_ms``, at or above ``bright`` ``t1_short_ms``.
    Brighter means shorter T1 only in a T1-weighted, non-fat-saturated source; elsewhere this is a
    stand-in until segmentations or T1 maps are used.
    """
    frac = ((x - dark) / max(bright - dark, 1e-6)).clamp(0, 1)
    log_t1 = math.log(t1_long_ms) + frac * (math.log(t1_short_ms) - math.log(t1_long_ms))
    return torch.exp(log_t1)


def apply(x: Tensor, table: Tensor, t1_grid: Tensor, t1: Tensor, slice_dim: int = -1) -> Tensor:
    """Scale every voxel of ``x`` by its slice's gain at its T1, interpolated in log T1.

    ``table`` is ``(slices, len(t1_grid))`` from `gain_table`, ``t1`` matches ``x``.
    """
    x_last, t1_last = x.movedim(slice_dim, -1), t1.movedim(slice_dim, -1)
    logs = torch.log(t1_grid.to(x.device, torch.float64))
    points = logs.numel()
    pos = (torch.log(t1_last.to(torch.float64)) - logs[0]) / (logs[-1] - logs[0]) * (points - 1)
    pos = pos.clamp(0, points - 1)
    low = pos.floor().long().clamp(max=points - 2)
    weight = pos - low
    rows = torch.arange(x_last.shape[-1], device=x.device).expand_as(low)
    tab = table.to(x.device)
    gain = (1 - weight) * tab[rows, low] + weight * tab[rows, low + 1]
    return (x_last * gain.to(x.dtype)).movedim(-1, slice_dim)


def t1_grid(t1_short_ms: float, t1_long_ms: float, points: int = T1_POINTS) -> Tensor:
    """Log-spaced T1 values spanning the surrogate's range."""
    low, high = sorted((t1_short_ms, t1_long_ms))
    return torch.logspace(math.log10(low), math.log10(high), points, dtype=torch.float64)


def streaks(
    slabs: Tensor,
    levels: Sequence[float],
    spacing_mm: float,
    thickness_mm: float,
    flip_deg: float,
    tr_ms: float,
    tbw: float,
    t1_short_ms: float,
    t1_long_ms: float,
    order: str = "interleaved",
    packages: int = 1,
    layout: str = "contiguous",
    slice_time_ms: float | None = None,
) -> Tensor:
    """Crosstalk on acquired slices ``(x, y, slices)``; ``levels`` are the dark and bright ends."""
    grid = t1_grid(t1_short_ms, t1_long_ms)
    table = gain_table(
        slabs.shape[-1], spacing_mm, thickness_mm, flip_deg, tr_ms, tbw, grid, order, packages,
        layout, slice_time_ms,
    )  # fmt: skip
    t1 = t1_from_intensity(slabs, levels[0], levels[1], t1_short_ms, t1_long_ms)
    return apply(slabs, table, grid, t1)
