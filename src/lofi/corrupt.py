"""Corrupt one image: the forward model from a good scan to a low-field reconstruction.

Stages run in acquisition order, so what the image shows emerges rather than being painted on: noise
goes in on the coarse acquired grid before the magnitude, so the Rician floor and the noise grain
come from the same reconstruction as the anatomy.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from lofi import crosstalk, kspace, masks, physics
from lofi.profile import Profile, Range

RAYLEIGH_MEDIAN: float = (2 * math.log(2)) ** 0.5


@dataclass(frozen=True)
class Result:
    """A corrupted image on its output grid."""

    image: Tensor  # (*spatial)
    spacing: tuple[float, ...]
    fov: Tensor  # bool, (*spatial): where the scanner sees anything
    padding: tuple[tuple[int, int], ...]  # voxels added (before, after) per spatial axis


def draw(value: Range, generator: torch.Generator | None, device: torch.device) -> float:
    """Resolve a profile value: a number is fixed, ``[low, high]`` is drawn uniformly."""
    if isinstance(value, (int, float)):
        return float(value)
    low, high = value
    return float(low + (high - low) * float(torch.rand((), generator=generator, device=device)))


def estimate_sigma(image: Tensor, spacing: Sequence[float]) -> float:
    """Noise per real channel from the Rayleigh median of the air; 0 if there is no air."""
    air = masks.air(image, spacing)
    if not air.any():
        return 0.0
    return float(masks.quantile(image[air], 0.5)) / float(RAYLEIGH_MEDIAN)


def match_contrast(
    image: Tensor,
    reference: Tensor,
    values: Sequence[float] | None,
    weight: Tensor | None = None,
) -> Tensor:
    """Monotonic map taking the ``reference`` voxels' histogram onto ``values``.

    ``weight`` (0 to 1) blends the map in, so it can stay on tissue: fat-sat sources have tissue
    almost as dark as their air, and any gain applied there would light up the background. This
    changes intensities only; it cannot turn one tissue's contrast into another's.
    """
    target = torch.tensor(values or [], device=image.device, dtype=image.dtype)
    if target.numel() < 2:
        return image
    probs = torch.linspace(0, 1, target.numel(), device=image.device, dtype=image.dtype)
    source = masks.quantile(image[reference], probs)
    source = torch.maximum(source, source.cummax(0).values + 1e-6 * probs)
    index = torch.bucketize(image, source).clamp(1, source.numel() - 1)
    low, high = source[index - 1], source[index]
    out = target[index - 1] + (image - low) / (high - low) * (target[index] - target[index - 1])
    below = image * target[0] / source[0].clamp_min(1e-6)
    above = image * target[-1] / source[-1]
    high_side = torch.where(image > source[-1], above, out)
    mapped = torch.where(image < source[0], below, high_side)
    return mapped if weight is None else weight * mapped + (1 - weight) * image


def _line_phases(
    profile: Profile, slices: int, lines: int, gen: torch.Generator | None, device: torch.device
) -> Tensor:
    """Per-slice, per-line phase errors ``(lines, slices)`` around the k-space centre."""
    ghost = profile.ghost
    rms = draw(ghost.rms, gen, device)
    period = draw(ghost.period_lines, gen, device)
    line = torch.arange(lines, device=device, dtype=torch.float32)[:, None]
    start = 2 * math.pi * torch.rand((1, slices), generator=gen, device=device)
    wave = 2**0.5 * torch.sin(2 * math.pi * line / period + start)
    white = torch.randn((lines, slices), generator=gen, device=device)
    share = ghost.periodic
    theta: Tensor = rms * (share**0.5 * wave + (1 - share) ** 0.5 * white)
    return theta


def _emi(
    profile: Profile, shape: Sequence[int], gen: torch.Generator | None, device: torch.device
) -> Tensor | None:
    """A tone at a fixed read position, its phase running on across lines; ``(read, lines, 1)``."""
    if float(torch.rand((), generator=gen, device=device)) >= profile.emi.prob:
        return None
    amplitude = draw(profile.emi.amplitude, gen, device)
    read, lines = shape
    frequency = float(torch.rand((), generator=gen, device=device)) - 0.5
    step = 2 * math.pi * frequency
    r = torch.arange(read, device=device, dtype=torch.float32)[:, None]
    p = torch.arange(lines, device=device, dtype=torch.float32)[None]
    tone = torch.polar(torch.ones_like(r * p), step * r + (step * read) * p)
    return (amplitude * tone)[..., None]


def _acquire(
    object_: Tensor,
    in_plane: tuple[int, int],
    pe_dim: int,
    sigma: float,
    alpha: float,
    profile: Profile,
    spacing: Sequence[float],
    gen: torch.Generator | None,
) -> Tensor:
    """One acquisition of ``(x, y, slices)`` real data; returns the acquired-grid magnitude."""
    device = object_.device
    phase = 2 * math.pi * masks.smooth_field(object_.shape, spacing, 40.0, gen, device)
    k = kspace.fftc(torch.polar(object_, phase), (0, 1))
    k = kspace.crop(kspace.crop(k, 0, in_plane[0], alpha), 1, in_plane[1], alpha)

    theta = _line_phases(profile, k.shape[2], k.shape[pe_dim], gen, device)
    if pe_dim == 0:
        k = k * torch.polar(torch.ones_like(theta), theta)[:, None]
    else:
        k = k * torch.polar(torch.ones_like(theta), theta)[None]
    tone = _emi(profile, k.shape[:2], gen, device)
    if tone is not None:
        k = k + tone
    white = torch.randn((2, *k.shape), generator=gen, device=device, dtype=object_.dtype)
    k = k + sigma * torch.complex(white[0], white[1])
    return kspace.ifftc(k, (0, 1)).abs()


def _reconstruct(x: Tensor, shape: tuple[int, int, int]) -> Tensor:
    """Resampling onto the output grid, linear on every axis."""
    plane = x.permute(2, 0, 1)[:, None]
    plane = F.interpolate(plane, shape[:2], mode="bilinear", align_corners=False)
    lines = plane[:, 0].permute(1, 2, 0).reshape(-1, 1, x.shape[2])
    return F.interpolate(lines, shape[2], mode="linear", align_corners=False).reshape(shape)


def _fov_mask(
    shape: tuple[int, int, int],
    spacing: Sequence[float],
    profile: Profile,
    gen: torch.Generator | None,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Warped FOV (bool) and the fade towards its edges, both ``(x, y, slices)``."""
    half = draw(profile.fov.size_mm, gen, device) / 2
    warp = draw(profile.fov.warp_mm, gen, device)
    fade = max(draw(profile.fov.fade_mm, gen, device), 1e-3)
    plane = (shape[0], shape[1], 1)
    axes = [
        (torch.arange(n, device=device, dtype=torch.float32) - (n - 1) / 2) * d
        for n, d in zip(plane[:2], spacing[:2], strict=True)
    ]
    x, y = torch.meshgrid(*axes, indexing="ij")
    wobble = masks.smooth_field((*shape[:2],), spacing[:2], 50.0, gen, device)
    wobble_y = masks.smooth_field((*shape[:2],), spacing[:2], 50.0, gen, device)
    edge_x = half - x.abs() + warp * wobble
    edge_y = half - y.abs() + warp * wobble_y
    inside = (edge_x > 0) & (edge_y > 0)
    ramp = (edge_x / fade).clamp(0, 1) * (edge_y / fade).clamp(0, 1)
    return (
        inside[..., None].expand(shape).clone(),
        (ramp * (2 - ramp))[..., None].expand(shape).clone(),
    )


def _side_generator(gen: torch.Generator | None, device: torch.device) -> torch.Generator | None:
    """A generator seeded from ``gen``'s state without drawing from it, so an optional stage's
    draws leave every other stage's unchanged."""
    if gen is None:
        return None
    digest = hashlib.blake2b(gen.get_state().numpy().tobytes(), digest_size=8).digest()
    return torch.Generator(device).manual_seed(int.from_bytes(digest, "little") >> 1)


def _crosstalk(
    profile: Profile, x: Tensor, tissue: Tensor, gen: torch.Generator | None, device: torch.device
) -> Callable[[Tensor, float, float], Tensor] | None:
    """The crosstalk stage for this sample, as a function of (slices, spacing, thickness)."""
    c = profile.crosstalk
    if c.prob <= 0 or x.shape[2] < 2:
        return None
    sub = _side_generator(gen, device)
    if float(torch.rand((), generator=sub, device=device)) >= c.prob:
        return None
    levels = masks.quantile(x[tissue], torch.tensor([0.05, 0.95], device=device))
    short = draw(c.t1_short_ms, sub, device)
    settings = {
        "levels": [float(v) for v in levels],
        "flip_deg": draw(c.flip_deg, sub, device),
        "tr_ms": draw(c.tr_ms, sub, device),
        "tbw": draw(c.tbw, sub, device),
        "t1_short_ms": short,
        "t1_long_ms": max(draw(c.t1_long_ms, sub, device), short * 1.01),
        "order": c.order,
        "packages": c.packages,
        "layout": c.layout,
        "slice_time_ms": None if c.slice_time_ms is None else draw(c.slice_time_ms, sub, device),
    }

    def stage(slabs: Tensor, spacing_mm: float, thickness_mm: float) -> Tensor:
        return crosstalk.streaks(
            slabs,
            spacing_mm=spacing_mm,
            thickness_mm=thickness_mm,
            **settings,  # type: ignore[arg-type]
        )

    return stage


def corrupt(
    image: Tensor,
    spacing: Sequence[float],
    profile: Profile,
    generator: torch.Generator | None = None,
) -> Result:
    """Corrupt a real 2D or 3D image; ``spacing`` is in mm per spatial axis."""
    if image.ndim not in (2, 3):
        raise ValueError(f"expected a 2D or 3D image, got shape {tuple(image.shape)}")
    flat = image.ndim == 2
    x = (image[..., None] if flat else image).to(torch.float32).nan_to_num().clamp_min(0)
    spacing = (*spacing, 1.0) if flat else tuple(spacing)
    gen, device = generator, x.device
    acq = profile.acquisition

    slice_axis = 2 if flat else acq.slice_axis
    plane = [a for a in range(3) if a != slice_axis]
    order = (*plane, slice_axis)
    x = x.permute(order)
    grid = [float(spacing[a]) for a in order]
    pe_dim = plane.index(acq.pe_axis)

    # Normalise to the foreground median, on which every relative profile value is based.
    tissue = masks.foreground(x, grid)
    median = masks.quantile(x[tissue], 0.5).clamp_min(1e-6)
    source_sigma = estimate_sigma(x, grid) / float(median)
    x = x / median
    x = match_contrast(
        x, tissue, profile.contrast.values, masks.gaussian(tissue.float(), grid, 2.0)
    )
    streaks = None if flat else _crosstalk(profile, x, tissue, gen, device)

    # The scanner's frame: tissue keeps its place, the new voxels are empty.
    pads = [(0, 0), (0, 0), (0, 0)]
    if profile.canvas_mm is not None:
        for dim in (0, 1):
            extra = max(round(profile.canvas_mm / grid[dim]) - x.shape[dim], 0)
            pads[dim] = (extra // 2, extra - extra // 2)
        x = F.pad(x, (0, 0, *pads[1], *pads[0]))

    # Output grid.
    out_mm = profile.output_spacing_mm
    if out_mm is None:
        out_shape = (x.shape[0], x.shape[1], x.shape[2])
        out_grid = tuple(grid)
    else:
        out_shape = (
            round(x.shape[0] * grid[0] / out_mm),
            round(x.shape[1] * grid[1] / out_mm),
            max(round(x.shape[2] * grid[2] / out_mm), 1),
        )
        out_grid = (out_mm, out_mm, out_mm if not flat else grid[2])

    # Slice selection, then in-plane sampling.
    step = draw(acq.slice_spacing_mm, gen, device)
    thickness = draw(acq.slice_thickness_mm, gen, device)
    jitter = draw(acq.slice_jitter_mm, gen, device)

    def select(offset: float) -> Tensor:
        return x if flat else kspace.thick_slices(x, 2, grid[2], step, thickness, offset)

    n_slabs = select(0.0).shape[2]
    slab_mm = thickness if flat else grid[2] * x.shape[2] / n_slabs
    voxel = draw(acq.in_plane_mm, gen, device)
    in_plane = (
        min(max(round(x.shape[0] * grid[0] / voxel), 2), x.shape[0]),
        min(max(round(x.shape[1] * grid[1] / voxel), 2), x.shape[1]),
    )

    # Receive sensitivity.
    log_rms = draw(profile.shading.log_rms, gen, device)
    scale = draw(profile.shading.scale_mm, gen, device)
    slab_grid = (grid[0], grid[1], slab_mm)
    shape = (x.shape[0], x.shape[1], n_slabs)
    gain = torch.exp(log_rms * masks.smooth_field(shape, slab_grid, scale, gen, device))

    # The source's own noise survives into the acquired grid, attenuated by the voxel change.
    volume_gain = (voxel / grid[0]) * (voxel / grid[1]) * (thickness / grid[2] if not flat else 1)
    carried = source_sigma / max(volume_gain, 1.0) ** 0.5
    noise = profile.noise
    if noise.physics is not None:
        p = noise.physics
        a = p.target_b0 / p.source_b0
        target = carried / physics.snr_ratio(a, p.lam, p.bandwidth_ratio)
    else:
        target = draw(noise.sigma if noise.sigma is not None else 0.0, gen, device)
    sigma = max(target**2 - carried**2, 0.0) ** 0.5

    # Independent acquisitions, combined after magnitude and interpolation.
    args = (in_plane, pe_dim, sigma, acq.window, profile, slab_grid, gen)
    frames = []
    for _ in range(max(acq.n_averages, 1)):
        offset = jitter * (2 * float(torch.rand((), generator=gen, device=device)) - 1)
        slabs = select(offset)
        if streaks is not None:  # a signal effect: before the receive field and the noise
            slabs = streaks(slabs, step, thickness)
        frames.append(_reconstruct(_acquire(slabs * gain, *args), out_shape))
    out = torch.stack(frames).mean(0)

    inside, ramp = _fov_mask(out_shape, out_grid, profile, gen, device)
    out = out * ramp * inside * draw(profile.output_scale, gen, device)

    inverse = [order.index(a) for a in range(3)]
    out, inside = out.permute(inverse), inside.permute(inverse)
    final = tuple(out_grid[order.index(a)] for a in range(3))
    padding = tuple(pads[order.index(a)] for a in range(3))
    if flat:
        return Result(out[..., 0], final[:2], inside[..., 0], padding[:2])
    return Result(out, final, inside, padding)
