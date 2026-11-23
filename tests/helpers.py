"""Synthetic scans for tests."""

import torch
from torch import Tensor

from lofi.profile import Acquisition, Emi, Fov, Ghost, Noise, Profile, Shading


def phantom(shape: tuple[int, ...], spacing: tuple[float, ...], noise: float = 0.0) -> Tensor:
    """A bright ellipsoid with a dark core on an empty background; radii are in mm."""
    axes = [(torch.arange(n) - (n - 1) / 2) * d for n, d in zip(shape, spacing, strict=True)]
    grid = torch.meshgrid(*axes, indexing="ij")
    radii = [0.32 * n * d for n, d in zip(shape, spacing, strict=True)]
    r = sum((g / a) ** 2 for g, a in zip(grid, radii, strict=True))
    image = (r < 1).float() * 100 + (r < 0.3).float() * 60
    if noise:
        pair = torch.randn((2, *shape), generator=torch.Generator().manual_seed(1))
        image = ((image + noise * pair[0]) ** 2 + (noise * pair[1]) ** 2).sqrt()
    return image


def quiet(**sections: object) -> Profile:
    """A profile with every stage neutral, so a test can switch on the one it is about."""
    base: dict[str, object] = {
        "acquisition": Acquisition(in_plane_mm=4.0, slice_spacing_mm=6.0, slice_thickness_mm=6.0),
        "noise": Noise(sigma=0.0),
        "ghost": Ghost(rms=0.0),
        "shading": Shading(log_rms=0.0),
        "emi": Emi(prob=0.0),
        "fov": Fov(size_mm=1e6, warp_mm=0.0, fade_mm=0.0),
    }
    return Profile(name="test", **{**base, **sections})  # type: ignore[arg-type]
