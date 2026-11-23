"""Scanner profiles: strict YAML presets for one target scanner and protocol.

A value written as ``[low, high]`` is drawn uniformly per sample; a plain number is fixed.
Calibrated values come from `lofi.calibrate`, as a ratio to the foreground median (``relative``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

Range = float | list[float]


@dataclass(frozen=True)
class Acquisition:
    """How the target scanner samples the object."""

    in_plane_mm: Range = 4.0  # acquired voxel size
    slice_spacing_mm: Range = 6.0
    slice_thickness_mm: Range = 6.0
    slice_jitter_mm: Range = 0.0  # each acquisition's slice offset, up to this far either way
    n_averages: int = 2  # independent acquisitions combined after magnitude
    slice_axis: int = 2  # spatial axis
    pe_axis: int = 1  # spatial axis, in the slice plane
    window: float = 0.2  # Tukey taper fraction on the acquired k-space


@dataclass(frozen=True)
class Physics:
    """Predict the noise from field strength instead of using a calibrated level."""

    source_b0: float
    target_b0: float
    lam: float  # R_s / R_c at the source field
    bandwidth_ratio: float = 1.0  # per-sample bandwidth, target over source


@dataclass(frozen=True)
class Noise:
    """Thermal noise on the acquired grid."""

    sigma: Range | None = 0.05  # per real channel, relative to the foreground median
    physics: Physics | None = None  # overrides sigma


@dataclass(frozen=True)
class Ghost:
    """Line-by-line phase errors along the phase-encode axis."""

    rms: Range = 0.1  # radians
    periodic: float = 0.5  # share of rms in a periodic (respiratory) term
    period_lines: Range = 16.0


@dataclass(frozen=True)
class Shading:
    """Receive sensitivity: a smooth multiplicative field, applied before the noise."""

    log_rms: Range = 0.3
    scale_mm: Range = 80.0


@dataclass(frozen=True)
class Emi:
    """Narrowband interference; reconstructs as zipper lines."""

    prob: float = 0.0
    amplitude: Range = 0.1  # relative, per acquired sample


@dataclass(frozen=True)
class Fov:
    """Warped field of view from distortion correction; exact zeros outside it."""

    size_mm: Range = 345.0  # in-plane, square
    warp_mm: Range = 6.0  # edge displacement
    fade_mm: Range = 16.0  # intensity falls off over this distance inside the edge


@dataclass(frozen=True)
class Contrast:
    """Foreground histogram of the target: ``values[i]`` is quantile ``i / (len - 1)``, relative."""

    values: list[float] | None = None


@dataclass(frozen=True)
class Crosstalk:
    """Slice-to-slice contrast from slice-profile crosstalk, 2D multi-slice GRE (`lofi.crosstalk`).

    Off unless ``prob`` > 0; then drawn per sample, after the other stages' draws are fixed.
    """

    prob: float = 0.0
    flip_deg: Range = 30.0
    tr_ms: Range = 300.0  # one pass through a package's slices
    slice_time_ms: Range | None = 6.0  # between successive excitations; None spreads them over TR
    tbw: Range = 2.0  # RF time-bandwidth product: low means broad wings
    order: str = "interleaved"  # or "sequential"
    packages: int = 1
    layout: str = "contiguous"  # or "distributed": package p gets slices p, p + packages, ...
    t1_short_ms: Range = 250.0  # T1 surrogate: the brightest tissue (a T1-weighted source)
    t1_long_ms: Range = 1000.0  # and the darkest


@dataclass(frozen=True)
class Profile:
    """One scanner and protocol."""

    name: str
    target_b0: float = 0.5
    output_spacing_mm: float | None = None  # None keeps the input grid
    canvas_mm: float | None = None  # pad the slice plane to this square frame; None does not
    output_scale: Range = 1.0  # foreground median of the output
    acquisition: Acquisition = field(default_factory=Acquisition)
    noise: Noise = field(default_factory=Noise)
    ghost: Ghost = field(default_factory=Ghost)
    shading: Shading = field(default_factory=Shading)
    emi: Emi = field(default_factory=Emi)
    fov: Fov = field(default_factory=Fov)
    contrast: Contrast = field(default_factory=Contrast)
    crosstalk: Crosstalk = field(default_factory=Crosstalk)


_SECTIONS = {
    "acquisition": Acquisition,
    "noise": Noise,
    "ghost": Ghost,
    "shading": Shading,
    "emi": Emi,
    "fov": Fov,
    "contrast": Contrast,
    "crosstalk": Crosstalk,
}


def _section(cls: type, data: dict[str, Any], where: str) -> Any:
    unknown = set(data) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"unknown key(s) in {where}: {sorted(unknown)}")
    return cls(**data)


def from_dict(data: dict[str, Any]) -> Profile:
    """Build a `Profile`, rejecting unknown keys so a typo cannot silently use a default."""
    data = dict(data)
    for key, cls in _SECTIONS.items():
        if key in data:
            section = dict(data[key])
            if key == "noise" and section.get("physics") is not None:
                section["physics"] = _section(Physics, section["physics"], "noise.physics")
            data[key] = _section(cls, section, key)
    profile: Profile = _section(Profile, data, "profile")
    acquisition = profile.acquisition
    if acquisition.pe_axis == acquisition.slice_axis:
        raise ValueError("pe_axis must lie in the slice plane")
    order, layout = profile.crosstalk.order, profile.crosstalk.layout
    if order not in ("interleaved", "sequential"):
        raise ValueError(f"crosstalk.order must be interleaved or sequential, not {order!r}")
    if layout not in ("contiguous", "distributed"):
        raise ValueError(f"crosstalk.layout must be contiguous or distributed, not {layout!r}")
    return profile


def bundled() -> list[str]:
    """Names of the profiles shipped with the package."""
    root = resources.files("lofi") / "profiles"
    return sorted(p.name[:-5] for p in root.iterdir() if p.name.endswith(".yaml"))


def load(name_or_path: str | Path) -> Profile:
    """Load a bundled profile by name, or a YAML file by path."""
    if str(name_or_path) in bundled():
        text = (resources.files("lofi") / "profiles" / f"{name_or_path}.yaml").read_text()
    else:
        text = Path(name_or_path).read_text()
    data: dict[str, Any] = yaml.safe_load(text)
    return from_dict(data)


def save(profile: Profile, path: str | Path) -> None:
    """Write a profile as YAML."""
    Path(path).write_text(yaml.safe_dump(asdict(profile), sort_keys=False))
