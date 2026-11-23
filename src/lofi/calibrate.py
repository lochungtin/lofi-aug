"""Fit a scanner profile to real target scans.

`summarise` measures the target; `fit` then corrupts a few good scans and adjusts the two values
with no direct reading (noise, shading) until the corrupted statistics agree. Acquisition geometry
is not fitted: it comes from the spectra and the protocol.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import torch
from torch import Tensor

from lofi.corrupt import RAYLEIGH_MEDIAN, corrupt
from lofi.metrics import measure
from lofi.profile import Contrast, Noise, Profile

Scan = tuple[Tensor, Sequence[float]]


def summarise(stats: Iterable[dict[str, float | list[float]]]) -> dict[str, Any]:
    """10th, 50th and 90th percentile of each statistic across scans; lists are element-wise."""
    rows = list(stats)
    out: dict[str, Any] = {}
    for key in rows[0]:
        values = np.array([r[key] for r in rows if key in r], dtype=np.float64)
        out[key] = np.nanpercentile(values, [10, 50, 90], axis=0).tolist()
    return out


def initial(profile: Profile, target: dict[str, Any]) -> Profile:
    """Read the directly measurable values from the target."""
    low, _, high = target["air_median"]
    median = target["median"]
    return replace(
        profile,
        output_scale=[median[0], median[2]],
        noise=Noise(sigma=[low / RAYLEIGH_MEDIAN, high / RAYLEIGH_MEDIAN]),
        contrast=Contrast(values=[round(float(v), 4) for v in target["quantiles"][1]]),
    )


def _rescale(value: float | list[float], gain: float, digits: int = 5) -> float | list[float]:
    if isinstance(value, (int, float)):
        return round(float(value) * gain, digits)
    return [round(float(v) * gain, digits) for v in value]


def fit(
    profile: Profile,
    target: dict[str, Any],
    sources: Sequence[Scan],
    iterations: int = 4,
    seed: int = 0,
    damping: float = 0.7,
) -> tuple[Profile, list[dict[str, float]]]:
    """Multiplicative updates of the noise (air median) and shading.

    Returns the fitted profile and the history of what each iteration measured. Ghosting is not
    fitted: air spread is dominated by lung structure, so matching it destroys the anatomy.
    """
    history: list[dict[str, float]] = []
    for step in range(iterations):
        generator = torch.Generator(sources[0][0].device).manual_seed(seed + step)
        rows = []
        for image, spacing in sources:
            result = corrupt(image, spacing, profile, generator)
            rows.append(measure(result.image, result.spacing))
        got = summarise(rows)
        record = {key: got[key][1] / target[key][1] for key in ("air_median", "shading")}
        history.append(record)

        sigma = profile.noise.sigma if profile.noise.sigma is not None else 0.05
        noise_gain = float(np.clip(record["air_median"] ** -damping, 0.5, 2.0))
        shading_gain = float(np.clip(record["shading"] ** -damping, 0.5, 2.0))
        profile = replace(
            profile,
            noise=replace(profile.noise, sigma=_rescale(sigma, noise_gain)),
            shading=replace(
                profile.shading, log_rms=_rescale(profile.shading.log_rms, shading_gain)
            ),
        )
    return profile, history
