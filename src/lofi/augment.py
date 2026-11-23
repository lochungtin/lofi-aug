"""Dictionary transform for training pipelines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from lofi import profile as profiles
from lofi.corrupt import corrupt
from lofi.profile import Profile


def resample_label(
    label: Tensor, shape: Sequence[int], padding: Sequence[tuple[int, int]] = ()
) -> Tensor:
    """Move ``(*spatial)`` labels onto an output grid: padded like the image, then nearest."""
    if any(any(p) for p in padding):
        label = F.pad(label, [v for pair in reversed(padding) for v in pair])
    if tuple(label.shape) == tuple(shape):
        return label
    data = label.float()[None, None]
    out = F.interpolate(data, tuple(shape), mode="nearest-exact")
    return out[0, 0].to(label.dtype)


class LowFieldCorrupt:
    """Corrupt ``sample[image_key]`` (``(C, *spatial)``), keeping ``label_key`` aligned.

    Labels only change where the output grid differs, and are set to background outside the FOV,
    since no anatomy is visible there. Each channel is corrupted independently. The FOV is returned
    under ``"fov"`` for callers holding further label maps.
    """

    def __init__(
        self,
        profile: Profile | str = "low_field_breast",
        seed: int | None = None,
        prob: float = 1.0,
        image_key: str = "image",
        label_key: str = "label",
        spacing_key: str = "spacing",
        spacing: Sequence[float] | None = None,
    ) -> None:
        self.profile = profiles.load(profile) if isinstance(profile, str) else profile
        self.seed = seed
        self.prob = prob
        self.image_key, self.label_key, self.spacing_key = image_key, label_key, spacing_key
        self.spacing = spacing
        self._generators: dict[torch.device, torch.Generator] = {}

    def _generator(self, device: torch.device) -> torch.Generator:
        """One stream per device, seeded per DataLoader worker."""
        if device not in self._generators:
            info = torch.utils.data.get_worker_info()
            seed = self.seed if self.seed is not None else int(torch.seed() % 2**31)
            generator = torch.Generator(device)
            self._generators[device] = generator.manual_seed(seed + (info.id if info else 0))
        return self._generators[device]

    def __call__(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        out = dict(sample)
        image = torch.as_tensor(np.asarray(sample[self.image_key]))
        if image.ndim not in (3, 4):
            raise ValueError(f"expected (C, *spatial), got shape {tuple(image.shape)}")
        spacing = sample.get(self.spacing_key, self.spacing)
        if spacing is None:
            raise ValueError(f"no spacing: set {self.spacing_key!r} or pass spacing=")
        generator = self._generator(image.device)
        draw = float(torch.rand((), generator=generator, device=image.device))
        if draw >= self.prob:
            return out

        results = [corrupt(channel, spacing, self.profile, generator) for channel in image]
        out[self.image_key] = torch.stack([r.image for r in results])
        out[self.spacing_key] = results[0].spacing
        out["fov"] = results[0].fov
        if sample.get(self.label_key) is not None:
            label = torch.as_tensor(np.asarray(sample[self.label_key]))
            fov = results[0].fov
            padding = results[0].padding
            kept = [resample_label(c, fov.shape, padding) * fov for c in label]
            out[self.label_key] = torch.stack(kept).to(label.dtype)
        return out
