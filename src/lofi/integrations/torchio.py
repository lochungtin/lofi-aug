"""TorchIO adapter."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torchio as tio

from lofi.augment import LowFieldCorrupt
from lofi.profile import Profile
from lofi.profile import load as load_profile


class RandomLowField(tio.transforms.Transform):
    """Corrupt each ``ScalarImage``; each ``LabelMap`` keeps its grid, less out-of-FOV voxels."""

    def __init__(
        self, profile: Profile | str = "low_field_breast", seed: int | None = None, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        loaded = load_profile(profile) if isinstance(profile, str) else profile
        self.transform = LowFieldCorrupt(
            replace(loaded, output_spacing_mm=None, canvas_mm=None), seed
        )

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        images = subject.get_images(intensity_only=True)
        fov = None
        for image in images:
            result = self.transform({"image": image.data.float(), "spacing": image.spacing})
            image.set_data(result["image"])
            fov = result.get("fov", fov)
        if fov is not None:
            for image in subject.get_images(intensity_only=False):
                if isinstance(image, tio.LabelMap):
                    image.set_data(image.data * fov.to(image.data.dtype))
        return subject
