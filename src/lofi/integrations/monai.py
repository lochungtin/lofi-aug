"""MONAI adapter."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import replace
from typing import Any

import torch
from monai.transforms import MapTransform, Randomizable

from lofi.augment import LowFieldCorrupt
from lofi.profile import Profile
from lofi.profile import load as load_profile


class LowFieldCorruptd(Randomizable, MapTransform):
    """Corrupt ``keys[0]``, keeping any other keys (labels) aligned; the grid is left unchanged.

    Spacing is read from the ``MetaTensor`` affine. Apply before random cropping.
    """

    def __init__(
        self,
        keys: Sequence[Hashable],
        profile: Profile | str = "low_field_breast",
        seed: int | None = None,
        prob: float = 1.0,
    ) -> None:
        MapTransform.__init__(self, keys)
        loaded = load_profile(profile) if isinstance(profile, str) else profile
        self.transform = LowFieldCorrupt(
            replace(loaded, output_spacing_mm=None, canvas_mm=None), seed, prob
        )

    def __call__(self, data: Mapping[Hashable, Any]) -> dict[Hashable, Any]:
        out = dict(data)
        image_key, *label_keys = self.keys
        image = out[image_key]
        spacing = torch.as_tensor(image.affine)[:3, :3].norm(dim=0).tolist()
        result = self.transform({"image": image.as_tensor(), "spacing": spacing})
        out[image_key] = image.clone().copy_(result["image"].to(image.dtype))
        if "fov" in result:
            for key in label_keys:
                out[key] = out[key] * result["fov"].to(out[key].dtype)
        return out
