"""NIfTI input and output (needs the ``io`` extra)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor


def read(path: str | Path, dtype: torch.dtype = torch.float32) -> tuple[Tensor, tuple[float, ...]]:
    """Load a volume in RAS orientation, returning it with its spacing in mm."""
    import nibabel as nib

    image = nib.as_closest_canonical(nib.load(str(path)))  # type: ignore[no-untyped-call]
    data = torch.from_numpy(np.asarray(image.dataobj, dtype=np.float32).copy()).to(dtype)
    return data, tuple(float(z) for z in image.header.get_zooms()[:3])


def write(path: str | Path, image: Tensor, spacing: tuple[float, ...]) -> None:
    """Save a volume with an axis-aligned RAS affine centred on the grid."""
    import nibabel as nib

    affine = np.diag([*spacing, 1.0])
    affine[:3, 3] = [-(n - 1) / 2 * s for n, s in zip(image.shape, spacing, strict=True)]
    array = image.detach().cpu().numpy()
    nib.save(nib.Nifti1Image(array, affine), str(path))  # type: ignore[no-untyped-call]
