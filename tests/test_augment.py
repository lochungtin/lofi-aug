import numpy as np
import pytest
import torch

from helpers import phantom, quiet
from lofi.augment import LowFieldCorrupt, resample_label
from lofi.profile import Fov, Noise, Profile

SPACING = (2.0, 2.0, 2.0)


def sample() -> dict[str, object]:
    image = phantom((64, 64, 48), SPACING)
    return {"image": image[None], "label": (image > 0).short()[None], "spacing": SPACING}


def test_labels_are_exact_inside_the_fov_and_background_outside() -> None:
    profile = quiet(fov=Fov(size_mm=90.0, warp_mm=3.0, fade_mm=5.0))
    before = sample()
    out = LowFieldCorrupt(profile, seed=0)(before)
    fov = out["fov"]
    assert torch.equal(out["label"][0][fov], before["label"][0][fov])
    assert int(out["label"][0][~fov].sum()) == 0
    assert out["label"].dtype == torch.int16
    assert torch.all(out["image"][0][~fov] == 0)


def test_is_reproducible_and_streams_differ_between_calls() -> None:
    profile = quiet(noise=Noise(sigma=0.05))
    a, b = LowFieldCorrupt(profile, seed=3), LowFieldCorrupt(profile, seed=3)
    first = a(sample())["image"]
    assert torch.equal(first, b(sample())["image"])
    assert not torch.equal(first, a(sample())["image"])


def test_prob_zero_passes_the_sample_through() -> None:
    before = sample()
    out = LowFieldCorrupt(quiet(), seed=0, prob=0.0)(before)
    assert out["image"] is before["image"]
    assert "fov" not in out


def test_accepts_numpy_and_a_default_spacing() -> None:
    image = phantom((48, 48), (2.0, 2.0)).numpy()[None]
    assert isinstance(image, np.ndarray)
    out = LowFieldCorrupt(quiet(), seed=0, spacing=(2.0, 2.0))({"image": image})
    assert out["image"].shape == (1, 48, 48)


def test_needs_a_spacing_and_a_channel_axis() -> None:
    with pytest.raises(ValueError, match="spacing"):
        LowFieldCorrupt(quiet(), seed=0)({"image": torch.zeros(1, 8, 8)})
    with pytest.raises(ValueError, match="spatial"):
        LowFieldCorrupt(quiet(), seed=0, spacing=(1.0, 1.0))({"image": torch.zeros(8, 8)})


def test_output_spacing_resamples_labels_with_the_image() -> None:
    profile = Profile(**{**quiet().__dict__, "output_spacing_mm": 4.0})
    out = LowFieldCorrupt(profile, seed=0)(sample())
    assert out["image"].shape == out["label"].shape == (1, 32, 32, 24)
    assert out["spacing"] == (4.0, 4.0, 4.0)


def test_canvas_pads_image_and_labels_together() -> None:
    profile = Profile(**{**quiet().__dict__, "canvas_mm": 180.0})
    before = sample()
    out = LowFieldCorrupt(profile, seed=0)(before)
    assert out["image"].shape == out["label"].shape == (1, 90, 90, 48)
    assert out["spacing"] == SPACING
    assert int(out["label"][0, :13].sum()) == 0
    assert torch.equal(out["label"][0, 13:77, 13:77], before["label"][0])
    tissue = out["image"][0] > 0.5 * out["image"].max()
    both = out["label"][0].bool() & tissue
    assert float(both.sum() / tissue.sum()) > 0.9


def test_resample_label_keeps_classes_and_same_shape_passes() -> None:
    label = torch.randint(0, 4, (10, 10), generator=torch.Generator().manual_seed(0))
    assert resample_label(label, (10, 10)) is label
    assert set(resample_label(label, (5, 5)).unique().tolist()) <= {0, 1, 2, 3}
