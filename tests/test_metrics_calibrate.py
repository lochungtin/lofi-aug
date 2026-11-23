import dataclasses

import pytest
import torch

from helpers import phantom, quiet
from lofi import calibrate, metrics
from lofi.corrupt import corrupt
from lofi.profile import Ghost, Noise, Shading

SHAPE, SPACING = (80, 80, 60), (2.0, 2.0, 2.0)


def test_measure_reports_every_statistic_relative_to_the_median() -> None:
    image = phantom(SHAPE, SPACING, noise=5.0)
    stats = metrics.measure(image, SPACING)
    scaled = metrics.measure(image * 7, SPACING)
    for key in ("air_median", "air_mean", "air_std", "shading"):
        assert stats[key] == pytest.approx(scaled[key], rel=1e-3)
    assert stats["median"] * 7 == pytest.approx(scaled["median"], rel=1e-3)
    assert len(stats["quantiles"]) == metrics.QUANTILES
    assert len(stats["band_x"]) == len(metrics.BANDS)


def test_band_power_sees_the_acquired_resolution() -> None:
    sharp = phantom(SHAPE, SPACING, noise=2.0)
    soft = corrupt(sharp, SPACING, quiet(), torch.Generator().manual_seed(0)).image
    assert metrics.band_power(soft, SPACING, 0)[2] < metrics.band_power(sharp, SPACING, 0)[2] - 1


def test_grain_is_longer_after_coarse_sampling() -> None:
    profile = quiet(noise=Noise(sigma=0.1))
    generator = torch.Generator().manual_seed(0)
    out = corrupt(phantom(SHAPE, SPACING), SPACING, profile, generator).image
    stats = metrics.measure(out, SPACING)
    assert stats["grain_x_mm"] >= 2.0
    assert stats["grain_z_mm"] >= 4.0


def test_slice_step_separates_interpolated_slices_from_noise() -> None:
    white = torch.randn(SHAPE, generator=torch.Generator().manual_seed(0))
    profile = quiet()
    image = phantom(SHAPE, SPACING) + 1
    out = corrupt(image, SPACING, profile, torch.Generator().manual_seed(0)).image
    assert metrics.slice_step(out) > metrics.slice_step(white)


def test_summarise_gives_three_percentiles_of_scalars_and_lists() -> None:
    rows = [{"a": float(i), "b": [float(i), 2.0 * i]} for i in range(11)]
    out = calibrate.summarise(rows)
    assert out["a"] == pytest.approx([1.0, 5.0, 9.0])
    assert out["b"][1] == pytest.approx([5.0, 10.0])


def test_initial_reads_noise_scale_and_contrast() -> None:
    target = calibrate.summarise(
        [metrics.measure(phantom(SHAPE, SPACING, noise=n), SPACING) for n in (4.0, 6.0, 8.0)]
    )
    profile = calibrate.initial(quiet(), target)
    assert profile.noise.sigma[0] < profile.noise.sigma[1]
    assert len(profile.contrast.values) == metrics.QUANTILES
    assert profile.output_scale[0] > 50


def test_fit_recovers_the_noise_of_a_simulated_target() -> None:
    source = phantom(SHAPE, SPACING)
    truth = quiet(noise=Noise(sigma=0.08), ghost=Ghost(rms=0.03), shading=Shading(log_rms=0.05))
    generator = torch.Generator().manual_seed(9)
    results = [corrupt(source, SPACING, truth, generator) for _ in range(3)]
    target = calibrate.summarise(metrics.measure(r.image, r.spacing) for r in results)
    start = dataclasses.replace(
        truth, noise=Noise(sigma=0.3), ghost=Ghost(rms=0.1), shading=Shading(log_rms=0.2)
    )
    fitted, history = calibrate.fit(start, target, [(source, SPACING)], iterations=8)
    assert fitted.noise.sigma == pytest.approx(0.08, rel=0.25)
    assert abs(history[-1]["air_median"] - 1) < abs(history[0]["air_median"] - 1)
