import pytest
import torch

from helpers import phantom
from lofi import masks


def test_foreground_finds_the_object(ndim: int) -> None:
    shape, spacing = (48,) * ndim, (2.0,) * ndim
    image = phantom(shape, spacing)
    assert float(((image > 0) ^ masks.foreground(image, spacing)).float().mean()) < 0.08


def test_erode_and_dilate(ndim: int) -> None:
    mask = torch.zeros((9,) * ndim, dtype=torch.bool)
    mask[(slice(2, 7),) * ndim] = True
    assert int(masks.erode(mask, 1).sum()) == 3**ndim
    assert int(masks.dilate(mask, 1).sum()) == 7**ndim
    assert int(masks.erode(torch.ones_like(mask), 1).sum()) == 7**ndim
    assert int(masks.erode(torch.ones_like(mask), 1, outside=True).sum()) == 9**ndim


def test_inside_fov_ignores_isolated_zeros_and_keeps_the_exact_zero_rim() -> None:
    x = torch.ones(40, 40)
    x[:5] = 0
    x[20, 20] = 0
    fov = masks.inside_fov(x)
    assert not fov[:5].any()
    assert bool(fov[20, 20])


def test_gaussian_preserves_the_mean() -> None:
    x = torch.randn(32, 32, generator=torch.Generator().manual_seed(0))
    smooth = masks.gaussian(x, (1.0, 1.0), 3.0)
    assert float(smooth.mean()) == pytest.approx(float(x.mean()), abs=1e-5)
    assert float(smooth.std()) < float(x.std()) / 3


def test_smooth_field_is_normalised_and_correlated() -> None:
    field = masks.smooth_field((64, 64), (1.0, 1.0), 6.0, torch.Generator().manual_seed(0))
    assert float(field.std()) == pytest.approx(1.0, abs=1e-5)
    assert float((field * field.roll(1, 0)).mean()) > 0.9


def test_otsu_splits_two_groups() -> None:
    x = torch.cat([torch.zeros(100), torch.ones(100) * 10])
    assert 0 < masks.otsu(x) < 10
    assert masks.otsu(torch.ones(4)) == 1.0


def test_quantile_handles_more_voxels_than_torch_allows() -> None:
    x = torch.arange(20_000_000, dtype=torch.float32)
    assert float(masks.quantile(x, 0.5)) == pytest.approx(1e7, rel=1e-2)
