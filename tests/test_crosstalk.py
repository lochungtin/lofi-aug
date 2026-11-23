import dataclasses
import math

import pytest
import torch

from helpers import phantom, quiet
from lofi import crosstalk
from lofi.corrupt import corrupt
from lofi.profile import Crosstalk, from_dict

T1 = torch.tensor([250.0, 1000.0])


def table(n: int = 20, **kw: object) -> torch.Tensor:
    args: dict[str, object] = {
        "spacing_mm": 6.0, "thickness_mm": 6.0, "flip_deg": 45.0, "tr_ms": 400.0, "tbw": 2.0,
        "t1_ms": T1, "slice_time_ms": 5.0,
    }  # fmt: skip
    return crosstalk.gain_table(n, **{**args, **kw})  # type: ignore[arg-type]


def test_profile_is_one_at_the_centre_half_near_the_edge_and_falls_away() -> None:
    p = crosstalk.slice_profile(torch.tensor([0.0, 3.0, 12.0]), 6.0, 4.0)
    assert p[0] == pytest.approx(1.0)
    assert 0.3 < p[1] < 0.7
    assert p[2] < 0.02


def test_acquisition_order() -> None:
    assert crosstalk.acquisition_order(6, 1, "contiguous", "interleaved") == [[0, 2, 4, 1, 3, 5]]
    assert crosstalk.acquisition_order(6, 2, "contiguous", "sequential") == [[0, 1, 2], [3, 4, 5]]
    assert crosstalk.acquisition_order(6, 2, "distributed", "interleaved") == [[0, 4, 2], [1, 5, 3]]
    with pytest.raises(ValueError):
        crosstalk.acquisition_order(6, 1, "contiguous", "random")


def test_gains_average_one_over_slices() -> None:
    assert torch.allclose(table().mean(0), torch.ones(2, dtype=torch.float64))


def test_far_apart_slices_do_not_interact() -> None:
    assert torch.allclose(table(spacing_mm=40.0), torch.ones(1, dtype=torch.float64), atol=1e-4)


def test_interleaving_darkens_the_second_half_more_for_long_t1() -> None:
    g = table()
    inner = g[4:16]
    first, second = inner[0::2], inner[1::2]  # even slices are excited first in each TR
    assert (first > second).all()
    swing = (first.mean(0) - second.mean(0)) / g.mean(0)
    assert swing[1] > swing[0] > 0.005  # the long-T1 tissue changes more: contrast, not shading


def test_packages_repeat_their_pattern() -> None:
    g = table(n=24, packages=3, order="sequential")
    assert torch.allclose(g[0:8], g[8:16]) and torch.allclose(g[8:16], g[16:24])
    assert not torch.allclose(g[0:8], g[0:8].mean(0).expand(8, 2), atol=1e-3)


def test_steady_state_matches_iterating_many_trs() -> None:
    """The closed form against brute force: run the pulse train until it settles."""
    n, spacing, thickness, flip, tr, tbw, dt = 5, 6.0, 6.0, 60.0, 200.0, 2.0, 8.0
    closed = table(n, spacing_mm=spacing, thickness_mm=thickness, flip_deg=flip, tr_ms=tr, tbw=tbw,
                   slice_time_ms=dt, order="sequential")  # fmt: skip
    z = torch.arange(-30.0, (n - 1) * spacing + 30.0, 0.05)
    alpha = math.radians(flip)
    p = torch.stack([crosstalk.slice_profile(z - i * spacing, thickness, tbw) for i in range(n)])
    gaps = [dt] * (n - 1) + [tr - dt * (n - 1)]
    signal = torch.zeros((n, 2), dtype=torch.float64)
    for k, t1 in enumerate(T1.tolist()):
        mz = torch.ones_like(z, dtype=torch.float64)
        for _ in range(60):
            for i, gap in enumerate(gaps):
                signal[i, k] = (mz * torch.sin(alpha * p[i])).sum()
                mz = mz * torch.cos(alpha * p[i])
                mz = 1 - (1 - mz) * math.exp(-gap / t1)
    assert torch.allclose(closed, signal / signal.mean(0), atol=2e-3)


def test_t1_surrogate_and_apply() -> None:
    x = torch.tensor([0.0, 0.5, 1.0, 2.0])
    t1 = crosstalk.t1_from_intensity(x, 0.0, 1.0, 250.0, 1000.0)
    assert t1[0] == pytest.approx(1000.0) and t1[2] == pytest.approx(250.0)
    assert t1[3] == pytest.approx(250.0) and t1[1] == pytest.approx(500.0)
    grid = crosstalk.t1_grid(250.0, 1000.0, 3)
    gains = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64)
    slabs = torch.ones((1, 1, 2))
    out = crosstalk.apply(slabs, gains, grid, torch.tensor([[[500.0, 1000.0]]]))
    assert torch.allclose(out, torch.tensor([[[2.0, 6.0]]]))


SHAPE, SPACING = (64, 64, 96), (2.0, 2.0, 2.0)


def test_corrupt_adds_slice_to_slice_changes_and_nothing_else() -> None:
    image = phantom(SHAPE, SPACING)
    base = quiet()
    on = dataclasses.replace(base, crosstalk=Crosstalk(prob=1.0, flip_deg=60.0, tr_ms=400.0))
    tiny = dataclasses.replace(base, crosstalk=Crosstalk(prob=1.0, flip_deg=1e-4))
    off = corrupt(image, SPACING, base, torch.Generator().manual_seed(3)).image
    weak = corrupt(image, SPACING, tiny, torch.Generator().manual_seed(3)).image
    strong = corrupt(image, SPACING, on, torch.Generator().manual_seed(3)).image
    assert torch.allclose(off, weak, atol=1e-4)  # the stage leaves the other draws alone
    tissue = off > 0.5 * off.max()
    ratio = torch.where(tissue, strong / off.clamp_min(1e-6), torch.nan)
    per_slice = ratio.nanmean((0, 1))
    per_slice = per_slice[~per_slice.isnan()]
    assert per_slice.std() > 0.005


def test_profile_rejects_an_unknown_order() -> None:
    with pytest.raises(ValueError):
        from_dict({"name": "x", "crosstalk": {"order": "random"}})
    assert from_dict({"name": "x", "crosstalk": {"prob": 0.5}}).crosstalk.prob == 0.5
