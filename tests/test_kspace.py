import pytest
import torch

from lofi import kspace


def test_fft_is_unitary_and_invertible() -> None:
    x = torch.randn(16, 15, dtype=torch.float64).to(torch.complex128)
    k = kspace.fftc(x, (0, 1))
    assert float((k.abs() ** 2).sum()) == pytest.approx(float((x.abs() ** 2).sum()))
    assert torch.allclose(kspace.ifftc(k, (0, 1)), x)


@pytest.mark.parametrize(("n", "m"), [(64, 32), (64, 16), (63, 21), (64, 31), (65, 32)])
def test_crop_keeps_intensity_and_position(n: int, m: int) -> None:
    pos = (torch.arange(n, dtype=torch.float64) + 0.5) / n
    x = 5.0 + torch.exp(-(((pos - 0.3) / 0.08) ** 2))
    k = kspace.fftc(x.to(torch.complex128), (0,))
    low = kspace.ifftc(kspace.crop(k, 0, m), (0,))
    assert float(low.imag.abs().max()) < 5e-3
    grid = (torch.arange(m, dtype=torch.float64) + 0.5) / m
    bump = low.real - 5.0
    centroid = float((bump * grid).sum() / bump.sum())
    assert centroid == pytest.approx(0.3, abs=0.002)
    assert float(low.real.mean()) == pytest.approx(float(x.mean()), rel=1e-6)


def test_white_noise_stays_at_its_level_per_acquired_voxel() -> None:
    gen = torch.Generator().manual_seed(0)
    k = torch.complex(*torch.randn(2, 128, 128, generator=gen)) / 2**0.5
    cropped = kspace.crop(k, 0, 64) * 2**0.5
    assert float(cropped.abs().pow(2).mean()) == pytest.approx(1.0, abs=0.03)


def test_tukey_tapers_only_the_edges() -> None:
    window = kspace.tukey(100, 0.2)
    assert float(window[50]) == 1.0
    assert float(window[0]) < 0.05
    assert torch.allclose(window, window.flip(0))
    assert torch.equal(kspace.tukey(8, 0.0), torch.ones(8))


def test_thick_slices_average_and_conserve() -> None:
    x = torch.arange(60, dtype=torch.float64).reshape(1, 60)
    out = kspace.thick_slices(x, 1, 1.0, 3.0, 3.0)
    assert out.shape == (1, 20)
    assert torch.allclose(out[0, :3], torch.tensor([1.0, 4.0, 7.0], dtype=torch.float64))
    assert float(out.mean()) == pytest.approx(float(x.mean()))


def test_thin_slices_with_gaps_sample_only_their_slab() -> None:
    x = torch.zeros(60, dtype=torch.float64)
    x[4] = 1.0  # between slabs at 1-2 and 7-8
    out = kspace.thick_slices(x, 0, 1.0, 6.0, 2.0)
    assert float(out.sum()) == 0.0


def test_offset_moves_the_slabs() -> None:
    x = torch.zeros(60, dtype=torch.float64)
    x[28:30] = 1.0
    still = kspace.thick_slices(x, 0, 1.0, 6.0, 6.0)
    moved = kspace.thick_slices(x, 0, 1.0, 6.0, 6.0, offset=6.0)
    assert int(moved.argmax()) == int(still.argmax()) - 1
    assert torch.isfinite(moved).all()  # the last slab has left the FOV
