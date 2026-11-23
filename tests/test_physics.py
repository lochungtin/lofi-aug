import math

import pytest
import torch

from lofi import physics


def test_eta_limits_match_the_two_noise_regimes() -> None:
    a = 0.2
    assert physics.eta(a, 0.0) == pytest.approx(a**0.5)
    assert physics.eta(a, 1e12) == pytest.approx(a**2, rel=1e-6)
    assert physics.eta(1.0, 3.0) == pytest.approx(1.0)


def test_snr_scaling_exponents() -> None:
    a = 0.1
    assert physics.snr_ratio(a, 1e12) == pytest.approx(a, rel=1e-6)
    assert physics.snr_ratio(a, 0.0) == pytest.approx(a**1.75)


def test_bandwidth_lowers_noise_as_a_square_root() -> None:
    assert physics.snr_ratio(0.5, 2.0, 0.25) == pytest.approx(2 * physics.snr_ratio(0.5, 2.0))


def test_correlation_falls_because_coil_noise_is_uncorrelated() -> None:
    sigma = torch.tensor([[1.0, 0.4], [0.4, 1.0]], dtype=torch.complex128)
    a, lam = 0.2, 5.0
    low = physics.scale_covariance(sigma, a, lam)
    assert low[0, 0].real == pytest.approx(physics.eta(a, lam))
    assert low[0, 1].real == pytest.approx(0.4 * a**2)
    rho = (low[0, 1] / low[0, 0]).real
    assert rho == pytest.approx(0.4 * a**2 / physics.eta(a, lam))


def test_scaling_is_the_identity_at_the_same_field() -> None:
    sigma = torch.tensor([[2.0, 0.5j], [-0.5j, 1.0]], dtype=torch.complex128)
    assert torch.allclose(physics.scale_covariance(sigma, 1.0, 4.0), sigma)


def test_noisy_source_is_refused() -> None:
    sigma = torch.eye(2, dtype=torch.complex128)
    with pytest.raises(ValueError, match="too noisy"):
        physics.complementary_covariance(sigma, 0.5, 5.0, bandwidth_ratio=0.01)


def test_corrupt_kspace_reaches_the_target_covariance() -> None:
    sigma = torch.tensor([[1.0, 0.3], [0.3, 1.5]], dtype=torch.complex128)
    a, lam = 0.25, 4.0
    source = physics.sample_complex(sigma, (400_000,), torch.Generator().manual_seed(0))
    low = physics.corrupt_kspace(source, sigma, a, lam, generator=torch.Generator().manual_seed(1))
    measured = low @ low.conj().T / low.shape[1]
    expected = physics.scale_covariance(sigma, a, lam)
    assert torch.allclose(measured, expected, atol=4e-3)


def test_sample_complex_is_circular() -> None:
    cov = torch.eye(1, dtype=torch.complex128)
    z = physics.sample_complex(cov, (200_000,), torch.Generator().manual_seed(0))[0]
    assert float(z.real.var()) == pytest.approx(0.5, abs=0.01)
    assert float(z.imag.var()) == pytest.approx(0.5, abs=0.01)
    assert float((z * z).mean().abs()) < 0.01
    assert not math.isnan(float(z.abs().mean()))
