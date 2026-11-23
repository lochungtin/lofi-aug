"""Field-strength scaling of MRI signal and thermal noise.

With ``a = B0_low / B0_high``, signal scales as ``a² f`` (reciprocity) and noise variance as the
circuit resistance times bandwidth. Coil resistance goes as ``ω^½`` (skin effect) and sample
resistance as ``ω²`` (tissue eddy currents). ``lam`` is ``R_s / R_c`` at the source field,
measurable on the bench as ``Q_unloaded / Q_loaded - 1``.

Coil resistance is uncorrelated between channels, whereas sample noise is shared, so the two parts
of a covariance matrix scale differently: `scale_covariance` keeps them apart.
"""

from __future__ import annotations

import torch
from torch import Tensor


def eta(a: float, lam: float) -> float:
    """Noise variance scaling for one channel: ``(a^½ + λ a²) / (1 + λ)``."""
    return float((a**0.5 + lam * a**2) / (1 + lam))


def snr_ratio(a: float, lam: float, bandwidth_ratio: float = 1.0, f: float = 1.0) -> float:
    """SNR at the low field over SNR at the high field, per sample, for the same coil."""
    return float(a**2 * f / (eta(a, lam) * bandwidth_ratio) ** 0.5)


def scale_covariance(
    sigma: Tensor, a: float, lam: float | Tensor, bandwidth_ratio: float = 1.0
) -> Tensor:
    """Scale a channel noise covariance to the low field.

    The diagonal holds coil and sample noise, in the ratio ``1 : λ`` (``lam`` may be one value per
    channel). Off-diagonal terms are sample noise only, so correlation falls by ``a² / η``.
    """
    coil = torch.diag(sigma.diagonal().real / (1 + torch.as_tensor(lam, dtype=sigma.real.dtype)))
    sample = sigma - coil.to(sigma.dtype)
    out: Tensor = bandwidth_ratio * (a**0.5 * coil.to(sigma.dtype) + a**2 * sample)
    return out


def complementary_covariance(
    sigma: Tensor, a: float, lam: float | Tensor, bandwidth_ratio: float = 1.0, f: float = 1.0
) -> Tensor:
    """Noise to add after scaling raw data by ``a² f``, which already scales the source noise.

    Raises ``ValueError`` if the result is not positive semi-definite: the source data are then too
    noisy to simulate that target.
    """
    out = scale_covariance(sigma, a, lam, bandwidth_ratio) - a**4 * f**2 * sigma
    out = (out + out.conj().T) / 2
    floor = -1e-6 * sigma.diagonal().real.abs().max()
    if torch.linalg.eigvalsh(out).min() < floor:
        raise ValueError("source data are too noisy to simulate this target")
    return out


def sample_complex(
    covariance: Tensor, shape: tuple[int, ...], generator: torch.Generator | None = None
) -> Tensor:
    """Draw ``CN(0, covariance)`` noise of shape ``(channels, *shape)``."""
    values, vectors = torch.linalg.eigh(covariance)
    root = vectors * values.clamp_min(0).sqrt().to(vectors.dtype)
    channels = covariance.shape[0]
    real_dtype = covariance.real.dtype
    z = torch.randn(
        (2, channels, *shape), generator=generator, device=covariance.device, dtype=real_dtype
    )
    white = torch.complex(z[0], z[1]) / 2**0.5
    out: Tensor = torch.tensordot(root, white, dims=([1], [0]))
    return out


def corrupt_kspace(
    y: Tensor,
    sigma: Tensor,
    a: float,
    lam: float | Tensor,
    bandwidth_ratio: float = 1.0,
    f: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Raw-data route: ``y_L = a² f y_H + n_add`` for k-space of shape ``(channels, *k)``.

    Crop ``y`` first if the target is coarser. ``f`` must be a scalar here: it is only exact for one
    tissue or for proton-density weighting.
    """
    added = complementary_covariance(sigma, a, lam, bandwidth_ratio, f)
    out: Tensor = a**2 * f * y + sample_complex(added, tuple(y.shape[1:]), generator)
    return out
