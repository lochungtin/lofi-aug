import dataclasses

import pytest
import torch

from helpers import phantom, quiet
from lofi import masks
from lofi.corrupt import corrupt, draw, estimate_sigma, match_contrast
from lofi.profile import Acquisition, Contrast, Emi, Fov, Ghost, Noise, Physics, Shading

SHAPE_3D, SPACING_3D = (80, 80, 60), (2.0, 2.0, 2.0)


def gen(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def test_draw_fixed_and_uniform() -> None:
    device = torch.device("cpu")
    assert draw(3, gen(), device) == 3.0
    values = [draw([1.0, 2.0], gen(i), device) for i in range(50)]
    assert all(1 <= v <= 2 for v in values)
    assert max(values) - min(values) > 0.5


def test_keeps_shape_and_spacing_on_the_input_grid(ndim: int) -> None:
    shape, spacing = (64,) * ndim, (2.0,) * ndim
    result = corrupt(phantom(shape, spacing), spacing, quiet(), gen())
    assert result.image.shape == shape
    assert result.spacing == spacing
    assert result.fov.shape == shape
    assert result.fov.dtype == torch.bool


def test_output_spacing_changes_the_grid() -> None:
    spacing = (1.0, 1.0, 1.5)
    image = phantom((120, 120, 60), spacing)
    profile = dataclasses.replace(quiet(), output_spacing_mm=2.0)
    result = corrupt(image, spacing, profile, gen())
    assert result.image.shape == (60, 60, 45)
    assert result.spacing == (2.0, 2.0, 2.0)


def test_same_seed_same_image_different_seed_different_image() -> None:
    image = phantom(SHAPE_3D, SPACING_3D)
    profile = quiet(noise=Noise(sigma=0.05), ghost=Ghost(rms=0.1))
    one = corrupt(image, SPACING_3D, profile, gen(1)).image
    two = corrupt(image, SPACING_3D, profile, gen(1)).image
    three = corrupt(image, SPACING_3D, profile, gen(2)).image
    assert torch.equal(one, two)
    assert not torch.equal(one, three)


def test_neutral_stages_only_blur_and_do_not_move_anatomy() -> None:
    image = phantom(SHAPE_3D, SPACING_3D)
    out = corrupt(image, SPACING_3D, quiet(), gen()).image
    tissue = out > 0.5 * out.max()
    coords = torch.meshgrid(*[torch.arange(n) for n in SHAPE_3D], indexing="ij")
    for c, n in zip(coords, SHAPE_3D, strict=True):
        centroid = float(c[tissue].float().mean())
        assert centroid == pytest.approx((n - 1) / 2, abs=0.3)
    assert float(masks.quantile(out[tissue], 0.5)) == pytest.approx(1.0, abs=0.3)


def test_acquired_resolution_cuts_in_plane_detail_and_slices_cut_through_plane() -> None:
    gen_ = gen()
    image = torch.rand(SHAPE_3D, generator=gen_) * 50 + phantom(SHAPE_3D, SPACING_3D)
    out = corrupt(image, SPACING_3D, quiet(), gen_).image
    spectrum = torch.fft.fftn(out - out.mean()).abs() ** 2
    x = spectrum.sum((1, 2))
    high = x[30:40].sum() / x.sum()  # above the 0.125 cycles/mm cut-off of 4 mm voxels
    assert float(high) < 1e-3


def test_background_noise_is_rician_and_keeps_its_level() -> None:
    sigma = 0.08
    profile = quiet(
        noise=Noise(sigma=sigma),
        acquisition=Acquisition(
            in_plane_mm=2.0, slice_spacing_mm=2.0, slice_thickness_mm=2.0, n_averages=1, window=0.0
        ),
    )
    out = corrupt(phantom(SHAPE_3D, SPACING_3D), SPACING_3D, profile, gen()).image
    corner = out[:8, :8]
    assert float(corner.min()) >= -0.1
    assert float(corner.mean()) == pytest.approx(1.2533 * sigma, rel=0.1)
    assert float(corner.std() / corner.mean()) == pytest.approx(0.5227, rel=0.15)


def test_averaging_two_acquisitions_narrows_the_noise_but_keeps_the_floor() -> None:
    def corner(n: int) -> torch.Tensor:
        acquisition = Acquisition(
            in_plane_mm=2.0, slice_spacing_mm=2.0, slice_thickness_mm=2.0, n_averages=n, window=0.0
        )
        profile = quiet(noise=Noise(sigma=0.08), acquisition=acquisition)
        return corrupt(phantom(SHAPE_3D, SPACING_3D), SPACING_3D, profile, gen()).image[:8, :8]

    one, two = corner(1), corner(2)
    assert float(two.mean()) == pytest.approx(float(one.mean()), rel=0.05)
    assert float(two.std()) == pytest.approx(float(one.std()) / 2**0.5, rel=0.15)


def test_noise_is_grainy_at_the_acquired_resolution() -> None:
    profile = quiet(noise=Noise(sigma=0.1))
    out = corrupt(phantom(SHAPE_3D, SPACING_3D), SPACING_3D, profile, gen()).image
    corner = out[:10, :10]
    d = corner - corner.mean()
    assert float((d * d.roll(1, 0)).mean() / d.var()) > 0.4


def test_slices_leave_piecewise_linear_profiles() -> None:
    image = phantom(SHAPE_3D, SPACING_3D)
    out = corrupt(image, SPACING_3D, quiet(), gen()).image
    line = out[40, 40]
    second = line.diff(n=2).abs()
    assert int((second > 1e-3 * second.max()).sum()) < 0.5 * second.numel()


def test_source_noise_is_not_counted_twice() -> None:
    spacing = (2.0, 2.0, 2.0)
    clean = phantom(SHAPE_3D, spacing)
    noisy = phantom(SHAPE_3D, spacing, noise=8.0)  # 5 % of the median: fully covers the target
    profile = quiet(noise=Noise(sigma=0.002))
    assert float(estimate_sigma(noisy, spacing)) == pytest.approx(8.0, rel=0.1)
    floor_clean = corrupt(clean, spacing, profile, gen()).image[:8, :8].mean()
    floor_noisy = corrupt(noisy, spacing, profile, gen()).image[:8, :8].mean()
    assert float(floor_noisy) > float(floor_clean)


def test_physics_mode_scales_source_noise_by_the_snr_ratio() -> None:
    noisy = phantom(SHAPE_3D, SPACING_3D, noise=1.0)
    near = Noise(physics=Physics(source_b0=1.5, target_b0=1.0, lam=5.0))
    far = Noise(physics=Physics(source_b0=1.5, target_b0=0.3, lam=5.0))
    floor = [
        float(corrupt(noisy, SPACING_3D, quiet(noise=n), gen()).image[:8, :8].mean())
        for n in (near, far)
    ]
    assert floor[1] > 2 * floor[0]


def test_ghosting_only_spreads_along_the_phase_encode_axis() -> None:
    acquisition = Acquisition(n_averages=1, pe_axis=1)
    profile = quiet(acquisition=acquisition, ghost=Ghost(rms=0.3))
    image = torch.zeros(SHAPE_3D)
    image[35:45, 35:45, 20:40] = 100
    out = corrupt(image, SPACING_3D, profile, gen()).image
    along_pe = float(out[38:42, :15].mean())
    along_read = float(out[:15, 38:42].mean())
    assert along_pe > 3 * along_read


def test_emi_adds_a_narrow_line() -> None:
    profile = quiet(emi=Emi(prob=1.0, amplitude=2.0))
    plain = corrupt(phantom(SHAPE_3D, SPACING_3D), SPACING_3D, quiet(), gen()).image
    zipper = corrupt(phantom(SHAPE_3D, SPACING_3D), SPACING_3D, profile, gen()).image
    change = (zipper - plain).abs().sum((1, 2)) + (zipper - plain).abs().sum((0, 2))
    assert float(change.max()) > 5 * float(change.median())


def test_shading_is_smooth_and_changes_tissue_brightness() -> None:
    image = phantom(SHAPE_3D, SPACING_3D)
    profile = quiet(shading=Shading(log_rms=0.5, scale_mm=30.0))
    out = corrupt(image, SPACING_3D, profile, gen()).image
    tissue = masks.erode(image > 0, 4) & (image < 150)
    assert float(out[tissue].std() / out[tissue].mean()) > 0.15


def test_fov_is_exact_zero_outside_and_warped() -> None:
    profile = quiet(fov=Fov(size_mm=110.0, warp_mm=4.0, fade_mm=10.0))
    result = corrupt(phantom(SHAPE_3D, SPACING_3D), SPACING_3D, profile, gen())
    assert torch.all(result.image[~result.fov] == 0)
    assert 0.2 < float((~result.fov).float().mean()) < 0.6
    assert result.fov[..., 0].float().sum(0).unique().numel() > 2


def test_contrast_maps_the_histogram_and_keeps_order() -> None:
    x = torch.linspace(1, 100, 1000)
    reference = x > 20
    target = [0.5, 1.0, 3.0]
    out = match_contrast(x, reference, target)
    assert float(masks.quantile(out[reference], 0.5)) == pytest.approx(1.0, abs=0.02)
    assert float(out[reference].max()) == pytest.approx(3.0, abs=0.02)
    assert torch.all(out.diff() >= -1e-5)
    assert torch.equal(match_contrast(x, reference, None), x)


def test_contrast_in_a_profile_is_applied() -> None:
    profile = quiet(contrast=Contrast(values=[0.2, 1.0, 1.1]))
    out = corrupt(phantom(SHAPE_3D, SPACING_3D), SPACING_3D, profile, gen()).image
    assert float(out.max()) < 1.4


def test_rejects_other_dimensions() -> None:
    with pytest.raises(ValueError, match="2D or 3D"):
        corrupt(torch.zeros(4), (1.0,), quiet())


def test_slice_axis_can_be_any_axis() -> None:
    acquisition = Acquisition(slice_axis=0, pe_axis=1, n_averages=1)
    out = corrupt(phantom(SHAPE_3D, SPACING_3D), SPACING_3D, quiet(acquisition=acquisition), gen())
    assert out.image.shape == SHAPE_3D
    second = out.image[:, 40, 30].diff(n=2).abs()
    assert int((second > 1e-3 * second.max()).sum()) < 0.75 * second.numel()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_runs_on_the_gpu() -> None:
    image = phantom(SHAPE_3D, SPACING_3D).cuda()
    result = corrupt(image, SPACING_3D, quiet(noise=Noise(sigma=0.05)), torch.Generator("cuda"))
    assert result.image.device.type == "cuda"


def test_contrast_weight_leaves_the_background_alone() -> None:
    x = torch.cat([torch.full((500,), 0.02), torch.linspace(0.03, 1.0, 500)])
    reference = x > 0.025
    weight = reference.float()
    out = match_contrast(x, reference, [0.5, 1.0, 2.0], weight)
    assert torch.equal(out[:500], x[:500])
    assert float(out[500:].min()) >= 0.49


def test_slice_jitter_moves_the_kinks_between_acquisitions() -> None:
    def line(jitter: float) -> torch.Tensor:
        z = torch.arange(SHAPE_3D[2], dtype=torch.float32)
        image = (2 + torch.sin(2 * torch.pi * z / 40)).expand(SHAPE_3D)
        acquisition = Acquisition(slice_jitter_mm=jitter, n_averages=2, window=0.0)
        return corrupt(image, SPACING_3D, quiet(acquisition=acquisition), gen()).image[40, 40, 6:-6]

    aligned = line(0.0).diff(n=2).abs()
    jittered = line(2.5).diff(n=2).abs()
    assert float((aligned > 1e-4).float().mean()) < 0.6
    assert float((jittered > 1e-4).float().mean()) > float((aligned > 1e-4).float().mean())
