import pytest
import torch

from helpers import phantom, quiet
from lofi.profile import Fov

SPACING = (2.0, 2.0, 2.0)
SHAPE = (48, 48, 36)
PROFILE = quiet(fov=Fov(size_mm=70.0, warp_mm=2.0, fade_mm=4.0))


def test_monai_adapter_shares_one_fov() -> None:
    pytest.importorskip("monai")
    from monai.data import MetaTensor

    from lofi.integrations.monai import LowFieldCorruptd

    affine = torch.diag(torch.tensor([2.0, 2.0, 2.0, 1.0]))
    image = MetaTensor(phantom(SHAPE, SPACING)[None], affine=affine)
    label = MetaTensor(torch.ones(1, *SHAPE), affine=affine)
    out = LowFieldCorruptd(["image", "label"], PROFILE, seed=0)({"image": image, "label": label})
    assert out["image"].shape == (1, *SHAPE)
    dropped = out["label"].as_tensor() == 0
    assert 0 < float(dropped.float().mean()) < 1
    assert torch.all(out["image"].as_tensor()[dropped] == 0)


def test_torchio_adapter_shares_one_fov() -> None:
    tio = pytest.importorskip("torchio")

    from lofi.integrations.torchio import RandomLowField

    affine = torch.diag(torch.tensor([2.0, 2.0, 2.0, 1.0])).numpy()
    subject = tio.Subject(
        image=tio.ScalarImage(tensor=phantom(SHAPE, SPACING)[None], affine=affine),
        label=tio.LabelMap(tensor=torch.ones(1, *SHAPE), affine=affine),
    )
    out = RandomLowField(PROFILE, seed=0)(subject)
    assert out.image.data.shape == (1, *SHAPE)
    dropped = out.label.data == 0
    assert 0 < float(dropped.float().mean()) < 1
    assert torch.all(out.image.data[dropped] == 0)
