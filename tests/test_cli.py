import json

import pytest
import torch

from helpers import phantom, quiet
from lofi import cli, io, profile

SPACING = (2.0, 2.0, 2.0)


@pytest.fixture
def scan(tmp_path):
    path = tmp_path / "scan.nii.gz"
    image = phantom((48, 48, 36), SPACING, noise=2.0)
    io.write(path, image, SPACING)
    label = tmp_path / "label.nii.gz"
    io.write(label, (image > 30).short(), SPACING)
    custom = tmp_path / "profile.yaml"
    profile.save(quiet(), custom)
    return path, label, custom


def test_profiles(capsys) -> None:
    cli.main(["profiles"])
    assert "low_field_breast" in capsys.readouterr().out


def test_io_round_trip(scan) -> None:
    image, spacing = io.read(scan[0])
    assert image.shape == (48, 48, 36)
    assert spacing == SPACING


def test_corrupt_writes_scans_and_labels(scan, tmp_path) -> None:
    out = tmp_path / "out"
    cli.main(
        [
            "corrupt",
            str(scan[0]),
            "--label",
            str(scan[1]),
            "--profile",
            str(scan[2]),
            "--n",
            "2",
            "--out",
            str(out),
            "--device",
            "cpu",
        ]
    )
    assert len(list(out.glob("*_label.nii.gz"))) == 2
    assert len(list(out.glob("*.nii.gz"))) == 4
    corrupted = io.read(out / "scan_lofi00.nii.gz")[0]
    assert not torch.equal(corrupted, io.read(scan[0])[0])


def test_measure_prints_json(scan, capsys) -> None:
    cli.main(["measure", str(scan[0]), "--device", "cpu"])
    assert "median" in json.loads(capsys.readouterr().out)


def test_calibrate_writes_a_profile(scan, tmp_path) -> None:
    out = tmp_path / "fitted.yaml"
    cli.main(
        [
            "calibrate",
            "--target",
            str(scan[0]),
            "--sources",
            str(scan[0]),
            "--start",
            str(scan[2]),
            "--iterations",
            "1",
            "--out",
            str(out),
            "--device",
            "cpu",
        ]
    )
    assert profile.load(out).contrast.values is not None
