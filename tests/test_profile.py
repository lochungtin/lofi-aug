import pytest

from lofi import profile
from lofi.profile import Noise, Physics


def test_bundled_profile_loads() -> None:
    assert "low_field_breast" in profile.bundled()
    assert profile.load("low_field_breast").name == "low_field_breast"


def test_unknown_keys_are_rejected_everywhere() -> None:
    with pytest.raises(ValueError, match="noise"):
        profile.from_dict({"name": "x", "noise": {"sigmaa": 1}})
    with pytest.raises(ValueError, match="profile"):
        profile.from_dict({"name": "x", "extra": 1})
    with pytest.raises(ValueError, match=r"noise\.physics"):
        profile.from_dict({"name": "x", "noise": {"physics": {"lamda": 1}}})


def test_pe_axis_must_lie_in_the_slice_plane() -> None:
    with pytest.raises(ValueError, match="slice plane"):
        profile.from_dict({"name": "x", "acquisition": {"pe_axis": 2, "slice_axis": 2}})


def test_round_trips_through_yaml(tmp_path) -> None:
    original = profile.from_dict(
        {
            "name": "x",
            "noise": {"physics": {"source_b0": 1.5, "target_b0": 0.5, "lam": 5.0}},
            "ghost": {"rms": [0.1, 0.2]},
        }
    )
    path = tmp_path / "x.yaml"
    profile.save(original, path)
    loaded = profile.load(path)
    assert loaded == original
    assert isinstance(loaded.noise, Noise)
    assert isinstance(loaded.noise.physics, Physics)
