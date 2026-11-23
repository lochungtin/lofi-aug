import lofi


def test_version_is_a_string() -> None:
    assert isinstance(lofi.__version__, str)
