import pytest


@pytest.fixture(params=[2, 3], ids=["2d", "3d"])
def ndim(request: pytest.FixtureRequest) -> int:
    return int(request.param)
