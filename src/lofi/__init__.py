"""LOFI: make good MRI scans look like low-field open-magnet scans."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("lofi-aug")
except PackageNotFoundError:  # pragma: no cover - running from a source tree without install
    __version__ = "0.0.0"

__all__ = ["__version__"]
