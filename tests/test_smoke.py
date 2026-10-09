"""Smoke test for lekka_ai package."""

from lekka_ai import __version__


def test_import_and_version() -> None:
    """Verify that lekka_ai imports correctly and defines __version__."""
    assert __version__ == "0.1.0"
