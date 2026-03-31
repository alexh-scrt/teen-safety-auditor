"""tests/test_package.py — Smoke tests confirming the package scaffold is correct."""

import importlib
import sys

import pytest


class TestPackageImport:
    """Verify that the top-level package is importable and exposes expected attributes."""

    def test_package_is_importable(self) -> None:
        """The teen_safety_auditor package must import without errors."""
        mod = importlib.import_module("teen_safety_auditor")
        assert mod is not None

    def test_version_attribute_exists(self) -> None:
        """__version__ must be a non-empty string."""
        import teen_safety_auditor

        assert isinstance(teen_safety_auditor.__version__, str)
        assert len(teen_safety_auditor.__version__) > 0

    def test_version_format(self) -> None:
        """__version__ should follow basic semver (MAJOR.MINOR.PATCH)."""
        import teen_safety_auditor

        parts = teen_safety_auditor.__version__.split(".")
        assert len(parts) >= 3, "Version should have at least three components"
        for part in parts:
            assert part.isdigit() or part[0].isdigit(), (
                f"Version component '{part}' should start with a digit"
            )

    def test_author_attribute(self) -> None:
        """__author__ must be a non-empty string."""
        import teen_safety_auditor

        assert isinstance(teen_safety_auditor.__author__, str)
        assert len(teen_safety_auditor.__author__) > 0

    def test_license_attribute(self) -> None:
        """__license__ must be present and set to MIT."""
        import teen_safety_auditor

        assert teen_safety_auditor.__license__ == "MIT"

    def test_description_attribute(self) -> None:
        """__description__ must be a non-empty string."""
        import teen_safety_auditor

        assert isinstance(teen_safety_auditor.__description__, str)
        assert len(teen_safety_auditor.__description__) > 0

    def test_all_list_is_defined(self) -> None:
        """__all__ must be a list."""
        import teen_safety_auditor

        assert isinstance(teen_safety_auditor.__all__, list)

    def test_all_contains_version(self) -> None:
        """__all__ must include '__version__'."""
        import teen_safety_auditor

        assert "__version__" in teen_safety_auditor.__all__

    def test_package_in_sys_modules(self) -> None:
        """After import, package must appear in sys.modules."""
        import teen_safety_auditor  # noqa: F401

        assert "teen_safety_auditor" in sys.modules
