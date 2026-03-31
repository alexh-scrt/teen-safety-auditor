"""teen_safety_auditor — Developer tool for auditing AI prompts against teen safety policies.

This package exposes the public API surface used by the FastAPI application and
can also be imported programmatically by other tools.

Example usage::

    from teen_safety_auditor import __version__, create_app
    print(__version__)  # "0.1.0"
    app = create_app()  # Returns the configured FastAPI instance
"""

from importlib.metadata import PackageNotFoundError, version

# ---------------------------------------------------------------------------
# Package version — read from installed metadata when available, fall back to
# the hard-coded string so the package works when run directly from source
# without being installed.
# ---------------------------------------------------------------------------
try:
    __version__: str = version("teen_safety_auditor")
except PackageNotFoundError:
    __version__ = "0.1.0"

__author__: str = "Teen Safety Auditor Contributors"
__license__: str = "MIT"
__description__: str = (
    "A developer-focused web tool to audit AI prompts and conversation flows "
    "against OpenAI teen safety policy guidelines."
)

# ---------------------------------------------------------------------------
# Lazy imports — these are defined in later phases but declared here so that
# callers can do `from teen_safety_auditor import create_app` once phase 5 is
# in place.  We guard with try/except so that importing the package during
# phase 1 (before other modules exist) does not raise ImportError.
# ---------------------------------------------------------------------------
try:
    from teen_safety_auditor.main import create_app  # noqa: F401
except ImportError:  # pragma: no cover — main.py not yet generated in phase 1
    pass

try:
    from teen_safety_auditor.auditor import AuditEngine  # noqa: F401
except ImportError:  # pragma: no cover
    pass

__all__: list[str] = [
    "__version__",
    "__author__",
    "__license__",
    "__description__",
    "create_app",
    "AuditEngine",
]
