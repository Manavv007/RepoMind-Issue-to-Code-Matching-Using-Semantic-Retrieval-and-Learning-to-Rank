"""WMI-free environment description for reports.

``platform.platform()`` / ``platform.uname()`` query WMI on Windows (Python 3.12+),
which can block indefinitely when the WMI service is unhealthy. Reports must never
hang on metadata, so this module uses only calls that do not touch WMI.
"""

from __future__ import annotations

import os
import platform
import sys


def platform_summary() -> str:
    """Short OS description, e.g. ``win32-10.0.19045`` or ``linux-x86_64``."""
    if sys.platform == "win32":
        v = sys.getwindowsversion()
        return f"win32-{v.major}.{v.minor}.{v.build}"
    try:
        return f"{sys.platform}-{os.uname().machine}"
    except AttributeError:  # pragma: no cover
        return sys.platform


def python_version() -> str:
    return platform.python_version()  # pure, no system queries
