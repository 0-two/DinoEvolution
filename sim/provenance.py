"""Facts about the machine and code that produced a result, saved next to the result."""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def git_commit() -> str:
    """Current commit hash, with a ``-dirty`` suffix if tracked files have changed."""
    try:
        run = lambda *args: subprocess.run(  # noqa: E731
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        commit = run("rev-parse", "HEAD")
        return commit + ("-dirty" if run("status", "--porcelain", "--untracked-files=no") else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def machine_info() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "implementation": platform.python_implementation(),
    }


def library_versions() -> dict[str, str]:
    """Versions of the libraries whose behaviour the results depend on."""
    from importlib import metadata

    names = ["numpy", "cma", "neat-python", "gymnasium", "stable-baselines3", "torch", "pyyaml"]
    out = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "not installed"
    return out
