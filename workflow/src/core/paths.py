"""Path helpers for active runtime code.

Keeps project-root resolution in one place so Streamlit and future app
entrypoints do not depend on fragile relative path math.
"""

from pathlib import Path


def project_root() -> Path:
    # src/core/paths.py -> src/core -> src -> Proj1
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return project_root() / "data"


def reports_dir() -> Path:
    return project_root() / "reports"
