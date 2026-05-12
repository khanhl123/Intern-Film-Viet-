"""Compatibility shim — re-exports from masalytics.paths for legacy imports."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from masalytics.paths import (
    DATA_DIR,
    OUTPUTS_LOCATIONQUESTIONS_DIR as OUTPUTS_LOCATION_QUESTIONS,
    OUTPUTS_SALESOVERVIEW_DIR as OUTPUTS_SALESOVERVIEW,
    OUTPUTS_TITLES_DISTRIBUTORS_DIR as OUTPUTS_TITLES_DISTRIBUTORS,
    PROJECT_ROOT as BASE_DIR,
    find_database,
)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_database_path(preferred: Optional[Path] = None) -> Path:
    # Old behaviour: a missing `preferred` falls through to glob search rather than erroring.
    if preferred is not None and Path(preferred).is_file():
        return Path(preferred)
    return find_database()


__all__ = [
    "BASE_DIR",
    "DATA_DIR",
    "OUTPUTS_LOCATION_QUESTIONS",
    "OUTPUTS_SALESOVERVIEW",
    "OUTPUTS_TITLES_DISTRIBUTORS",
    "ensure_dir",
    "find_database_path",
]
