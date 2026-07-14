"""Filesystem locations, overridable for shared-cluster and scratch workflows."""

from __future__ import annotations

import os
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1]


def _path(environment: str, default: Path) -> Path:
    value = os.environ.get(environment)
    return Path(value).expanduser().resolve() if value else default


# Editable checkouts keep their historical repo-local defaults. A wheel install
# uses the caller's working directory instead of writing inside site-packages.
DEFAULT_WORK_ROOT = SOURCE_ROOT if (SOURCE_ROOT / "pyproject.toml").is_file() else Path.cwd()
WORK_ROOT = _path("PGLLM_WORK_ROOT", DEFAULT_WORK_ROOT)
DATA_ROOT = _path("PGLLM_DATA_ROOT", WORK_ROOT / "data")
RESULTS_ROOT = _path("PGLLM_RESULTS_ROOT", WORK_ROOT / "results")
BASELINE_RESULTS_ROOT = _path("PGLLM_BASELINE_RESULTS_ROOT", WORK_ROOT / "results_baselines")
