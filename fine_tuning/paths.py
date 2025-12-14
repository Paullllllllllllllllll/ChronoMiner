from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def package_root() -> Path:
    return project_root() / "fine_tuning"


def artifacts_root() -> Path:
    return package_root() / "artifacts"


def editable_root() -> Path:
    return artifacts_root() / "editable_txt"


def annotations_root() -> Path:
    return artifacts_root() / "annotations_jsonl"


def datasets_root() -> Path:
    return artifacts_root() / "datasets"
