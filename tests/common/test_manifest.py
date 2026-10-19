"""common/MANIFEST.json pins every vendored module by its SHA-256.

The other tools carry byte-identical copies of these files; a change here
must be made in every copy and the manifest updated with it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import chronominer.common

COMMON_DIR = Path(chronominer.common.__file__).resolve().parent
MANIFEST = COMMON_DIR / "MANIFEST.json"


def file_hash(path: Path) -> str:
    """Return the SHA-256 of *path* with LF line endings."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def vendored_files() -> list[str]:
    """Return the files of common/ (manifest and caches left out), sorted."""
    return sorted(
        path.relative_to(COMMON_DIR).as_posix()
        for path in COMMON_DIR.rglob("*")
        if path.is_file() and path != MANIFEST and "__pycache__" not in path.parts
    )


def _listed() -> dict[str, str]:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert data["algorithm"] == "sha256"
    files: dict[str, str] = data["files"]
    return files


def test_every_file_in_common_is_listed() -> None:
    assert vendored_files() == sorted(_listed())


def test_every_listed_file_matches_its_hash() -> None:
    drifted = sorted(
        name
        for name, digest in _listed().items()
        if not (COMMON_DIR / name).is_file() or file_hash(COMMON_DIR / name) != digest
    )
    assert drifted == []


def test_the_manifest_is_sorted() -> None:
    assert list(_listed()) == sorted(_listed())
