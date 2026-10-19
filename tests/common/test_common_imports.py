"""Every vendored module in common/ imports with the declared dependencies."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

import chronominer.common

COMMON_DIR = Path(chronominer.common.__file__).resolve().parent
MANIFEST = COMMON_DIR / "MANIFEST.json"


def _modules() -> list[str]:
    files = json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]
    return sorted(
        "chronominer.common." + Path(name).with_suffix("").as_posix()
        for name in files
        if name.endswith(".py") and name != "__init__.py"
    )


@pytest.mark.parametrize("module", _modules())
def test_vendored_module_imports(module: str) -> None:
    importlib.import_module(module)
