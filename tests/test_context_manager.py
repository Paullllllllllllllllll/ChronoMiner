from __future__ import annotations

from pathlib import Path

import pytest

from modules.core.context_manager import ContextManager


@pytest.mark.unit
def test_context_manager_loads_and_returns_context(tmp_path: Path):
    ctx_dir = tmp_path / "additional_context"
    ctx_dir.mkdir()
    (ctx_dir / "MySchema.txt").write_text("context", encoding="utf-8")

    mgr = ContextManager(additional_context_dir=ctx_dir)
    mgr.load_additional_context()

    assert mgr.get_additional_context("MySchema") == "context"
    assert mgr.get_additional_context("Missing") is None


@pytest.mark.unit
def test_context_manager_creates_missing_directory(tmp_path: Path):
    ctx_dir = tmp_path / "missing"
    assert not ctx_dir.exists()

    mgr = ContextManager(additional_context_dir=ctx_dir)
    mgr.load_additional_context()

    assert ctx_dir.exists()
