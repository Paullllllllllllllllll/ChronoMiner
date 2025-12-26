from __future__ import annotations

from pathlib import Path

import pytest

from modules.core.prompt_context import (
    apply_context_placeholders,
    load_basic_context,
    resolve_additional_context,
)


@pytest.mark.unit
def test_load_basic_context_schema_specific(tmp_path: Path):
    ctx_dir = tmp_path / "basic_context"
    ctx_dir.mkdir()
    (ctx_dir / "TestSchema.txt").write_text("basic", encoding="utf-8")

    text = load_basic_context(str(ctx_dir), schema_name="TestSchema")
    assert text == "basic"


@pytest.mark.unit
def test_apply_context_placeholders():
    template = "A={{BASIC_CONTEXT}} B={{ADDITIONAL_CONTEXT}}"
    rendered = apply_context_placeholders(template, basic_context="x", additional_context="y")
    assert rendered == "A=x B=y"


@pytest.mark.unit
def test_resolve_additional_context_file_specific(tmp_path: Path):
    input_file = tmp_path / "doc.txt"
    input_file.write_text("hi", encoding="utf-8")
    (tmp_path / "doc_context.txt").write_text("filectx", encoding="utf-8")

    ctx = resolve_additional_context(
        "AnySchema",
        context_settings={"use_additional_context": True, "use_default_context": False},
        context_manager=None,
        text_file=input_file,
    )
    assert ctx == "filectx"
