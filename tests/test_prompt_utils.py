from __future__ import annotations

import pytest

from modules.llm.prompt_utils import render_prompt_with_schema


@pytest.mark.unit
def test_render_prompt_with_schema_replaces_placeholders():
    prompt = "Name={{SCHEMA_NAME}} Schema={{TRANSCRIPTION_SCHEMA}}"
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}

    rendered = render_prompt_with_schema(prompt, schema, schema_name="S", inject_schema=True)

    assert "Name=S" in rendered
    assert "\"properties\"" in rendered


@pytest.mark.unit
def test_render_prompt_with_schema_removes_schema_when_disabled():
    prompt = "Schema={{TRANSCRIPTION_SCHEMA}}"
    rendered = render_prompt_with_schema(prompt, {"type": "object"}, inject_schema=False)
    assert rendered == "Schema="
