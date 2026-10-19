"""Prompt templating: schema token or marker replacement and context injection.

``render_prompt_with_schema`` fills ``{{TRANSCRIPTION_SCHEMA}}`` with compact
JSON, replaces the JSON after ``The JSON schema:`` or appends the schema. The
injection function fills ``{{CONTEXT}}`` or drops its labeled
section when the context is blank; ``prepare_prompt_with_context`` applies
both.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chronominer.llm.prompt_utils import (
    inject_additional_context,
    load_prompt_template,
    prepare_prompt_with_context,
    render_prompt_with_schema,
)
from chronominer.llm.prompts import (
    BOUNDARY_PROMPT_PATH,
    IMAGE_PROMPT_PATH,
    TEXT_PROMPT_PATH,
)


class TestRenderPromptWithSchema:
    """Tests for render_prompt_with_schema function."""

    @pytest.mark.unit
    def test_replaces_token(self) -> None:
        """The token takes the schema as compact JSON."""
        prompt = "Use this schema: {{TRANSCRIPTION_SCHEMA}}"
        schema = {"type": "object", "properties": {"text": {"type": "string"}}}

        result = render_prompt_with_schema(prompt, schema)

        assert "{{TRANSCRIPTION_SCHEMA}}" not in result
        assert '{"type":"object","properties":{"text":{"type":"string"}}}' in result

    @pytest.mark.unit
    def test_replaces_marker_with_json(self) -> None:
        """Test replacement of JSON after 'The JSON schema:' marker."""
        prompt = 'Use this. The JSON schema:\n{"old": "schema"}'
        schema = {"new": "schema"}

        result = render_prompt_with_schema(prompt, schema)

        assert '{"new":"schema"}' in result
        assert '"old": "schema"' not in result

    @pytest.mark.unit
    def test_marker_replacement_keeps_later_braces(self) -> None:
        """Only the balanced JSON block after the marker is replaced."""
        prompt = 'The JSON schema:\n{"old": {"x": 1}}\n\nKeep {this}.'

        result = render_prompt_with_schema(prompt, {"new": 1})

        assert result == 'The JSON schema:\n{"new":1}\n\nKeep {this}.'

    @pytest.mark.unit
    def test_marker_block_ends_at_its_own_brace(self) -> None:
        """A closing brace in text after the schema block stays in place."""
        prompt = (
            'The JSON schema:\n{"old": {"nested": 1}}\n'
            "Write sums as {a+b}; never emit a lone }.\n"
            "{{ADDITIONAL_CONTEXT}}"
        )

        result = render_prompt_with_schema(prompt, {"new": "schema"})

        assert result.startswith('The JSON schema:\n{"new":"schema"}\n')
        assert result.endswith(
            "Write sums as {a+b}; never emit a lone }.\n{{ADDITIONAL_CONTEXT}}"
        )
        assert '"old"' not in result

    @pytest.mark.unit
    def test_marker_without_a_block_gets_the_schema_appended(self) -> None:
        result = render_prompt_with_schema("Instructions. The JSON schema:", {"a": 1})

        assert result == 'Instructions. The JSON schema:\n{"a":1}'

    @pytest.mark.unit
    def test_appends_when_no_marker(self) -> None:
        """Test that schema is appended when no token or marker exists."""
        prompt = "Extract the entries from this text."
        schema = {"type": "object"}

        result = render_prompt_with_schema(prompt, schema)

        assert result == prompt + '\n\nThe JSON schema:\n{"type":"object"}'


class TestInjectAdditionalContext:
    """Tests for inject_additional_context function."""

    @pytest.mark.unit
    def test_replaces_marker(self) -> None:
        """Test replacement of {{CONTEXT}} marker."""
        prompt = "Consider: {{CONTEXT}}"
        context = "This is a price list from 1850."

        result = inject_additional_context(prompt, context)

        assert "{{CONTEXT}}" not in result
        assert "price list from 1850" in result

    @pytest.mark.unit
    def test_no_marker_unchanged(self) -> None:
        """Test that prompt without marker is unchanged."""
        prompt = "No marker in this prompt."

        assert inject_additional_context(prompt, "Some context") == prompt

    @pytest.mark.unit
    @pytest.mark.parametrize("context", ["", None, "   \n\t  "])
    def test_blank_context_removes_section(self, context: str | None) -> None:
        """A blank or missing context removes the labeled context section."""
        prompt = "Intro.\nRules for this corpus and work:\n{{CONTEXT}}\nOther content."

        result = inject_additional_context(prompt, context)

        assert "{{CONTEXT}}" not in result
        assert "Rules for this corpus and work:" not in result
        assert "Other content." in result

    @pytest.mark.unit
    def test_context_trimmed(self) -> None:
        """Test that context whitespace is trimmed."""
        result = inject_additional_context("Context: {{CONTEXT}}", "  trimmed  ")

        assert result == "Context: trimmed"

    @pytest.mark.unit
    def test_multiple_markers_all_replaced(self) -> None:
        """Test that multiple markers are all replaced."""
        prompt = "First: {{CONTEXT}} Second: {{CONTEXT}}"

        result = inject_additional_context(prompt, "context value")

        assert result.count("context value") == 2
        assert "{{CONTEXT}}" not in result


class TestPreparePromptWithContext:
    """Tests for prepare_prompt_with_context function."""

    @pytest.mark.unit
    def test_combines_schema_and_context(self) -> None:
        """Test that schema and context are both applied."""
        prompt = "{{TRANSCRIPTION_SCHEMA}}\n\nContext:\n{{CONTEXT}}"

        result = prepare_prompt_with_context(prompt, {"type": "object"}, "From 1850.")

        assert result == '{"type":"object"}\n\nContext:\nFrom 1850.'

    @pytest.mark.unit
    def test_schema_only(self) -> None:
        """Test with schema but no context."""
        prompt = "{{TRANSCRIPTION_SCHEMA}}\n\nContext:\n{{CONTEXT}}"

        result = prepare_prompt_with_context(prompt, {"type": "object"}, None)

        assert '"type":"object"' in result
        assert "Context:" not in result

    @pytest.mark.unit
    def test_context_only(self) -> None:
        """Test with context but no schema."""
        prompt = "Instructions. {{CONTEXT}}"

        result = prepare_prompt_with_context(prompt, None, "Some context")

        assert result == "Instructions. Some context"

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "path", [TEXT_PROMPT_PATH, IMAGE_PROMPT_PATH, BOUNDARY_PROMPT_PATH]
    )
    def test_shipped_templates_have_no_orphan_context_label(self, path: Path) -> None:
        template = load_prompt_template(path)
        label = template.splitlines()[-2]

        empty = prepare_prompt_with_context(template, {"type": "object"}, None)
        given = prepare_prompt_with_context(template, {"type": "object"}, "ctx")

        assert label.endswith(":")
        assert label not in empty
        assert "{{" not in empty.replace("{{BOUNDARY_TYPE}}", "")
        assert given.endswith(f"{label}\nctx")


class TestLoadPromptTemplate:
    """Tests for load_prompt_template function."""

    @pytest.mark.unit
    def test_strips_outer_whitespace(self, tmp_path: Path) -> None:
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("\n\n  Line 1\n\nLine 2  \n", encoding="utf-8")

        assert load_prompt_template(prompt_file) == "Line 1\n\nLine 2"

    @pytest.mark.unit
    def test_raises_for_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_prompt_template(tmp_path / "nonexistent.txt")
