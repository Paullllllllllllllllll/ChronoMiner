from __future__ import annotations

from pathlib import Path

import pytest

from modules.llm.prompt_utils import load_prompt_template, render_prompt_with_schema


class TestRenderPromptWithSchema:
    """Tests for render_prompt_with_schema function."""

    @pytest.mark.unit
    def test_replaces_placeholders(self):
        """Test that schema placeholders are replaced."""
        prompt = "Name={{SCHEMA_NAME}} Schema={{TRANSCRIPTION_SCHEMA}}"
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}

        rendered = render_prompt_with_schema(
            prompt, schema, schema_name="S", inject_schema=True
        )

        assert "Name=S" in rendered
        assert '"properties"' in rendered

    @pytest.mark.unit
    def test_removes_schema_when_disabled(self):
        """Test schema removal when inject_schema is False."""
        prompt = "Schema={{TRANSCRIPTION_SCHEMA}}"
        rendered = render_prompt_with_schema(
            prompt, {"type": "object"}, inject_schema=False
        )
        assert rendered == "Schema="

    @pytest.mark.unit
    def test_removes_schema_section_header_when_disabled(self):
        """No orphaned 'The JSON schema:' header without a schema after it."""
        prompt = (
            "OUTPUT REQUIREMENTS\n- Return only valid JSON\n\n"
            "The JSON schema:\n{{TRANSCRIPTION_SCHEMA}}\n\nContext:\n{{CONTEXT}}"
        )
        rendered = render_prompt_with_schema(
            prompt, {"type": "object"}, inject_schema=False, context="ctx"
        )

        assert "{{TRANSCRIPTION_SCHEMA}}" not in rendered
        assert "The JSON schema:" not in rendered
        assert "Context:" in rendered
        assert "ctx" in rendered

    @pytest.mark.unit
    def test_removes_schema_section_header_in_shipped_templates(self):
        """The bundled templates must render cleanly with schema injection off."""
        from modules.llm.prompt_utils import prompt_path

        for name in ("text_extraction_prompt.txt", "image_extraction_prompt.txt"):
            template = load_prompt_template(prompt_path(name))
            rendered = render_prompt_with_schema(
                template, {"type": "object"}, inject_schema=False, context=None
            )
            assert "{{TRANSCRIPTION_SCHEMA}}" not in rendered
            assert "The JSON schema:" not in rendered
            assert "OUTPUT REQUIREMENTS" in rendered

    @pytest.mark.unit
    def test_schema_section_kept_when_injecting(self):
        prompt = "The JSON schema:\n{{TRANSCRIPTION_SCHEMA}}\n"
        rendered = render_prompt_with_schema(
            prompt, {"type": "object"}, inject_schema=True
        )
        assert "The JSON schema:" in rendered
        assert '"type":"object"' in rendered

    @pytest.mark.unit
    def test_injects_context(self):
        """Test context injection."""
        prompt = "Prompt text\nContext:\n{{CONTEXT}}\nMore text"
        rendered = render_prompt_with_schema(
            prompt, {}, inject_schema=False, context="This is context information"
        )

        assert "This is context information" in rendered
        assert "{{CONTEXT}}" not in rendered

    @pytest.mark.unit
    def test_removes_context_section_when_empty(self):
        """Test context section removal when no context provided."""
        prompt = "Prompt text\nContext:\n{{CONTEXT}}\nMore text"
        rendered = render_prompt_with_schema(
            prompt, {}, inject_schema=False, context=None
        )

        assert "{{CONTEXT}}" not in rendered
        assert "More text" in rendered

    @pytest.mark.unit
    def test_handles_empty_schema_name(self):
        """Test handling of empty schema name."""
        prompt = "Schema: {{SCHEMA_NAME}}"
        rendered = render_prompt_with_schema(
            prompt, {}, schema_name=None, inject_schema=False
        )

        assert rendered == "Schema: "

    @pytest.mark.unit
    def test_appends_schema_when_no_placeholder(self):
        """Test schema appending when no placeholder exists."""
        prompt = "Simple prompt without placeholder"
        schema = {"type": "object"}

        rendered = render_prompt_with_schema(prompt, schema, inject_schema=True)

        assert "Simple prompt without placeholder" in rendered
        assert '"type":"object"' in rendered

    @pytest.mark.unit
    def test_handles_complex_schema(self):
        """Test handling of complex nested schema."""
        prompt = "Schema={{TRANSCRIPTION_SCHEMA}}"
        schema = {
            "type": "object",
            "properties": {
                "entries": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "value": {"type": "number"},
                        },
                    },
                }
            },
        }

        rendered = render_prompt_with_schema(prompt, schema, inject_schema=True)

        assert "entries" in rendered
        assert "items" in rendered

    @pytest.mark.unit
    def test_handles_whitespace_only_context(self):
        """Test that whitespace-only context is treated as empty."""
        prompt = "Context:\n{{CONTEXT}}\nEnd"
        rendered = render_prompt_with_schema(
            prompt, {}, inject_schema=False, context="   \n\t  "
        )

        assert "{{CONTEXT}}" not in rendered


class TestLoadPromptTemplate:
    """Tests for load_prompt_template function."""

    @pytest.mark.unit
    def test_loads_existing_file(self, tmp_path: Path):
        """Test loading an existing prompt file."""
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("  Test prompt content  \n", encoding="utf-8")

        result = load_prompt_template(prompt_file)

        assert result == "Test prompt content"

    @pytest.mark.unit
    def test_raises_for_missing_file(self, tmp_path: Path):
        """Test that FileNotFoundError is raised for missing file."""
        nonexistent = tmp_path / "nonexistent.txt"

        with pytest.raises(FileNotFoundError):
            load_prompt_template(nonexistent)

    @pytest.mark.unit
    def test_strips_whitespace(self, tmp_path: Path):
        """Test that leading/trailing whitespace is stripped."""
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("\n\n  Content  \n\n", encoding="utf-8")

        result = load_prompt_template(prompt_file)

        assert result == "Content"

    @pytest.mark.unit
    def test_preserves_internal_whitespace(self, tmp_path: Path):
        """Test that internal whitespace is preserved."""
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("Line 1\n\nLine 2", encoding="utf-8")

        result = load_prompt_template(prompt_file)

        assert result == "Line 1\n\nLine 2"
