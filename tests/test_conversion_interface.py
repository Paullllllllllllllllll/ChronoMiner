"""Interface-level tests for the ``modules.conversion`` package.

Exercises the public surface (:class:`BaseConverter`, :class:`CSVConverter`,
:class:`DocumentConverter`, :func:`extract_entries_from_json`,
:func:`parse_json_from_text`, :func:`parse_llm_response_text`) without
reaching into private helpers. Ensures the package facade remains stable
through future refactors.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from modules.conversion import (
    BaseConverter,
    CSVConverter,
    DocumentConverter,
    extract_entries_from_json,
    parse_json_from_text,
    parse_llm_response_text,
    resolve_field,
)


@pytest.mark.unit
class TestPublicInterface:
    """The conversion package's public API must be importable as documented."""

    def test_exports_are_all_non_none(self):
        # If any symbol failed to import, the module-level import at the
        # top of this file would have raised.
        for obj in (
            BaseConverter,
            CSVConverter,
            DocumentConverter,
            extract_entries_from_json,
            parse_json_from_text,
            parse_llm_response_text,
            resolve_field,
        ):
            assert obj is not None


@pytest.mark.unit
class TestResolveField:
    """``resolve_field`` is the shared field-lookup helper used by converters."""

    def test_plain_key(self):
        assert resolve_field({"a": 1}, "a") == 1

    def test_missing_returns_default(self):
        assert resolve_field({}, "missing", default="N/A") == "N/A"

    def test_dotted_nested_key(self):
        assert resolve_field({"a": {"b": 5}}, "a.b") == 5

    def test_dotted_missing_returns_default(self):
        assert resolve_field({"a": {}}, "a.b", default="?") == "?"


@pytest.mark.unit
class TestParseJsonFromText:
    """``parse_json_from_text`` recovers JSON from wrapped model output."""

    def test_bare_json(self):
        assert parse_json_from_text('{"x": 1}') == '{"x": 1}'

    def test_fenced_json(self):
        result = parse_json_from_text('```json\n{"x": 1}\n```')
        assert result is not None
        assert json.loads(result) == {"x": 1}

    def test_preamble_before_json(self):
        text = 'Here is the output:\n\n{"entries": [{"a": 1}]}'
        result = parse_json_from_text(text)
        assert result is not None
        parsed = json.loads(result)
        assert parsed == {"entries": [{"a": 1}]}

    def test_empty_returns_none(self):
        assert parse_json_from_text("") is None

    def test_nonjson_returns_none(self):
        assert parse_json_from_text("hello world, no JSON here") is None


@pytest.mark.unit
class TestParseLlmResponseText:
    """``parse_llm_response_text`` extracts text from various API body shapes."""

    def test_output_text_string(self):
        assert parse_llm_response_text({"output_text": "hello"}) == "hello"

    def test_output_text_list(self):
        body = {
            "output_text": [
                {"type": "text", "text": "world"},
                {"type": "other", "text": "ignored"},
            ]
        }
        assert parse_llm_response_text(body) == "world"

    def test_chat_completions_shape(self):
        body = {"choices": [{"message": {"content": "chat answer"}}]}
        assert parse_llm_response_text(body) == "chat answer"

    def test_responses_api_output_shape(self):
        body = {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "piece1"}, {"text": "piece2"}],
                }
            ]
        }
        # ``piece2`` is included even without an explicit type; helper
        # concatenates any dict whose "text" field is a string.
        assert "piece1" in parse_llm_response_text(body)

    def test_non_dict_returns_empty(self):
        assert parse_llm_response_text("not a dict") == ""
        assert parse_llm_response_text(None) == ""


@pytest.mark.unit
class TestExtractEntriesFromJson:
    """``extract_entries_from_json`` handles the ChronoMiner output formats."""

    def _write(self, path: Path, payload: object) -> Path:
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path

    def test_direct_entries(self, tmp_path):
        f = self._write(tmp_path / "out.json", {"entries": [{"a": 1}, {"a": 2}]})
        assert extract_entries_from_json(f) == [{"a": 1}, {"a": 2}]

    def test_records_envelope(self, tmp_path):
        payload = {
            "records": [
                {"response": {"entries": [{"a": 1}]}},
                {"response": {"entries": [{"a": 2}]}},
            ]
        }
        f = self._write(tmp_path / "out.json", payload)
        assert extract_entries_from_json(f) == [{"a": 1}, {"a": 2}]

    def test_contains_no_content_flag(self, tmp_path):
        payload = {"contains_no_content_of_requested_type": True, "entries": [{"a": 1}]}
        f = self._write(tmp_path / "out.json", payload)
        assert extract_entries_from_json(f) == []

    def test_nonexistent_returns_empty(self, tmp_path):
        assert extract_entries_from_json(tmp_path / "missing.json") == []


@pytest.mark.unit
class TestConverterConstruction:
    """``CSVConverter`` and ``DocumentConverter`` accept a schema name."""

    def test_csv_converter_constructs(self):
        c = CSVConverter("BibliographicEntries")
        assert c is not None
        assert isinstance(c, BaseConverter)

    def test_document_converter_constructs(self):
        c = DocumentConverter("BibliographicEntries")
        assert c is not None
        assert isinstance(c, BaseConverter)
