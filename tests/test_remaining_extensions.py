"""Remaining coverage extensions for schema_handlers, json_utils, text_utils,
prompt_utils, and chunking_service uncovered paths."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# ---------------------------------------------------------------------------
# schema_handlers — convert methods & registry
# ---------------------------------------------------------------------------
from modules.operations.extraction.schema_handlers import (
    BaseSchemaHandler,
    get_schema_handler,
    register_schema_handler,
    schema_handlers_registry,
)


class TestBaseSchemaHandler:
    def test_prepare_payload(self):
        handler = BaseSchemaHandler("TestSchema")
        payload = handler.prepare_payload(
            text_chunk="Hello world",
            dev_message="Extract data",
            model_config={"transcription_model": {"name": "gpt-4o"}},
            schema={"name": "TestSchema", "schema": {"type": "object"}},
        )
        assert isinstance(payload, dict)

    def test_process_response_valid(self):
        handler = BaseSchemaHandler("TestSchema")
        response = json.dumps({"entries": [{"id": 1}]})
        result = handler.process_response(response)
        assert isinstance(result, dict)

    def test_convert_to_csv(self, tmp_path):
        handler = BaseSchemaHandler("TestSchema")
        json_file = tmp_path / "data.json"
        json_file.write_text(json.dumps({"entries": []}), encoding="utf-8")
        csv_file = tmp_path / "data.csv"

        # CSVConverter may fail without proper schema mapping, but we test the wiring
        with patch("modules.operations.extraction.schema_handlers.CSVConverter") as mock_csv:
            mock_instance = MagicMock()
            mock_csv.return_value = mock_instance
            handler.convert_to_csv(json_file, csv_file)
            mock_csv.assert_called_once_with("TestSchema")
            mock_instance.convert_to_csv.assert_called_once_with(json_file, csv_file)

    def test_convert_to_docx(self, tmp_path):
        handler = BaseSchemaHandler("TestSchema")
        json_file = tmp_path / "data.json"
        docx_file = tmp_path / "data.docx"

        with patch("modules.operations.extraction.schema_handlers.DocumentConverter") as mock_doc:
            mock_instance = MagicMock()
            mock_doc.return_value = mock_instance
            handler.convert_to_docx(json_file, docx_file)
            mock_doc.assert_called_once_with("TestSchema")
            mock_instance.convert_to_docx.assert_called_once_with(json_file, docx_file)

    def test_convert_to_txt(self, tmp_path):
        handler = BaseSchemaHandler("TestSchema")
        json_file = tmp_path / "data.json"
        txt_file = tmp_path / "data.txt"

        with patch("modules.operations.extraction.schema_handlers.DocumentConverter") as mock_doc:
            mock_instance = MagicMock()
            mock_doc.return_value = mock_instance
            handler.convert_to_txt(json_file, txt_file)
            mock_doc.assert_called_once_with("TestSchema")
            mock_instance.convert_to_txt.assert_called_once_with(json_file, txt_file)


class TestSchemaHandlerRegistry:
    def test_get_registered_handler(self):
        handler = get_schema_handler("BibliographicEntries")
        assert isinstance(handler, BaseSchemaHandler)
        assert handler.schema_name == "BibliographicEntries"

    def test_get_unregistered_returns_default(self):
        handler = get_schema_handler("UnregisteredSchemaXYZ")
        assert isinstance(handler, BaseSchemaHandler)
        assert handler.schema_name == "UnregisteredSchemaXYZ"

    def test_register_and_get(self):
        register_schema_handler("CustomTestSchema", BaseSchemaHandler)
        handler = get_schema_handler("CustomTestSchema")
        assert handler.schema_name == "CustomTestSchema"
        # Clean up
        if "CustomTestSchema" in schema_handlers_registry:
            del schema_handlers_registry["CustomTestSchema"]


# ---------------------------------------------------------------------------
# json_utils — uncovered extraction paths
# ---------------------------------------------------------------------------
from modules.core.json_utils import (
    _extract_text_from_api_body,
    _parse_entries_from_text,
    _extract_entries_from_record,
    extract_entries_from_json,
)


class TestExtractTextFromApiBody:
    def test_non_dict_returns_empty(self):
        assert _extract_text_from_api_body("not a dict") == ""
        assert _extract_text_from_api_body(None) == ""

    def test_output_text_string(self):
        body = {"output_text": "hello"}
        assert _extract_text_from_api_body(body) == "hello"

    def test_output_text_list(self):
        body = {"output_text": [{"type": "text", "text": "hello"}]}
        assert _extract_text_from_api_body(body) == "hello"

    def test_output_text_list_empty(self):
        body = {"output_text": [{"type": "image", "url": "http://x"}]}
        assert _extract_text_from_api_body(body) == ""

    def test_chat_completions_format(self):
        body = {"choices": [{"message": {"content": "hello"}}]}
        assert _extract_text_from_api_body(body) == "hello"

    def test_chat_completions_empty_choices(self):
        body = {"choices": []}
        assert _extract_text_from_api_body(body) == ""

    def test_responses_api_nested(self):
        body = {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "text", "text": "part1"}, {"type": "text", "text": "part2"}],
                }
            ]
        }
        assert _extract_text_from_api_body(body) == "part1part2"

    def test_no_known_format(self):
        body = {"unknown_key": "value"}
        assert _extract_text_from_api_body(body) == ""


class TestParseEntriesFromText:
    def test_empty_string(self):
        assert _parse_entries_from_text("") is None

    def test_invalid_json(self):
        assert _parse_entries_from_text("not json") is None

    def test_non_dict(self):
        assert _parse_entries_from_text("[1,2,3]") is None

    def test_no_content_flag(self):
        assert _parse_entries_from_text(json.dumps({"contains_no_content_of_requested_type": True})) is None

    def test_no_entries_key(self):
        assert _parse_entries_from_text(json.dumps({"data": [1, 2]})) is None

    def test_valid_entries(self):
        result = _parse_entries_from_text(json.dumps({"entries": [{"id": 1}, {"id": 2}]}))
        assert result == [{"id": 1}, {"id": 2}]

    def test_filters_none_entries(self):
        result = _parse_entries_from_text(json.dumps({"entries": [{"id": 1}, None, {"id": 2}]}))
        assert result == [{"id": 1}, {"id": 2}]


class TestExtractEntriesFromRecord:
    def test_non_dict_record(self):
        assert _extract_entries_from_record("string") == []
        assert _extract_entries_from_record(None) == []

    def test_no_response(self):
        assert _extract_entries_from_record({"other": "key"}) == []

    def test_response_is_json_string(self):
        response_str = json.dumps({"entries": [{"id": 1}]})
        result = _extract_entries_from_record({"response": response_str})
        assert result == [{"id": 1}]

    def test_response_dict_with_entries(self):
        result = _extract_entries_from_record({"response": {"entries": [{"id": 1}]}})
        assert result == [{"id": 1}]

    def test_response_dict_with_no_content_flag(self):
        result = _extract_entries_from_record(
            {"response": {"contains_no_content_of_requested_type": True}}
        )
        assert result == []

    def test_response_dict_api_body(self):
        result = _extract_entries_from_record(
            {"response": {"output_text": json.dumps({"entries": [{"id": 1}]})}}
        )
        assert result == [{"id": 1}]

    def test_response_none(self):
        assert _extract_entries_from_record({"response": None}) == []


class TestExtractEntriesFromJson:
    def test_direct_entries(self, tmp_path):
        f = tmp_path / "data.json"
        f.write_text(json.dumps({"entries": [{"id": 1}, {"id": 2}]}), encoding="utf-8")
        result = extract_entries_from_json(f)
        assert len(result) == 2

    def test_no_content_flag(self, tmp_path):
        f = tmp_path / "data.json"
        f.write_text(json.dumps({"contains_no_content_of_requested_type": True}), encoding="utf-8")
        result = extract_entries_from_json(f)
        assert result == []

    def test_records_format(self, tmp_path):
        f = tmp_path / "data.json"
        data = {
            "records": [
                {"response": json.dumps({"entries": [{"id": 1}]})},
                {"response": json.dumps({"entries": [{"id": 2}]})},
            ]
        }
        f.write_text(json.dumps(data), encoding="utf-8")
        result = extract_entries_from_json(f)
        assert len(result) == 2

    def test_responses_format_string(self, tmp_path):
        f = tmp_path / "data.json"
        data = {
            "responses": [
                json.dumps({"entries": [{"id": 1}]}),
            ]
        }
        f.write_text(json.dumps(data), encoding="utf-8")
        result = extract_entries_from_json(f)
        assert len(result) == 1

    def test_responses_format_dict_with_body(self, tmp_path):
        f = tmp_path / "data.json"
        data = {
            "responses": [
                {
                    "body": {
                        "output_text": json.dumps({"entries": [{"id": 1}]}),
                    }
                }
            ]
        }
        f.write_text(json.dumps(data), encoding="utf-8")
        result = extract_entries_from_json(f)
        assert len(result) == 1

    def test_responses_format_dict_with_raw_response(self, tmp_path):
        f = tmp_path / "data.json"
        data = {
            "responses": [
                {
                    "raw_response": {
                        "output_text": json.dumps({"entries": [{"id": 1}]}),
                    }
                }
            ]
        }
        f.write_text(json.dumps(data), encoding="utf-8")
        result = extract_entries_from_json(f)
        assert len(result) == 1

    def test_responses_with_none(self, tmp_path):
        f = tmp_path / "data.json"
        data = {"responses": [None, json.dumps({"entries": [{"id": 1}]})]}
        f.write_text(json.dumps(data), encoding="utf-8")
        result = extract_entries_from_json(f)
        assert len(result) == 1

    def test_list_format_records(self, tmp_path):
        f = tmp_path / "data.json"
        data = [
            {"response": json.dumps({"entries": [{"id": 1}]})},
            {"response": json.dumps({"entries": [{"id": 2}]})},
        ]
        f.write_text(json.dumps(data), encoding="utf-8")
        result = extract_entries_from_json(f)
        assert len(result) == 2

    def test_list_format_direct_entries_fallback(self, tmp_path):
        f = tmp_path / "data.json"
        data = [{"id": 1}, {"id": 2}]
        f.write_text(json.dumps(data), encoding="utf-8")
        result = extract_entries_from_json(f)
        assert len(result) == 2

    def test_nonexistent_file(self, tmp_path):
        result = extract_entries_from_json(tmp_path / "nonexistent.json")
        assert result == []

    def test_filters_none_entries(self, tmp_path):
        f = tmp_path / "data.json"
        f.write_text(json.dumps({"entries": [{"id": 1}, None, {"id": 2}]}), encoding="utf-8")
        result = extract_entries_from_json(f)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# text_utils — load_line_ranges edge cases
# ---------------------------------------------------------------------------
from modules.core.text_utils import load_line_ranges


class TestLoadLineRangesExtended:
    def test_parenthesized_ranges(self, tmp_path):
        f = tmp_path / "ranges.txt"
        f.write_text("(1, 10)\n(11, 20)\n", encoding="utf-8")
        result = load_line_ranges(f)
        assert result == [(1, 10), (11, 20)]

    def test_bare_ranges(self, tmp_path):
        f = tmp_path / "ranges.txt"
        f.write_text("1, 10\n11, 20\n", encoding="utf-8")
        result = load_line_ranges(f)
        assert result == [(1, 10), (11, 20)]

    def test_empty_lines_skipped(self, tmp_path):
        f = tmp_path / "ranges.txt"
        f.write_text("\n1, 10\n\n11, 20\n\n", encoding="utf-8")
        result = load_line_ranges(f)
        assert result == [(1, 10), (11, 20)]

    def test_invalid_format_skipped(self, tmp_path):
        f = tmp_path / "ranges.txt"
        f.write_text("# comment\n1, 10\nbad line\n11, 20\n", encoding="utf-8")
        result = load_line_ranges(f)
        assert len(result) == 2

    def test_wrong_number_of_parts(self, tmp_path):
        f = tmp_path / "ranges.txt"
        f.write_text("1, 10, 20\n11, 20\n", encoding="utf-8")
        result = load_line_ranges(f)
        assert result == [(11, 20)]

    def test_non_integer_values(self, tmp_path):
        f = tmp_path / "ranges.txt"
        f.write_text("abc, def\n1, 10\n", encoding="utf-8")
        result = load_line_ranges(f)
        assert result == [(1, 10)]

    def test_nonexistent_file(self, tmp_path):
        result = load_line_ranges(tmp_path / "nonexistent.txt")
        assert result == []

    def test_empty_file(self, tmp_path):
        f = tmp_path / "ranges.txt"
        f.write_text("", encoding="utf-8")
        result = load_line_ranges(f)
        assert result == []


# ---------------------------------------------------------------------------
# prompt_utils — uncovered marker/context paths
# ---------------------------------------------------------------------------
from modules.llm.prompt_utils import render_prompt_with_schema, load_prompt_template


class TestRenderPromptExtended:
    def test_context_injection(self):
        prompt = "Prompt:\nContext:\n{{CONTEXT}}\nDo things."
        result = render_prompt_with_schema(
            prompt, {"type": "object"}, context="Historical cookbook context"
        )
        assert "Historical cookbook context" in result
        assert "{{CONTEXT}}" not in result

    def test_empty_context_removed(self):
        prompt = "Prompt:\nContext:\n{{CONTEXT}}\nDo things."
        result = render_prompt_with_schema(prompt, {"type": "object"}, context="")
        assert "{{CONTEXT}}" not in result
        assert "Context:" not in result

    def test_none_context_removed(self):
        prompt = "Prompt:\nContext:\n{{CONTEXT}}\nDo things."
        result = render_prompt_with_schema(prompt, {"type": "object"}, context=None)
        assert "{{CONTEXT}}" not in result

    def test_schema_name_injection(self):
        prompt = "Extract {{SCHEMA_NAME}} entries"
        result = render_prompt_with_schema(
            prompt, {}, schema_name="BibliographicEntries"
        )
        assert "BibliographicEntries" in result

    def test_schema_placeholder_injection(self):
        prompt = "Schema: {{TRANSCRIPTION_SCHEMA}}"
        schema = {"type": "object", "properties": {"id": {"type": "integer"}}}
        result = render_prompt_with_schema(prompt, schema)
        assert "object" in result
        assert "{{TRANSCRIPTION_SCHEMA}}" not in result

    def test_no_inject_schema(self):
        prompt = "Schema: {{TRANSCRIPTION_SCHEMA}}"
        result = render_prompt_with_schema(prompt, {"type": "object"}, inject_schema=False)
        assert "{{TRANSCRIPTION_SCHEMA}}" not in result
        assert "object" not in result

    def test_empty_schema_no_inject(self):
        prompt = "Schema: {{TRANSCRIPTION_SCHEMA}}"
        result = render_prompt_with_schema(prompt, {})
        assert "{{TRANSCRIPTION_SCHEMA}}" not in result

    def test_marker_based_injection(self):
        prompt = 'The JSON schema:\n{"old": "schema"}'
        schema = {"new": "schema"}
        result = render_prompt_with_schema(prompt, schema)
        assert '"new"' in result

    def test_marker_without_brace(self):
        prompt = "The JSON schema: missing braces"
        schema = {"key": "value"}
        result = render_prompt_with_schema(prompt, schema)
        assert '"key"' in result

    def test_no_placeholder_appends(self):
        prompt = "Simple prompt without any placeholder"
        schema = {"key": "value"}
        result = render_prompt_with_schema(prompt, schema)
        assert "The JSON schema:" in result
        assert '"key"' in result

    def test_non_serializable_schema_fallback(self):
        prompt = "Schema: {{TRANSCRIPTION_SCHEMA}}"
        # Use an object that's not JSON serializable to trigger fallback
        bad_schema = {"key": float("inf")}
        result = render_prompt_with_schema(prompt, bad_schema)
        assert "{{TRANSCRIPTION_SCHEMA}}" not in result


class TestLoadPromptTemplate:
    def test_loads_file(self, tmp_path):
        f = tmp_path / "prompt.txt"
        f.write_text("  Hello world  ", encoding="utf-8")
        result = load_prompt_template(f)
        assert result == "Hello world"

    def test_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_prompt_template(tmp_path / "nonexistent.txt")


# ---------------------------------------------------------------------------
# chunking_service — from_config already covered, but test auto-adjust wiring
# ---------------------------------------------------------------------------
from modules.core.chunking_service import ChunkSlice, apply_chunk_slice


class TestChunkSliceExtended:
    def test_both_set_raises(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            ChunkSlice(first_n=1, last_n=1)

    def test_first_n_zero_raises(self):
        with pytest.raises(ValueError, match="first_n must be >= 1"):
            ChunkSlice(first_n=0)

    def test_last_n_zero_raises(self):
        with pytest.raises(ValueError, match="last_n must be >= 1"):
            ChunkSlice(last_n=0)

    def test_negative_first_n_raises(self):
        with pytest.raises(ValueError, match="first_n must be >= 1"):
            ChunkSlice(first_n=-1)


class TestApplyChunkSliceExtended:
    def test_first_n_exceeds_total(self):
        chunks = ["a", "b"]
        ranges = [(1, 5), (6, 10)]
        result_c, result_r = apply_chunk_slice(chunks, ranges, ChunkSlice(first_n=10))
        assert result_c == chunks
        assert result_r == ranges

    def test_last_n_exceeds_total(self):
        chunks = ["a", "b"]
        ranges = [(1, 5), (6, 10)]
        result_c, result_r = apply_chunk_slice(chunks, ranges, ChunkSlice(last_n=10))
        assert result_c == chunks
        assert result_r == ranges

    def test_first_n_subset(self):
        chunks = ["a", "b", "c", "d"]
        ranges = [(1, 5), (6, 10), (11, 15), (16, 20)]
        result_c, result_r = apply_chunk_slice(chunks, ranges, ChunkSlice(first_n=2))
        assert result_c == ["a", "b"]
        assert result_r == [(1, 5), (6, 10)]

    def test_last_n_subset(self):
        chunks = ["a", "b", "c", "d"]
        ranges = [(1, 5), (6, 10), (11, 15), (16, 20)]
        result_c, result_r = apply_chunk_slice(chunks, ranges, ChunkSlice(last_n=2))
        assert result_c == ["c", "d"]
        assert result_r == [(11, 15), (16, 20)]
