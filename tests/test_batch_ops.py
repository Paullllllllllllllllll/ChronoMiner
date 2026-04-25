"""Interface-level tests for ``modules.batch.ops``.

Covers the shared helpers that were extracted from ``main/check_batches.py``
in Phase D. These helpers are consumed by all three batch scripts
(``check_batches``, ``cancel_batches``, ``repair_extractions``) plus
``modules.batch`` itself, so they need reliable interface-level coverage.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from modules.batch.ops import (
    ERROR_FILE_KEYS,
    OUTPUT_FILE_KEYS,
    _extract_chunk_index,
    _normalize_response_entry,
    _order_responses,
    _recover_missing_batch_ids,
    _resolve_file_id_by_keys,
    _response_to_text,
    load_config,
    process_batch_output_file,
)


@pytest.mark.unit
class TestConstants:
    def test_output_file_keys_are_strings(self):
        assert all(isinstance(k, str) for k in OUTPUT_FILE_KEYS)
        # Must contain both singular and list spellings so that provider
        # batches using either convention are recognized.
        assert "output_file_id" in OUTPUT_FILE_KEYS
        assert "output_file_ids" in OUTPUT_FILE_KEYS

    def test_error_file_keys_are_strings(self):
        assert all(isinstance(k, str) for k in ERROR_FILE_KEYS)
        assert "error_file_id" in ERROR_FILE_KEYS


@pytest.mark.unit
class TestExtractChunkIndex:
    def test_chunk_suffix(self):
        assert _extract_chunk_index("doc-chunk-7") == 7

    def test_page_suffix(self):
        assert _extract_chunk_index("doc-page-3") == 3

    def test_req_suffix(self):
        assert _extract_chunk_index("req-12") == 12

    def test_unknown_pattern(self):
        # Unknown IDs sort to the end (10**9).
        assert _extract_chunk_index("no-pattern") == 10**9

    def test_non_string_input(self):
        assert _extract_chunk_index(None) == 10**9
        assert _extract_chunk_index(42) == 10**9


@pytest.mark.unit
class TestOrderResponses:
    def test_order_by_chunk_index(self):
        items = [
            {"custom_id": "d-chunk-3", "v": "c"},
            {"custom_id": "d-chunk-1", "v": "a"},
            {"custom_id": "d-chunk-2", "v": "b"},
        ]
        ordered = _order_responses(items)
        assert [r["v"] for r in ordered] == ["a", "b", "c"]

    def test_order_map_takes_priority(self):
        items = [
            {"custom_id": "a", "v": "A"},
            {"custom_id": "b", "v": "B"},
            {"custom_id": "c", "v": "C"},
        ]
        ordered = _order_responses(items, order_map={"c": 0, "a": 1, "b": 2})
        assert [r["v"] for r in ordered] == ["C", "A", "B"]

    def test_non_dict_items_kept_at_end(self):
        items = [{"custom_id": "d-chunk-1", "v": "first"}, "stray-string"]
        ordered = _order_responses(items)
        assert ordered[0]["v"] == "first"
        assert ordered[1] == "stray-string"


@pytest.mark.unit
class TestResponseToText:
    def test_string_passthrough(self):
        assert _response_to_text("already plain") == "already plain"

    def test_output_text_string(self):
        assert _response_to_text({"output_text": "hello  "}) == "hello"

    def test_output_message_list(self):
        body = {
            "output": [
                {
                    "type": "message",
                    "content": [{"text": "part1"}, {"text": "part2"}],
                }
            ]
        }
        assert _response_to_text(body) == "part1part2"

    def test_non_dict_returns_empty(self):
        assert _response_to_text(None) == ""
        assert _response_to_text(42) == ""


@pytest.mark.unit
class TestNormalizeResponseEntry:
    def test_preserves_plain_string_response(self):
        result = _normalize_response_entry(
            {"custom_id": "x", "response": "raw-text"}
        )
        assert result["response"] == "raw-text"
        assert result["raw_response"] == "raw-text"

    def test_extracts_text_from_dict_response(self):
        entry = {
            "custom_id": "x",
            "response": {"output_text": "extracted"},
        }
        result = _normalize_response_entry(entry)
        assert result["response"] == "extracted"
        assert isinstance(result["raw_response"], dict)


@pytest.mark.unit
class TestResolveFileIdByKeys:
    def test_returns_first_match(self):
        batch = {"output_file_id": "file-abc", "response_file_id": "file-def"}
        assert (
            _resolve_file_id_by_keys(batch, OUTPUT_FILE_KEYS) == "file-abc"
        )

    def test_returns_none_for_missing(self):
        assert _resolve_file_id_by_keys({}, OUTPUT_FILE_KEYS) is None


@pytest.mark.unit
class TestProcessBatchOutputFile:
    def test_parses_responses_and_tracking(self, tmp_path):
        path = tmp_path / "temp.jsonl"
        lines = [
            json.dumps({"custom_id": "d-chunk-1", "response": "r1"}),
            json.dumps({"batch_tracking": {"batch_id": "b1", "provider": "openai"}}),
            json.dumps({"custom_id": "d-chunk-2", "response": "r2", "chunk_range": [1, 10]}),
            "",  # blank line ignored
        ]
        path.write_text("\n".join(lines), encoding="utf-8")

        result = process_batch_output_file(path)
        assert len(result["responses"]) == 2
        assert result["tracking"] == [
            {"batch_id": "b1", "provider": "openai"}
        ]

    def test_handles_malformed_lines(self, tmp_path):
        path = tmp_path / "temp.jsonl"
        path.write_text(
            "{not json}\n"
            + json.dumps({"custom_id": "ok", "response": "x"})
            + "\n",
            encoding="utf-8",
        )
        result = process_batch_output_file(path)
        assert len(result["responses"]) == 1
        assert result["tracking"] == []


@pytest.mark.unit
class TestRecoverMissingBatchIds:
    def test_returns_empty_when_no_debug_artifact(self, tmp_path):
        result = _recover_missing_batch_ids(
            temp_file=tmp_path / "missing.jsonl",
            identifier="doc",
            persist=False,
        )
        assert result == set()

    def test_reads_from_debug_artifact(self, tmp_path):
        temp_file = tmp_path / "doc_temp.jsonl"
        artifact = tmp_path / "doc_batch_submission_debug.json"
        artifact.write_text(
            json.dumps({"batch_ids": ["batch_a", "batch_b"]}),
            encoding="utf-8",
        )
        result = _recover_missing_batch_ids(
            temp_file=temp_file, identifier="doc", persist=False
        )
        assert result == {"batch_a", "batch_b"}

    def test_persist_appends_tracking_records(self, tmp_path):
        temp_file = tmp_path / "doc_temp.jsonl"
        temp_file.write_text("", encoding="utf-8")
        artifact = tmp_path / "doc_batch_submission_debug.json"
        artifact.write_text(json.dumps({"batch_ids": ["b1"]}), encoding="utf-8")

        _recover_missing_batch_ids(
            temp_file=temp_file, identifier="doc", persist=True
        )
        content = temp_file.read_text(encoding="utf-8").strip()
        assert content, "persist=True must append a tracking record"
        record = json.loads(content)
        assert record["batch_tracking"]["batch_id"] == "b1"


@pytest.mark.unit
class TestLoadConfigIntegration:
    """``load_config`` pulls the schema and processing settings from the
    shared config loader."""

    def test_returns_tuple_of_repo_info_and_settings(self, config_loader):
        repo_info_list, settings = load_config()
        assert isinstance(repo_info_list, list)
        assert all(len(entry) == 3 for entry in repo_info_list)
        assert isinstance(settings, dict)
        assert "retain_temporary_jsonl" in settings
