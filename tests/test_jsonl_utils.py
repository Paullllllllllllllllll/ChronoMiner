"""Tests for modules.infra.jsonl."""

import json
import re
from pathlib import Path

import pytest

from modules.infra.jsonl import (
    JsonlWriter,
    extract_completed_ids,
    read_jsonl_records,
)


# ---------------------------------------------------------------------------
# JsonlWriter
# ---------------------------------------------------------------------------

class TestJsonlWriter:
    def test_write_single_record(self, tmp_path: Path) -> None:
        path = tmp_path / "out.jsonl"
        record = {"custom_id": "doc-chunk-1", "value": 42}
        with JsonlWriter(path, mode="w") as w:
            w.write_record(record)

        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0]) == record

    def test_write_multiple_records(self, tmp_path: Path) -> None:
        path = tmp_path / "out.jsonl"
        records = [{"id": i} for i in range(5)]
        with JsonlWriter(path, mode="w") as w:
            for r in records:
                w.write_record(r)

        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 5
        for line, expected in zip(lines, records):
            assert json.loads(line) == expected

    def test_append_mode(self, tmp_path: Path) -> None:
        path = tmp_path / "out.jsonl"
        with JsonlWriter(path, mode="w") as w:
            w.write_record({"first": True})
        with JsonlWriter(path, mode="a") as w:
            w.write_record({"second": True})

        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"first": True}
        assert json.loads(lines[1]) == {"second": True}

    def test_ensure_ascii_false(self, tmp_path: Path) -> None:
        path = tmp_path / "out.jsonl"
        record = {"text": "Ünïcödé"}
        with JsonlWriter(path, mode="w") as w:
            w.write_record(record)

        raw = path.read_text(encoding="utf-8")
        assert "Ünïcödé" in raw  # not escaped

    def test_invalid_mode_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="mode must be"):
            JsonlWriter(tmp_path / "out.jsonl", mode="r")

    def test_write_outside_context_raises(self, tmp_path: Path) -> None:
        writer = JsonlWriter(tmp_path / "out.jsonl")
        with pytest.raises(RuntimeError, match="not open"):
            writer.write_record({"a": 1})


# ---------------------------------------------------------------------------
# read_jsonl_records
# ---------------------------------------------------------------------------

class TestReadJsonlRecords:
    def test_read_valid_file(self, tmp_path: Path) -> None:
        path = tmp_path / "data.jsonl"
        records = [{"id": 1}, {"id": 2}]
        path.write_text(
            "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
        )
        result = list(read_jsonl_records(path))
        assert result == records

    def test_skips_empty_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "data.jsonl"
        path.write_text('{"a":1}\n\n\n{"b":2}\n', encoding="utf-8")
        result = list(read_jsonl_records(path))
        assert result == [{"a": 1}, {"b": 2}]

    def test_skips_malformed_json(self, tmp_path: Path) -> None:
        path = tmp_path / "data.jsonl"
        path.write_text('{"ok":1}\nNOT_JSON\n{"ok":2}\n', encoding="utf-8")
        result = list(read_jsonl_records(path))
        assert result == [{"ok": 1}, {"ok": 2}]

    def test_missing_file_yields_nothing(self, tmp_path: Path) -> None:
        result = list(read_jsonl_records(tmp_path / "missing.jsonl"))
        assert result == []


# ---------------------------------------------------------------------------
# extract_completed_ids
# ---------------------------------------------------------------------------

class TestExtractCompletedIds:
    def test_extract_chunk_ids(self, tmp_path: Path) -> None:
        path = tmp_path / "temp.jsonl"
        records = [
            {"custom_id": "doc-chunk-1"},
            {"custom_id": "doc-chunk-3"},
            {"custom_id": "doc-chunk-5"},
        ]
        path.write_text(
            "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
        )
        ids = extract_completed_ids(path)
        assert ids == {1, 3, 5}

    def test_extract_range_ids(self, tmp_path: Path) -> None:
        path = tmp_path / "adjust.jsonl"
        records = [
            {"custom_id": "doc-range-2"},
            {"custom_id": "doc-range-4"},
        ]
        path.write_text(
            "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
        )
        ids = extract_completed_ids(path)
        assert ids == {2, 4}

    def test_custom_pattern(self, tmp_path: Path) -> None:
        path = tmp_path / "data.jsonl"
        records = [
            {"custom_id": "file-page-10"},
            {"custom_id": "file-page-20"},
        ]
        path.write_text(
            "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
        )
        pat = re.compile(r"-page-(\d+)$")
        ids = extract_completed_ids(path, id_pattern=pat)
        assert ids == {10, 20}

    def test_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        assert extract_completed_ids(path) == set()

    def test_missing_file(self, tmp_path: Path) -> None:
        assert extract_completed_ids(tmp_path / "missing.jsonl") == set()

    def test_records_without_custom_id_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "data.jsonl"
        records = [
            {"custom_id": "doc-chunk-1"},
            {"batch_tracking": {"id": "xyz"}},
            {"other_field": True},
        ]
        path.write_text(
            "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
        )
        ids = extract_completed_ids(path)
        assert ids == {1}
