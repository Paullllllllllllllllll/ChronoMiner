"""Tests for modules/llm/batching.py.

Covers write_batch_file, build_batch_files (splitting, single file, empty),
get_batch_chunk_size, get_max_batch_bytes, and _get_extraction_config.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from modules.llm.batching import (
    DEFAULT_BATCH_CHUNK_SIZE,
    DEFAULT_MAX_BATCH_BYTES,
    build_batch_files,
    get_batch_chunk_size,
    get_max_batch_bytes,
    write_batch_file,
)


# ---------------------------------------------------------------------------
# write_batch_file
# ---------------------------------------------------------------------------

class TestWriteBatchFile:
    def test_writes_lines(self, tmp_path):
        output = tmp_path / "batch.jsonl"
        lines = ['{"id": 1}', '{"id": 2}', '{"id": 3}']
        result = write_batch_file(lines, output)

        assert result.exists()
        content = result.read_text(encoding="utf-8").strip().split("\n")
        assert len(content) == 3
        assert content[0] == '{"id": 1}'

    def test_empty_lines(self, tmp_path):
        output = tmp_path / "empty.jsonl"
        result = write_batch_file([], output)
        assert result.exists()
        assert result.read_text(encoding="utf-8") == ""

    def test_return_path(self, tmp_path):
        output = tmp_path / "batch.jsonl"
        result = write_batch_file(["line1"], output)
        assert isinstance(result, Path)


# ---------------------------------------------------------------------------
# get_batch_chunk_size / get_max_batch_bytes
# ---------------------------------------------------------------------------

class TestConfigFunctions:
    def test_batch_chunk_size_default(self):
        result = get_batch_chunk_size()
        assert isinstance(result, int)
        assert result > 0

    def test_max_batch_bytes_default(self):
        result = get_max_batch_bytes()
        assert isinstance(result, int)
        assert result > 0

    def test_batch_chunk_size_from_config(self):
        with patch(
            "modules.llm.batching._get_extraction_config",
            return_value={"batch_chunk_size": 25},
        ):
            assert get_batch_chunk_size() == 25

    def test_max_batch_bytes_from_config(self):
        with patch(
            "modules.llm.batching._get_extraction_config",
            return_value={"max_batch_bytes": 1024},
        ):
            assert get_max_batch_bytes() == 1024

    def test_batch_chunk_size_invalid_falls_back(self):
        with patch(
            "modules.llm.batching._get_extraction_config",
            return_value={"batch_chunk_size": -5},
        ):
            assert get_batch_chunk_size() == DEFAULT_BATCH_CHUNK_SIZE

    def test_max_batch_bytes_invalid_falls_back(self):
        with patch(
            "modules.llm.batching._get_extraction_config",
            return_value={"max_batch_bytes": 0},
        ):
            assert get_max_batch_bytes() == DEFAULT_MAX_BATCH_BYTES

    def test_batch_chunk_size_non_integer_falls_back(self):
        with patch(
            "modules.llm.batching._get_extraction_config",
            return_value={"batch_chunk_size": "not_a_number"},
        ):
            assert get_batch_chunk_size() == DEFAULT_BATCH_CHUNK_SIZE

    def test_max_batch_bytes_non_integer_falls_back(self):
        with patch(
            "modules.llm.batching._get_extraction_config",
            return_value={"max_batch_bytes": "bad"},
        ):
            assert get_max_batch_bytes() == DEFAULT_MAX_BATCH_BYTES


# ---------------------------------------------------------------------------
# build_batch_files
# ---------------------------------------------------------------------------

class TestBuildBatchFiles:
    def test_empty_request_lines(self, tmp_path):
        result = build_batch_files([], tmp_path / "batch.jsonl")
        assert result == []

    def test_single_batch(self, tmp_path):
        lines = [f'{{"id": {i}}}' for i in range(5)]
        base = tmp_path / "batch.jsonl"

        with patch("modules.llm.batching.get_batch_chunk_size", return_value=50):
            with patch("modules.llm.batching.get_max_batch_bytes", return_value=10 * 1024 * 1024):
                result = build_batch_files(lines, base)

        assert len(result) == 1
        assert result[0] == base
        content = result[0].read_text(encoding="utf-8").strip().split("\n")
        assert len(content) == 5

    def test_splits_by_chunk_size(self, tmp_path):
        lines = [f'{{"id": {i}}}' for i in range(10)]
        base = tmp_path / "batch.jsonl"

        with patch("modules.llm.batching.get_batch_chunk_size", return_value=3):
            with patch("modules.llm.batching.get_max_batch_bytes", return_value=10 * 1024 * 1024):
                result = build_batch_files(lines, base)

        # 10 lines / 3 per chunk = 4 batches (3+3+3+1)
        assert len(result) == 4
        # First part should be _part1
        assert "_part1" in result[0].name or result[0] == base

    def test_splits_by_byte_size(self, tmp_path):
        # Each line is ~13 bytes + newline = ~14 bytes
        lines = [f'{{"id": {i}}}' for i in range(10)]
        base = tmp_path / "batch.jsonl"

        with patch("modules.llm.batching.get_batch_chunk_size", return_value=100):
            with patch("modules.llm.batching.get_max_batch_bytes", return_value=50):
                result = build_batch_files(lines, base)

        # With ~14 bytes per line and 50 byte max, should split into multiple files
        assert len(result) > 1

    def test_single_line(self, tmp_path):
        lines = ['{"id": 1}']
        base = tmp_path / "batch.jsonl"

        with patch("modules.llm.batching.get_batch_chunk_size", return_value=50):
            with patch("modules.llm.batching.get_max_batch_bytes", return_value=10 * 1024 * 1024):
                result = build_batch_files(lines, base)

        assert len(result) == 1
        assert result[0] == base
