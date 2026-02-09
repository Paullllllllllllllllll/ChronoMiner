from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest


class TestFileStatus:
    """Tests for FileStatus enum and detection utilities."""

    def test_file_status_values(self):
        from modules.core.resume import FileStatus

        assert FileStatus.NOT_STARTED.value == "not_started"
        assert FileStatus.PARTIAL.value == "partial"
        assert FileStatus.COMPLETE.value == "complete"


class TestBuildExtractionMetadata:
    """Tests for build_extraction_metadata helper."""

    def test_metadata_fields(self):
        from modules.core.resume import build_extraction_metadata

        meta = build_extraction_metadata(
            schema_name="BibliographicEntries",
            model_name="gpt-4o",
            chunking_method="auto",
            total_chunks=5,
        )
        assert meta["schema_name"] == "BibliographicEntries"
        assert meta["model_name"] == "gpt-4o"
        assert meta["chunking_method"] == "auto"
        assert meta["total_chunks"] == 5
        assert "timestamp" in meta
        assert meta["version"] == 1

    def test_custom_timestamp(self):
        from modules.core.resume import build_extraction_metadata

        meta = build_extraction_metadata(
            schema_name="X",
            model_name="Y",
            chunking_method="Z",
            total_chunks=1,
            timestamp="2025-01-01T00:00:00+00:00",
        )
        assert meta["timestamp"] == "2025-01-01T00:00:00+00:00"


class TestDetectExtractionStatus:
    """Tests for detect_extraction_status."""

    def test_not_started_when_no_file(self, tmp_path: Path):
        from modules.core.resume import FileStatus, detect_extraction_status

        missing = tmp_path / "nonexistent_output.json"
        status, completed = detect_extraction_status(missing, expected_chunks=3)
        assert status == FileStatus.NOT_STARTED
        assert completed == set()

    def test_not_started_when_invalid_json(self, tmp_path: Path):
        from modules.core.resume import FileStatus, detect_extraction_status

        bad_file = tmp_path / "bad_output.json"
        bad_file.write_text("NOT JSON", encoding="utf-8")
        status, completed = detect_extraction_status(bad_file, expected_chunks=3)
        assert status == FileStatus.NOT_STARTED
        assert completed == set()

    def test_not_started_when_empty_records(self, tmp_path: Path):
        from modules.core.resume import FileStatus, detect_extraction_status, _METADATA_KEY

        out = tmp_path / "empty_output.json"
        out.write_text(json.dumps({_METADATA_KEY: {}, "records": []}), encoding="utf-8")
        status, completed = detect_extraction_status(out, expected_chunks=3)
        assert status == FileStatus.NOT_STARTED
        assert completed == set()

    def test_complete_when_all_chunks_present(self, tmp_path: Path):
        from modules.core.resume import FileStatus, detect_extraction_status, _METADATA_KEY

        records = [
            {"custom_id": "file-chunk-1", "response": {}},
            {"custom_id": "file-chunk-2", "response": {}},
            {"custom_id": "file-chunk-3", "response": {}},
        ]
        out = tmp_path / "complete_output.json"
        out.write_text(json.dumps({_METADATA_KEY: {}, "records": records}), encoding="utf-8")
        status, completed = detect_extraction_status(out, expected_chunks=3)
        assert status == FileStatus.COMPLETE
        assert completed == {1, 2, 3}

    def test_partial_when_some_chunks_missing(self, tmp_path: Path):
        from modules.core.resume import FileStatus, detect_extraction_status, _METADATA_KEY

        records = [
            {"custom_id": "file-chunk-1", "response": {}},
            {"custom_id": "file-chunk-3", "response": {}},
        ]
        out = tmp_path / "partial_output.json"
        out.write_text(json.dumps({_METADATA_KEY: {}, "records": records}), encoding="utf-8")
        status, completed = detect_extraction_status(out, expected_chunks=3)
        assert status == FileStatus.PARTIAL
        assert completed == {1, 3}

    def test_legacy_bare_list_format(self, tmp_path: Path):
        from modules.core.resume import FileStatus, detect_extraction_status

        records = [
            {"custom_id": "file-chunk-1", "response": {}},
            {"custom_id": "file-chunk-2", "response": {}},
        ]
        out = tmp_path / "legacy_output.json"
        out.write_text(json.dumps(records), encoding="utf-8")
        status, completed = detect_extraction_status(out, expected_chunks=2)
        assert status == FileStatus.COMPLETE
        assert completed == {1, 2}

    def test_complete_with_more_chunks_than_expected(self, tmp_path: Path):
        from modules.core.resume import FileStatus, detect_extraction_status, _METADATA_KEY

        records = [
            {"custom_id": "file-chunk-1", "response": {}},
            {"custom_id": "file-chunk-2", "response": {}},
            {"custom_id": "file-chunk-3", "response": {}},
        ]
        out = tmp_path / "extra_output.json"
        out.write_text(json.dumps({_METADATA_KEY: {}, "records": records}), encoding="utf-8")
        status, completed = detect_extraction_status(out, expected_chunks=2)
        assert status == FileStatus.COMPLETE
        assert completed == {1, 2, 3}


class TestGetOutputJsonPath:
    """Tests for get_output_json_path."""

    def test_output_in_schema_dir(self, tmp_path: Path):
        from modules.core.resume import get_output_json_path

        schema_paths = {"output": str(tmp_path / "output")}
        paths_config = {"general": {"input_paths_is_output_path": False}}
        result = get_output_json_path(
            tmp_path / "myfile.txt", paths_config, schema_paths
        )
        assert result.name == "myfile_output.json"
        assert "output" in str(result)

    def test_output_in_input_dir(self, tmp_path: Path):
        from modules.core.resume import get_output_json_path

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        paths_config = {"general": {"input_paths_is_output_path": True}}
        result = get_output_json_path(
            input_dir / "myfile.txt", paths_config, {}
        )
        assert result.name == "myfile_output.json"
        assert str(input_dir) in str(result)


class TestReadExtractionMetadata:
    """Tests for read_extraction_metadata."""

    def test_reads_metadata(self, tmp_path: Path):
        from modules.core.resume import read_extraction_metadata, _METADATA_KEY

        meta = {"schema_name": "Test", "version": 1}
        out = tmp_path / "out.json"
        out.write_text(json.dumps({_METADATA_KEY: meta, "records": []}), encoding="utf-8")
        result = read_extraction_metadata(out)
        assert result == meta

    def test_returns_none_for_missing_file(self, tmp_path: Path):
        from modules.core.resume import read_extraction_metadata

        result = read_extraction_metadata(tmp_path / "nope.json")
        assert result is None

    def test_returns_none_for_bare_list(self, tmp_path: Path):
        from modules.core.resume import read_extraction_metadata

        out = tmp_path / "legacy.json"
        out.write_text(json.dumps([{"custom_id": "x"}]), encoding="utf-8")
        result = read_extraction_metadata(out)
        assert result is None


class TestAdjustmentMarker:
    """Tests for line-range adjustment marker read/write/check."""

    def test_write_and_read_marker(self, tmp_path: Path):
        from modules.core.resume import write_adjustment_marker, read_adjustment_marker

        lr_file = tmp_path / "test_line_ranges.txt"
        lr_file.write_text("(1, 100)\n", encoding="utf-8")

        write_adjustment_marker(
            lr_file,
            boundary_type="BibliographicEntries",
            context_window=6,
            model_name="gpt-4o",
        )

        marker = read_adjustment_marker(lr_file)
        assert marker is not None
        assert marker["boundary_type"] == "BibliographicEntries"
        assert marker["context_window"] == 6
        assert marker["model_name"] == "gpt-4o"
        assert "adjusted_at" in marker

    def test_read_marker_returns_none_when_missing(self, tmp_path: Path):
        from modules.core.resume import read_adjustment_marker

        lr_file = tmp_path / "no_marker_line_ranges.txt"
        lr_file.write_text("(1, 50)\n", encoding="utf-8")
        assert read_adjustment_marker(lr_file) is None

    def test_is_adjustment_current_matches(self, tmp_path: Path):
        from modules.core.resume import write_adjustment_marker, is_adjustment_current

        lr_file = tmp_path / "test_line_ranges.txt"
        lr_file.write_text("(1, 100)\n", encoding="utf-8")

        write_adjustment_marker(
            lr_file,
            boundary_type="BibliographicEntries",
            context_window=6,
            model_name="gpt-4o",
        )

        assert is_adjustment_current(
            lr_file,
            boundary_type="BibliographicEntries",
            context_window=6,
            model_name="gpt-4o",
        )

    def test_is_adjustment_current_mismatch_boundary(self, tmp_path: Path):
        from modules.core.resume import write_adjustment_marker, is_adjustment_current

        lr_file = tmp_path / "test_line_ranges.txt"
        lr_file.write_text("(1, 100)\n", encoding="utf-8")

        write_adjustment_marker(
            lr_file,
            boundary_type="BibliographicEntries",
            context_window=6,
            model_name="gpt-4o",
        )

        assert not is_adjustment_current(
            lr_file,
            boundary_type="HistoricalRecipes",
            context_window=6,
            model_name="gpt-4o",
        )

    def test_is_adjustment_current_mismatch_model(self, tmp_path: Path):
        from modules.core.resume import write_adjustment_marker, is_adjustment_current

        lr_file = tmp_path / "test_line_ranges.txt"
        lr_file.write_text("(1, 100)\n", encoding="utf-8")

        write_adjustment_marker(
            lr_file,
            boundary_type="BibliographicEntries",
            context_window=6,
            model_name="gpt-4o",
        )

        assert not is_adjustment_current(
            lr_file,
            boundary_type="BibliographicEntries",
            context_window=6,
            model_name="claude-3-haiku-20240307",
        )

    def test_is_adjustment_current_mismatch_context_window(self, tmp_path: Path):
        from modules.core.resume import write_adjustment_marker, is_adjustment_current

        lr_file = tmp_path / "test_line_ranges.txt"
        lr_file.write_text("(1, 100)\n", encoding="utf-8")

        write_adjustment_marker(
            lr_file,
            boundary_type="BibliographicEntries",
            context_window=6,
            model_name="gpt-4o",
        )

        assert not is_adjustment_current(
            lr_file,
            boundary_type="BibliographicEntries",
            context_window=10,
            model_name="gpt-4o",
        )

    def test_is_adjustment_current_no_marker(self, tmp_path: Path):
        from modules.core.resume import is_adjustment_current

        lr_file = tmp_path / "no_marker.txt"
        lr_file.write_text("(1, 50)\n", encoding="utf-8")
        assert not is_adjustment_current(
            lr_file,
            boundary_type="X",
            context_window=6,
            model_name="Y",
        )

    def test_marker_path_format(self, tmp_path: Path):
        from modules.core.resume import _adjusted_marker_path

        lr_file = tmp_path / "test_line_ranges.txt"
        marker_path = _adjusted_marker_path(lr_file)
        assert marker_path.name == "test_line_ranges.txt.adjusted_meta"


class TestCliResumeFlags:
    """Tests for --resume and --force CLI argument parsing."""

    def test_process_parser_has_resume_flag(self):
        from modules.cli.args_parser import create_process_parser

        parser = create_process_parser()
        args = parser.parse_args([
            "--schema", "Test",
            "--input", "data/",
            "--resume",
        ])
        assert args.resume is True
        assert args.force is False

    def test_process_parser_has_force_flag(self):
        from modules.cli.args_parser import create_process_parser

        parser = create_process_parser()
        args = parser.parse_args([
            "--schema", "Test",
            "--input", "data/",
            "--force",
        ])
        assert args.force is True
        assert args.resume is False

    def test_process_parser_resume_and_force(self):
        from modules.cli.args_parser import create_process_parser

        parser = create_process_parser()
        args = parser.parse_args([
            "--schema", "Test",
            "--input", "data/",
            "--resume",
            "--force",
        ])
        assert args.resume is True
        assert args.force is True

    def test_adjust_ranges_parser_has_resume_flag(self):
        from modules.cli.args_parser import create_adjust_ranges_parser

        parser = create_adjust_ranges_parser()
        args = parser.parse_args([
            "--schema", "Test",
            "--input", "data/",
            "--resume",
        ])
        assert args.resume is True

    def test_adjust_ranges_parser_has_force_flag(self):
        from modules.cli.args_parser import create_adjust_ranges_parser

        parser = create_adjust_ranges_parser()
        args = parser.parse_args([
            "--schema", "Test",
            "--input", "data/",
            "--force",
        ])
        assert args.force is True
