from __future__ import annotations

import json
from pathlib import Path


class TestFileStatus:
    """Tests for FileStatus enum and detection utilities."""

    def test_file_status_values(self):
        from modules.extract.resume import FileStatus

        assert FileStatus.NOT_STARTED.value == "not_started"
        assert FileStatus.PARTIAL.value == "partial"
        assert FileStatus.COMPLETE.value == "complete"


class TestBuildExtractionMetadata:
    """Tests for build_extraction_metadata helper."""

    def test_metadata_fields(self):
        from modules.extract.resume import build_extraction_metadata

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
        from modules.extract.resume import build_extraction_metadata

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
        from modules.extract.resume import FileStatus, detect_extraction_status

        missing = tmp_path / "nonexistent_output.json"
        status, completed = detect_extraction_status(missing, expected_chunks=3)
        assert status == FileStatus.NOT_STARTED
        assert completed == set()

    def test_not_started_when_invalid_json(self, tmp_path: Path):
        from modules.extract.resume import FileStatus, detect_extraction_status

        bad_file = tmp_path / "bad_output.json"
        bad_file.write_text("NOT JSON", encoding="utf-8")
        status, completed = detect_extraction_status(bad_file, expected_chunks=3)
        assert status == FileStatus.NOT_STARTED
        assert completed == set()

    def test_not_started_when_empty_records(self, tmp_path: Path):
        from modules.extract.resume import (
            _METADATA_KEY,
            FileStatus,
            detect_extraction_status,
        )

        out = tmp_path / "empty_output.json"
        out.write_text(json.dumps({_METADATA_KEY: {}, "records": []}), encoding="utf-8")
        status, completed = detect_extraction_status(out, expected_chunks=3)
        assert status == FileStatus.NOT_STARTED
        assert completed == set()

    def test_complete_when_all_chunks_present(self, tmp_path: Path):
        from modules.extract.resume import (
            _METADATA_KEY,
            FileStatus,
            detect_extraction_status,
        )

        records = [
            {"custom_id": "file-chunk-1", "response": {}},
            {"custom_id": "file-chunk-2", "response": {}},
            {"custom_id": "file-chunk-3", "response": {}},
        ]
        out = tmp_path / "complete_output.json"
        out.write_text(
            json.dumps({_METADATA_KEY: {}, "records": records}), encoding="utf-8"
        )
        status, completed = detect_extraction_status(out, expected_chunks=3)
        assert status == FileStatus.COMPLETE
        assert completed == {1, 2, 3}

    def test_partial_when_some_chunks_missing(self, tmp_path: Path):
        from modules.extract.resume import (
            _METADATA_KEY,
            FileStatus,
            detect_extraction_status,
        )

        records = [
            {"custom_id": "file-chunk-1", "response": {}},
            {"custom_id": "file-chunk-3", "response": {}},
        ]
        out = tmp_path / "partial_output.json"
        out.write_text(
            json.dumps({_METADATA_KEY: {}, "records": records}), encoding="utf-8"
        )
        status, completed = detect_extraction_status(out, expected_chunks=3)
        assert status == FileStatus.PARTIAL
        assert completed == {1, 3}

    def test_legacy_bare_list_format(self, tmp_path: Path):
        from modules.extract.resume import FileStatus, detect_extraction_status

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
        from modules.extract.resume import (
            _METADATA_KEY,
            FileStatus,
            detect_extraction_status,
        )

        records = [
            {"custom_id": "file-chunk-1", "response": {}},
            {"custom_id": "file-chunk-2", "response": {}},
            {"custom_id": "file-chunk-3", "response": {}},
        ]
        out = tmp_path / "extra_output.json"
        out.write_text(
            json.dumps({_METADATA_KEY: {}, "records": records}), encoding="utf-8"
        )
        status, completed = detect_extraction_status(out, expected_chunks=2)
        assert status == FileStatus.COMPLETE
        assert completed == {1, 2, 3}


class TestGetOutputJsonPath:
    """Tests for get_output_json_path."""

    def test_output_in_schema_dir(self, tmp_path: Path):
        from modules.extract.resume import get_output_json_path

        schema_paths = {"output": str(tmp_path / "output")}
        paths_config = {"general": {"input_paths_is_output_path": False}}
        result = get_output_json_path(
            tmp_path / "myfile.txt", paths_config, schema_paths
        )
        assert result.name == "myfile_output.json"
        assert "output" in str(result)

    def test_output_in_input_dir(self, tmp_path: Path):
        from modules.extract.resume import get_output_json_path

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        paths_config = {"general": {"input_paths_is_output_path": True}}
        result = get_output_json_path(input_dir / "myfile.txt", paths_config, {})
        assert result.name == "myfile_output.json"
        assert str(input_dir) in str(result)


class TestReadExtractionMetadata:
    """Tests for read_extraction_metadata."""

    def test_reads_metadata(self, tmp_path: Path):
        from modules.extract.resume import _METADATA_KEY, read_extraction_metadata

        meta = {"schema_name": "Test", "version": 1}
        out = tmp_path / "out.json"
        out.write_text(
            json.dumps({_METADATA_KEY: meta, "records": []}), encoding="utf-8"
        )
        result = read_extraction_metadata(out)
        assert result == meta

    def test_returns_none_for_missing_file(self, tmp_path: Path):
        from modules.extract.resume import read_extraction_metadata

        result = read_extraction_metadata(tmp_path / "nope.json")
        assert result is None

    def test_returns_none_for_bare_list(self, tmp_path: Path):
        from modules.extract.resume import read_extraction_metadata

        out = tmp_path / "legacy.json"
        out.write_text(json.dumps([{"custom_id": "x"}]), encoding="utf-8")
        result = read_extraction_metadata(out)
        assert result is None


class TestJsonlAdjustmentComplete:
    """Tests for JSONL-based adjustment completion check."""

    def test_complete_jsonl_detected(self, tmp_path: Path):
        from modules.infra.jsonl import (
            JsonlWriter,
            build_jsonl_header,
            finalize_jsonl_header,
            is_jsonl_adjustment_complete,
        )

        lr_file = tmp_path / "test_line_ranges.txt"
        lr_file.write_text("(1, 100)\n", encoding="utf-8")

        jsonl_path = tmp_path / "test_line_ranges_adjust_temp.jsonl"
        fingerprint = "abc123"
        with JsonlWriter(jsonl_path, mode="w") as writer:
            writer.write_record(
                build_jsonl_header(
                    ranges_fingerprint=fingerprint,
                    total_ranges=1,
                    boundary_type="BibliographicEntries",
                    model_name="gpt-4o",
                    context_window=6,
                )
            )

        finalize_jsonl_header(
            jsonl_path,
            stats={
                "total_ranges": 1,
                "ranges_adjusted": 0,
                "ranges_deleted": 0,
                "ranges_kept_original": 1,
                "total_llm_calls": 1,
            },
        )

        assert is_jsonl_adjustment_complete(
            lr_file,
            boundary_type="BibliographicEntries",
            context_window=6,
            model_name="gpt-4o",
            ranges_fingerprint=fingerprint,
        )

    def test_incomplete_jsonl_not_detected(self, tmp_path: Path):
        from modules.infra.jsonl import (
            JsonlWriter,
            build_jsonl_header,
            is_jsonl_adjustment_complete,
        )

        lr_file = tmp_path / "test_line_ranges.txt"
        lr_file.write_text("(1, 100)\n", encoding="utf-8")

        jsonl_path = tmp_path / "test_line_ranges_adjust_temp.jsonl"
        with JsonlWriter(jsonl_path, mode="w") as writer:
            writer.write_record(
                build_jsonl_header(
                    ranges_fingerprint="abc123",
                    total_ranges=1,
                    boundary_type="BibliographicEntries",
                    model_name="gpt-4o",
                    context_window=6,
                )
            )

        assert not is_jsonl_adjustment_complete(
            lr_file,
            boundary_type="BibliographicEntries",
            context_window=6,
            model_name="gpt-4o",
            ranges_fingerprint="abc123",
        )

    def test_mismatched_settings_not_detected(self, tmp_path: Path):
        from modules.infra.jsonl import (
            JsonlWriter,
            build_jsonl_header,
            finalize_jsonl_header,
            is_jsonl_adjustment_complete,
        )

        lr_file = tmp_path / "test_line_ranges.txt"
        lr_file.write_text("(1, 100)\n", encoding="utf-8")

        jsonl_path = tmp_path / "test_line_ranges_adjust_temp.jsonl"
        with JsonlWriter(jsonl_path, mode="w") as writer:
            writer.write_record(
                build_jsonl_header(
                    ranges_fingerprint="abc123",
                    total_ranges=1,
                    boundary_type="BibliographicEntries",
                    model_name="gpt-4o",
                    context_window=6,
                )
            )

        finalize_jsonl_header(
            jsonl_path,
            stats={
                "total_ranges": 1,
                "ranges_adjusted": 0,
                "ranges_deleted": 0,
                "ranges_kept_original": 1,
                "total_llm_calls": 1,
            },
        )

        assert not is_jsonl_adjustment_complete(
            lr_file,
            boundary_type="HistoricalRecipes",
            context_window=6,
            model_name="gpt-4o",
            ranges_fingerprint="abc123",
        )

    def test_no_jsonl_not_detected(self, tmp_path: Path):
        from modules.infra.jsonl import is_jsonl_adjustment_complete

        lr_file = tmp_path / "no_jsonl_line_ranges.txt"
        lr_file.write_text("(1, 50)\n", encoding="utf-8")
        assert not is_jsonl_adjustment_complete(
            lr_file,
            boundary_type="X",
            context_window=6,
            model_name="Y",
        )


class TestCliResumeFlags:
    """Tests for --resume and --force CLI argument parsing."""

    def test_process_parser_has_resume_flag(self):
        from main.cli_args import create_process_parser

        parser = create_process_parser()
        args = parser.parse_args(
            [
                "--schema",
                "Test",
                "--input",
                "data/",
                "--resume",
            ]
        )
        assert args.resume is True
        assert args.force is False

    def test_process_parser_has_force_flag(self):
        from main.cli_args import create_process_parser

        parser = create_process_parser()
        args = parser.parse_args(
            [
                "--schema",
                "Test",
                "--input",
                "data/",
                "--force",
            ]
        )
        assert args.force is True
        assert args.resume is False

    def test_process_parser_resume_and_force(self):
        from main.cli_args import create_process_parser

        parser = create_process_parser()
        args = parser.parse_args(
            [
                "--schema",
                "Test",
                "--input",
                "data/",
                "--resume",
                "--force",
            ]
        )
        assert args.resume is True
        assert args.force is True

    def test_adjust_ranges_parser_has_resume_flag(self):
        from main.cli_args import create_adjust_ranges_parser

        parser = create_adjust_ranges_parser()
        args = parser.parse_args(
            [
                "--schema",
                "Test",
                "--input",
                "data/",
                "--resume",
            ]
        )
        assert args.resume is True

    def test_adjust_ranges_parser_has_force_flag(self):
        from main.cli_args import create_adjust_ranges_parser

        parser = create_adjust_ranges_parser()
        args = parser.parse_args(
            [
                "--schema",
                "Test",
                "--input",
                "data/",
                "--force",
            ]
        )
        assert args.force is True


class _StubHandler:
    """Minimal schema handler: only ``schema_name`` is read by output gen."""

    schema_name = "TestSchema"


def _make_file_processor():
    """Build a FileProcessor with the minimum config it needs offline."""
    from modules.extract.file_processor import FileProcessor

    return FileProcessor(
        paths_config={"general": {"input_paths_is_output_path": True}},
        model_config={"extraction_model": {"name": "gpt-4o"}},
        chunking_config={"chunking": {"default_tokens_per_chunk": 10}},
        concurrency_config={},
    )


def _read_records(output_json: Path) -> list[dict]:
    data = json.loads(output_json.read_text(encoding="utf-8"))
    return data["records"]


def _completed_indices(records: list[dict]) -> set[int]:
    indices: set[int] = set()
    for rec in records:
        cid = str(rec.get("custom_id", ""))
        if "-chunk-" in cid:
            indices.add(int(cid.rsplit("-chunk-", 1)[1]))
    return indices


class TestResumeMergePreservesPriorRecords:
    """Regression: resume must not drop previously-saved records when the temp
    JSONL no longer holds them (e.g. retain_temporary_jsonl is false)."""

    def test_merge_existing_recovers_prior_records(self, tmp_path: Path):
        import asyncio

        from modules.extract.file_processor import _MessagingAdapter

        fp = _make_file_processor()

        # Prior run saved chunks 1-2 to output.json.
        output_json = tmp_path / "doc_output.json"
        prior_records = [
            {
                "custom_id": "doc-chunk-1",
                "chunk_index": 1,
                "chunk_range": None,
                "response": {"entries": ["a"]},
            },
            {
                "custom_id": "doc-chunk-2",
                "chunk_index": 2,
                "chunk_range": None,
                "response": {"entries": ["b"]},
            },
        ]
        output_json.write_text(
            json.dumps({"_chronominer_metadata": {}, "records": prior_records}),
            encoding="utf-8",
        )

        # Resume run: the temp JSONL was deleted after the prior run, so it now
        # holds only the newly-processed chunk 3.
        temp_jsonl = tmp_path / "doc_temp.jsonl"
        temp_jsonl.write_text(
            json.dumps(
                {
                    "custom_id": "doc-chunk-3",
                    "chunk_index": 3,
                    "response": {"body": {"entries": ["c"]}},
                }
            )
            + "\n",
            encoding="utf-8",
        )

        asyncio.run(
            fp._generate_output_files(
                temp_jsonl,
                output_json,
                _StubHandler(),
                {},  # no csv/docx/txt outputs configured
                _MessagingAdapter(),
                partial=False,
                merge_existing=True,
            )
        )

        records = _read_records(output_json)
        # All three chunks survive and are ordered by chunk_index.
        assert _completed_indices(records) == {1, 2, 3}
        assert [r["chunk_index"] for r in records] == [1, 2, 3]

    def test_no_merge_overwrites_cleanly(self, tmp_path: Path):
        """Without merge_existing (force / non-resume), output is rebuilt from
        the temp JSONL alone — prior records are not reintroduced."""
        import asyncio

        from modules.extract.file_processor import _MessagingAdapter

        fp = _make_file_processor()

        output_json = tmp_path / "doc_output.json"
        output_json.write_text(
            json.dumps(
                {
                    "_chronominer_metadata": {},
                    "records": [
                        {"custom_id": "doc-chunk-1", "chunk_index": 1, "response": {}}
                    ],
                }
            ),
            encoding="utf-8",
        )

        temp_jsonl = tmp_path / "doc_temp.jsonl"
        temp_jsonl.write_text(
            json.dumps(
                {
                    "custom_id": "doc-chunk-2",
                    "chunk_index": 2,
                    "response": {"body": {"entries": ["c"]}},
                }
            )
            + "\n",
            encoding="utf-8",
        )

        asyncio.run(
            fp._generate_output_files(
                temp_jsonl,
                output_json,
                _StubHandler(),
                {},
                _MessagingAdapter(),
                partial=False,
                merge_existing=False,
            )
        )

        records = _read_records(output_json)
        assert _completed_indices(records) == {2}
