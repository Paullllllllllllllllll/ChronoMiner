"""Regression tests for the v1.20.0 batch/sync output-format unification.

Batch finalization (check_batches / repair_extractions) writes the same
``{stem}_output.json`` shape as synchronous extraction: ``records`` plus
``_chronominer_metadata`` with a ``batch_tracking`` block. The legacy
``{stem}_final_output.json`` shape is no longer written but remains
readable via the fallbacks in ``json_utils`` and ``resume``.
"""

import json
from pathlib import Path

import main.check_batches as check_batches
from modules.batch.backends import BatchStatus, BatchStatusInfo
from modules.conversion.json_utils import extract_entries_from_json
from modules.extract.batch_output import build_unified_batch_output
from modules.extract.resume import (
    FileStatus,
    completed_indices_from_outputs,
    detect_extraction_status,
)

# ---------------------------------------------------------------------------
# build_unified_batch_output
# ---------------------------------------------------------------------------


def _entry(custom_id: str, payload: dict, chunk_range=None) -> dict:
    text = json.dumps(payload, ensure_ascii=False)
    entry = {
        "custom_id": custom_id,
        "response": text,
        "raw_response": {"output_text": text, "usage": {"total_tokens": 42}},
    }
    if chunk_range is not None:
        entry["chunk_range"] = chunk_range
    return entry


class TestBuildUnifiedBatchOutput:
    def test_writes_sync_shape_with_batch_tracking(self) -> None:
        entries = [
            _entry("doc-chunk-2", {"entries": [{"t": "B"}]}, chunk_range=[41, 80]),
            _entry("doc-chunk-1", {"entries": [{"t": "A"}]}, chunk_range=[1, 40]),
        ]
        tracking = [
            {"batch_id": "batch_a", "provider": "openai", "request_count": 2}
        ]
        result = build_unified_batch_output(
            entries,
            tracking,
            schema_name="TestSchema",
            completed_batches=1,
        )

        assert set(result) == {"_chronominer_metadata", "records"}
        meta = result["_chronominer_metadata"]
        assert meta["schema_name"] == "TestSchema"
        assert meta["total_chunks"] == 2
        assert "partial" not in meta
        bt = meta["batch_tracking"]
        assert bt["provider"] == "openai"
        assert bt["batch_ids"] == ["batch_a"]
        assert bt["parts"] == 1
        assert bt["fully_completed"] is True
        assert bt["completed_batches"] == 1
        assert bt["tracking"] == tracking

        records = result["records"]
        assert [r["custom_id"] for r in records] == ["doc-chunk-1", "doc-chunk-2"]
        assert [r["chunk_index"] for r in records] == [1, 2]
        assert records[0]["chunk_range"] == [1, 40]
        body = records[0]["response"]
        assert json.loads(body["output_text"]) == {"entries": [{"t": "A"}]}
        assert body["response_data"]["usage"] == {"total_tokens": 42}

    def test_chunk_metadata_from_custom_id_map(self) -> None:
        entries = [_entry("doc-chunk-3", {"entries": []})]
        custom_id_map = {
            "doc-chunk-3": {
                "chunk_index": 3,
                "chunk_range": [81, 120],
                "total_chunks": 5,
            }
        }
        result = build_unified_batch_output(
            entries,
            [{"batch_id": "b1"}],
            schema_name="TestSchema",
            custom_id_map=custom_id_map,
        )
        record = result["records"][0]
        assert record["chunk_index"] == 3
        assert record["chunk_range"] == [81, 120]
        meta = result["_chronominer_metadata"]
        assert meta["total_chunks"] == 5
        # 1 of 5 expected records present -> partial
        assert meta["partial"] is True

    def test_failed_entries_fold_into_failed_chunks(self) -> None:
        entries = [
            _entry("doc-chunk-1", {"entries": [{"t": "A"}]}),
            {
                "custom_id": "doc-chunk-2",
                "response": None,
                "error": "boom",
                "error_code": "500",
            },
        ]
        result = build_unified_batch_output(
            entries,
            [{"batch_id": "b1", "provider": "anthropic"}],
            schema_name="TestSchema",
            fully_completed=False,
            failed_batches=1,
        )
        meta = result["_chronominer_metadata"]
        assert meta["partial"] is True
        assert meta["failed_chunks"] == [2]
        assert [r["custom_id"] for r in result["records"]] == ["doc-chunk-1"]

    def test_parsed_output_fallback_when_no_text(self) -> None:
        entries = [
            {
                "custom_id": "doc-chunk-1",
                "response": "",
                "raw_response": {},
                "parsed_output": {"entries": [{"t": "P"}]},
            }
        ]
        result = build_unified_batch_output(
            entries, [{"batch_id": "b1"}], schema_name="S"
        )
        text = result["records"][0]["response"]["output_text"]
        assert json.loads(text) == {"entries": [{"t": "P"}]}


# ---------------------------------------------------------------------------
# Downstream compatibility of the unified batch output
# ---------------------------------------------------------------------------


class TestUnifiedOutputDownstream:
    def _write_unified(self, path: Path, n_chunks: int = 2) -> None:
        entries = [
            _entry(f"doc-chunk-{i}", {"entries": [{"t": f"E{i}"}]})
            for i in range(1, n_chunks + 1)
        ]
        result = build_unified_batch_output(
            entries, [{"batch_id": "b1", "provider": "openai"}], schema_name="S"
        )
        path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def test_detect_extraction_status_complete(self, tmp_path: Path) -> None:
        out = tmp_path / "doc_output.json"
        self._write_unified(out, n_chunks=3)
        status, completed = detect_extraction_status(out, expected_chunks=3)
        assert status == FileStatus.COMPLETE
        assert completed == {1, 2, 3}

    def test_detect_extraction_status_partial(self, tmp_path: Path) -> None:
        out = tmp_path / "doc_output.json"
        self._write_unified(out, n_chunks=2)
        status, completed = detect_extraction_status(out, expected_chunks=4)
        assert status == FileStatus.PARTIAL
        assert completed == {1, 2}

    def test_detect_extraction_status_reads_page_ids(self, tmp_path: Path) -> None:
        out = tmp_path / "doc_output.json"
        data = {
            "records": [
                {"custom_id": "doc-page-1", "response": {}},
                {"custom_id": "doc-page-2", "response": {}},
            ]
        }
        out.write_text(json.dumps(data), encoding="utf-8")
        status, completed = detect_extraction_status(out, expected_chunks=2)
        assert status == FileStatus.COMPLETE
        assert completed == {1, 2}

    def test_converters_parse_unified_batch_output(self, tmp_path: Path) -> None:
        out = tmp_path / "doc_output.json"
        self._write_unified(out, n_chunks=2)
        entries = extract_entries_from_json(out)
        assert entries == [{"t": "E1"}, {"t": "E2"}]

    def test_converters_still_parse_legacy_responses(self, tmp_path: Path) -> None:
        legacy = tmp_path / "doc_final_output.json"
        payload = json.dumps({"entries": [{"t": "L"}]})
        legacy.write_text(
            json.dumps(
                {
                    "responses": [
                        {
                            "custom_id": "doc-chunk-1",
                            "response": payload,
                            "raw_response": {"output_text": payload},
                        }
                    ],
                    "tracking": [{"batch_id": "b1"}],
                }
            ),
            encoding="utf-8",
        )
        assert extract_entries_from_json(legacy) == [{"t": "L"}]

    def test_resume_reads_unified_and_legacy_outputs(self, tmp_path: Path) -> None:
        unified = tmp_path / "doc_output.json"
        self._write_unified(unified, n_chunks=2)
        legacy = tmp_path / "doc_final_output.json"
        legacy.write_text(
            json.dumps({"responses": [{"custom_id": "doc-chunk-5"}]}),
            encoding="utf-8",
        )
        assert completed_indices_from_outputs(unified, legacy) == {1, 2, 5}


# ---------------------------------------------------------------------------
# check_batches finalization (integration, backends mocked)
# ---------------------------------------------------------------------------


class _FakeBackend:
    def __init__(self) -> None:
        self.cleaned_up: list[str] = []

    def get_status(self, handle) -> BatchStatusInfo:
        return BatchStatusInfo(status=BatchStatus.COMPLETED, results_available=True)

    def cleanup(self, handle) -> None:
        self.cleaned_up.append(handle.batch_id)


def _write_part(path: Path, custom_ids: list[str], batch_id: str) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for i, cid in enumerate(custom_ids, 1):
            fh.write(
                json.dumps(
                    {
                        "batch_request": {
                            "custom_id": cid,
                            "order_index": i,
                            "metadata": {
                                "chunk_index": int(cid.rsplit("-", 1)[1]),
                                "total_chunks": 4,
                                "chunk_range": None,
                            },
                        }
                    }
                )
                + "\n"
            )
        fh.write(
            json.dumps(
                {
                    "batch_tracking": {
                        "batch_id": batch_id,
                        "provider": "openai",
                        "request_count": len(custom_ids),
                    }
                }
            )
            + "\n"
        )


class TestCheckBatchesFinalization:
    def test_multipart_finalization_writes_one_unified_output(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        root = tmp_path / "repo"
        out_dir = tmp_path / "out"
        root.mkdir()
        _write_part(root / "doc_temp_part1.jsonl", ["doc-chunk-1", "doc-chunk-2"], "bA")
        _write_part(root / "doc_temp_part2.jsonl", ["doc-chunk-3", "doc-chunk-4"], "bB")

        backend = _FakeBackend()
        monkeypatch.setattr(check_batches, "get_batch_backend", lambda p: backend)

        def fake_retrieve(track, temp_dir, status_cache):
            ids = {
                "bA": ["doc-chunk-1", "doc-chunk-2"],
                "bB": ["doc-chunk-3", "doc-chunk-4"],
            }[track["batch_id"]]
            return [
                _entry(cid, {"entries": [{"t": cid}]})
                for cid in ids
            ]

        monkeypatch.setattr(
            check_batches, "retrieve_responses_from_batch", fake_retrieve
        )

        class _FakeLoader:
            def get_paths_config(self):
                return {"general": {}}

        monkeypatch.setattr(check_batches, "get_config_loader", lambda: _FakeLoader())

        agg: dict[str, int] = {}
        check_batches.process_all_batches(
            root_folder=root,
            processing_settings={"retain_temporary_jsonl": True},
            schema_name="TestSchema",
            schema_config={"output": str(out_dir)},
            ui=None,
            agg=agg,
        )

        unified = out_dir / "doc_output.json"
        assert unified.exists()
        assert not (out_dir / "doc_final_output.json").exists()
        assert agg.get("finalized") == 1

        data = json.loads(unified.read_text(encoding="utf-8"))
        assert [r["custom_id"] for r in data["records"]] == [
            f"doc-chunk-{i}" for i in range(1, 5)
        ]
        meta = data["_chronominer_metadata"]
        assert meta["total_chunks"] == 4
        assert "partial" not in meta
        assert meta["batch_tracking"]["batch_ids"] == ["bA", "bB"]
        assert meta["batch_tracking"]["parts"] == 2
        # Remote cleanup ran only after the unified output was written.
        assert sorted(backend.cleaned_up) == ["bA", "bB"]

        # Resume treats the unified batch output like any sync output.
        status, completed = detect_extraction_status(unified, expected_chunks=4)
        assert status == FileStatus.COMPLETE
        assert completed == {1, 2, 3, 4}
