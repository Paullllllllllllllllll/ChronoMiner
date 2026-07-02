"""Regression tests for the Tier 2 hardening fixes.

Covers: the CLI agent contract flags, the stricter transient-error classifier,
batch temp-file content sniffing, cross-shape completed-index detection, and
batch resume parity.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

import modules.extract.processing_strategy as ps


class _StubHandler:
    schema_name = "TestSchema"


# ---------------------------------------------------------------------------
# Item 16: transient-error classifier
# ---------------------------------------------------------------------------


class _StatusErr(Exception):
    def __init__(self, msg: str, status_code: int) -> None:
        super().__init__(msg)
        self.status_code = status_code


def test_classifier_uses_structured_status_code() -> None:
    from modules.extract.processing_strategy import classify_transient_error

    is_429, _, server = classify_transient_error("boom", _StatusErr("boom", 502))
    assert server is True
    is_429, _, server = classify_transient_error("boom", _StatusErr("boom", 429))
    assert is_429 is True
    is_429, is_timeout, server = classify_transient_error(
        "bad request", _StatusErr("bad request", 400)
    )
    assert not (is_429 or is_timeout or server)


def test_classifier_no_false_positive_on_stray_number() -> None:
    from modules.extract.processing_strategy import classify_transient_error

    # A stray 5xx-looking number with no status/http/error context must NOT
    # classify as a retryable server error (previously did, burning retries).
    _, _, server = classify_transient_error("Traceback: line 502 of file.py")
    assert server is False
    # But a real structured form still does.
    _, _, server = classify_transient_error("HTTP 503 Service Unavailable")
    assert server is True


# ---------------------------------------------------------------------------
# Item 11: batch temp-file content sniff
# ---------------------------------------------------------------------------


def test_is_batch_temp_file_discriminates(tmp_path: Path) -> None:
    from modules.batch.ops import is_batch_temp_file

    batch = tmp_path / "a_temp.jsonl"
    batch.write_text(
        json.dumps({"batch_request": {"custom_id": "a-chunk-1"}})
        + "\n"
        + json.dumps({"batch_tracking": {"batch_id": "b1"}})
        + "\n",
        encoding="utf-8",
    )
    assert is_batch_temp_file(batch) is True

    sync = tmp_path / "b_temp.jsonl"
    sync.write_text(
        json.dumps({"_chronominer_temp_version": 2})
        + "\n"
        + json.dumps({"custom_id": "b-chunk-1", "chunk_index": 1})
        + "\n",
        encoding="utf-8",
    )
    assert is_batch_temp_file(sync) is False

    readjust = tmp_path / "c_line_ranges_adjust_temp.jsonl"
    readjust.write_text(
        json.dumps({"ranges_fingerprint": "x", "total_ranges": 1}) + "\n",
        encoding="utf-8",
    )
    assert is_batch_temp_file(readjust) is False


# ---------------------------------------------------------------------------
# Item 14: cross-shape completed-index detection + batch resume parity
# ---------------------------------------------------------------------------


def test_completed_indices_from_outputs_reads_both_shapes(tmp_path: Path) -> None:
    from modules.extract.resume import completed_indices_from_outputs

    sync = tmp_path / "doc_output.json"
    sync.write_text(
        json.dumps(
            {"records": [{"custom_id": "doc-chunk-1"}, {"custom_id": "doc-chunk-2"}]}
        ),
        encoding="utf-8",
    )
    batch = tmp_path / "doc_final_output.json"
    batch.write_text(
        json.dumps({"responses": [{"custom_id": "doc-chunk-5"}]}),
        encoding="utf-8",
    )
    assert completed_indices_from_outputs(sync, batch) == {1, 2, 5}


@pytest.mark.asyncio
async def test_batch_resume_skips_completed_requests(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(ps, "supports_batch", lambda provider: True)

    captured: list[list[object]] = []

    class _Backend:
        max_batch_size = 50000
        max_batch_bytes = 10**9

        def submit_batch(self, requests, model_config, **_kw):  # type: ignore[no-untyped-def]
            captured.append(list(requests))
            return ps.BatchHandle(provider="openai", batch_id="b", metadata={})

    monkeypatch.setattr(ps, "get_batch_backend", lambda provider: _Backend())

    strat = ps.BatchProcessingStrategy()
    await strat.process_chunks(
        chunks=["c1", "c2", "c3"],
        handler=_StubHandler(),
        dev_message="dev",
        model_config={"extraction_model": {"provider": "openai", "name": "gpt-4o"}},
        schema={"type": "object"},
        file_path=tmp_path / "doc.txt",
        temp_jsonl_path=tmp_path / "doc_temp.jsonl",
        console_print=lambda *_a, **_k: None,
        completed_chunk_indices={1, 3},
    )
    # Only chunk 2 remains to submit.
    assert len(captured) == 1
    submitted_ids = [r.custom_id for r in captured[0]]
    assert submitted_ids == ["doc-chunk-2"]


@pytest.mark.asyncio
async def test_batch_resume_nothing_to_submit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(ps, "supports_batch", lambda provider: True)
    called = {"n": 0}

    class _Backend:
        max_batch_size = 50000
        max_batch_bytes = 10**9

        def submit_batch(self, requests, model_config, **_kw):  # type: ignore[no-untyped-def]
            called["n"] += 1
            return ps.BatchHandle(provider="openai", batch_id="b", metadata={})

    monkeypatch.setattr(ps, "get_batch_backend", lambda provider: _Backend())

    strat = ps.BatchProcessingStrategy()
    res = await strat.process_chunks(
        chunks=["c1", "c2"],
        handler=_StubHandler(),
        dev_message="dev",
        model_config={"extraction_model": {"provider": "openai", "name": "gpt-4o"}},
        schema={"type": "object"},
        file_path=tmp_path / "doc.txt",
        temp_jsonl_path=tmp_path / "doc_temp.jsonl",
        console_print=lambda *_a, **_k: None,
        completed_chunk_indices={1, 2},
    )
    assert res == []
    assert called["n"] == 0  # no batch submitted


# ---------------------------------------------------------------------------
# CLI agent contract: parser flags + mode override
# ---------------------------------------------------------------------------


def test_process_parser_has_cli_contract_flags() -> None:
    from main.cli_args import create_process_parser

    parser = create_process_parser()
    args = parser.parse_args(
        ["--schema", "S", "--input", "d/", "--json", "--dry-run", "--non-interactive"]
    )
    assert args.json_summary is True
    assert args.dry_run is True
    assert args.non_interactive is True


def test_check_batches_parser_has_json_flag() -> None:
    from main.cli_args import create_check_batches_parser

    parser = create_check_batches_parser()
    args = parser.parse_args(["--json"])
    assert args.json_summary is True


def test_mode_detector_honors_explicit_flags() -> None:
    from main.mode_detector import detect_execution_mode

    class _Loader:
        def get_paths_config(self):
            return {"general": {"interactive_mode": True}}

    with patch.object(sys, "argv", ["script.py", "--non-interactive"]):
        assert detect_execution_mode(_Loader()) is False
    with patch.object(sys, "argv", ["script.py", "--interactive"]):
        assert detect_execution_mode(_Loader()) is True


def test_chunk_slice_indices_matches_apply_chunk_slice() -> None:
    from modules.infra.chunking import (
        ChunkSlice,
        apply_chunk_slice,
        chunk_slice_indices,
    )

    chunks = [f"c{i}" for i in range(1, 11)]
    ranges = [(i, i) for i in range(1, 11)]
    for cs in (
        None,
        ChunkSlice(first_n=3),
        ChunkSlice(last_n=4),
        ChunkSlice(page_range=(2, 5)),
        ChunkSlice(first_n=99),  # exceeds total -> all
    ):
        idx = chunk_slice_indices(len(chunks), cs)
        sliced_chunks, _ = apply_chunk_slice(chunks, ranges, cs)
        assert len(idx) == len(sliced_chunks)
        # Indices point back to the exact retained chunks.
        assert [chunks[i - 1] for i in idx] == sliced_chunks
