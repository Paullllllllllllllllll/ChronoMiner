# tests/test_batch_output_merge.py

"""Regression tests for batch finalization/repair output merging.

Covers the data-loss bug where a ``--batch --resume`` finalization (or a
later ``repair_extractions`` run) rebuilt ``{stem}_output.json`` only from the
newly-retrieved responses and overwrote records completed on earlier runs.
``merge_existing_batch_output`` restores the sync-path merge semantics.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from modules.extract.batch_output import merge_existing_batch_output

pytestmark = pytest.mark.unit


def _record(stem: str, idx: int, text: str) -> dict:
    return {
        "custom_id": f"{stem}-chunk-{idx}",
        "chunk_index": idx,
        "chunk_range": None,
        "response": {"output_text": text, "response_data": {}},
    }


def _built(records: list[dict], *, total_chunks: int, fully_completed: bool) -> dict:
    return {
        "_chronominer_metadata": {
            "schema_name": "S",
            "total_chunks": total_chunks,
            "batch_tracking": {"fully_completed": fully_completed},
        },
        "records": list(records),
    }


def _write_existing(path: Path, records: list[dict], *, total_chunks: int) -> None:
    payload = {
        "_chronominer_metadata": {"schema_name": "S", "total_chunks": total_chunks},
        "records": records,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_merge_preserves_prior_records(tmp_path: Path) -> None:
    out = tmp_path / "doc_output.json"
    _write_existing(
        out,
        [_record("doc", 1, "one"), _record("doc", 2, "two")],
        total_chunks=4,
    )
    # Second run only retrieved chunks 3 and 4.
    built = _built(
        [_record("doc", 3, "three"), _record("doc", 4, "four")],
        total_chunks=4,
        fully_completed=True,
    )

    merged = merge_existing_batch_output(built, out)

    indices = sorted(r["chunk_index"] for r in merged["records"])
    assert indices == [1, 2, 3, 4], "prior records must survive finalization"
    assert merged["_chronominer_metadata"].get("partial") is not True


def test_merge_new_record_wins_on_conflict(tmp_path: Path) -> None:
    out = tmp_path / "doc_output.json"
    _write_existing(out, [_record("doc", 1, "stale")], total_chunks=2)
    built = _built(
        [_record("doc", 1, "fresh"), _record("doc", 2, "two")],
        total_chunks=2,
        fully_completed=True,
    )

    merged = merge_existing_batch_output(built, out)

    by_idx = {r["chunk_index"]: r for r in merged["records"]}
    assert by_idx[1]["response"]["output_text"] == "fresh"


def test_merge_no_existing_file_is_passthrough(tmp_path: Path) -> None:
    built = _built([_record("doc", 1, "one")], total_chunks=1, fully_completed=True)
    merged = merge_existing_batch_output(built, tmp_path / "missing.json")
    assert merged is built
    assert len(merged["records"]) == 1


def test_merge_drops_failed_chunk_now_present(tmp_path: Path) -> None:
    out = tmp_path / "doc_output.json"
    # A prior run completed chunk 2 (which the new run reports as failed).
    _write_existing(out, [_record("doc", 2, "prior-two")], total_chunks=3)
    built = _built(
        [_record("doc", 1, "one"), _record("doc", 3, "three")],
        total_chunks=3,
        fully_completed=True,
    )
    built["_chronominer_metadata"]["failed_chunks"] = [2]
    built["_chronominer_metadata"]["partial"] = True

    merged = merge_existing_batch_output(built, out)

    meta = merged["_chronominer_metadata"]
    assert "failed_chunks" not in meta, "chunk 2 now has a record; not failed"
    assert meta.get("partial") is not True
    assert sorted(r["chunk_index"] for r in merged["records"]) == [1, 2, 3]


def test_merge_recomputes_partial_when_still_incomplete(tmp_path: Path) -> None:
    out = tmp_path / "doc_output.json"
    _write_existing(out, [_record("doc", 1, "one")], total_chunks=3)
    built = _built(
        [_record("doc", 2, "two")],
        total_chunks=3,
        fully_completed=False,
    )

    merged = merge_existing_batch_output(built, out)

    meta = merged["_chronominer_metadata"]
    assert meta["total_chunks"] == 3
    assert meta.get("partial") is True


def test_merge_tolerates_corrupt_existing(tmp_path: Path) -> None:
    out = tmp_path / "doc_output.json"
    out.write_text("{not valid json", encoding="utf-8")
    built = _built([_record("doc", 1, "one")], total_chunks=1, fully_completed=True)

    merged = merge_existing_batch_output(built, out)

    assert [r["chunk_index"] for r in merged["records"]] == [1]
