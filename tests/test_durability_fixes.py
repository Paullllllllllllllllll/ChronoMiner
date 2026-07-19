# tests/test_durability_fixes.py

"""Regression tests for the output-durability and resume fixes.

Covers six confirmed bugs:

* FIX 1 -- atomic final-output writes (``atomic_write_json``).
* FIX 2 -- a sliced ``--resume`` run must not be reported COMPLETE by
  cardinality when none of the slice's indices are actually done.
* FIX 3 -- a resumable temp JSONL left by a hard-killed run must be
  recovered (its indices unioned into the skip set) instead of truncated.
* FIX 4 -- ``build_unified_batch_output`` deduplicates by ``custom_id``.
* FIX 5 -- a failed final-output write must not delete the temp JSONL.
* FIX 6 -- ``load_line_ranges`` tolerates a UTF-8 BOM on the first line.
"""

from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path

import pytest

from modules.extract.resume import build_temp_header


class DummyHandler:
    schema_name = "TestSchema"


class RecordingResumeStrategy:
    """Respects the skip set and append semantics like the real strategy.

    Records the ``completed_chunk_indices`` seen on the first call so tests can
    assert what the orchestration decided to skip. Writes one temp record per
    not-yet-completed absolute index, in append mode when a skip set is present.
    """

    def __init__(self) -> None:
        self.first_completed: set[int] | None = None
        self.calls = 0

    async def process_chunks(
        self,
        *,
        chunks,
        file_path,
        temp_jsonl_path,
        completed_chunk_indices=None,
        chunk_indices=None,
        **_kwargs,
    ):
        completed = set(completed_chunk_indices or ())
        if self.first_completed is None:
            self.first_completed = set(completed)
        self.calls += 1
        abs_indices = (
            list(chunk_indices)
            if chunk_indices is not None
            else list(range(1, len(chunks) + 1))
        )
        mode = "a" if completed else "w"
        results = []
        with temp_jsonl_path.open(mode, encoding="utf-8") as f:
            for pos, _chunk in enumerate(chunks):
                idx = abs_indices[pos]
                if idx in completed:
                    continue
                f.write(
                    json.dumps(
                        {
                            "custom_id": f"{file_path.stem}-chunk-{idx}",
                            "chunk_index": idx,
                            "chunk_range": [idx, idx],
                            "response": {"body": {"output_text": "ok"}},
                        }
                    )
                    + "\n"
                )
                results.append({"ok": True})
        return results


def _make_processor(config_loader, *, retain_temp: bool = True):
    from modules.extract.file_processor import FileProcessor

    paths_config = copy.deepcopy(config_loader.get_paths_config())
    paths_config.setdefault("general", {})["retain_temporary_jsonl"] = retain_temp
    return FileProcessor(
        paths_config=paths_config,
        model_config=config_loader.get_model_config(),
        chunking_config={"chunking": {"default_tokens_per_chunk": 10}},
        concurrency_config=config_loader.get_concurrency_config(),
    )


def _patch(monkeypatch, strategy):
    monkeypatch.setattr(
        "modules.extract.file_processor.create_processing_strategy",
        lambda use_batch, concurrency_config=None: strategy,
    )
    monkeypatch.setattr(
        "modules.extract.file_processor.get_schema_handler",
        lambda schema_name: DummyHandler(),
    )


def _run_process(fp, input_file, schema_paths, **kwargs):
    async def _run():
        return await fp.process_file(
            file_path=input_file,
            use_batch=False,
            selected_schema={"schema": {"type": "object"}},
            prompt_template="Schema={{TRANSCRIPTION_SCHEMA}}",
            schema_name="TestSchema",
            inject_schema=True,
            schema_paths=schema_paths,
            global_chunking_method="auto",
            ui=None,
            **kwargs,
        )

    return asyncio.run(_run())


# --------------------------------------------------------------------------- #
# FIX 1 -- atomic_write_json
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_atomic_write_json_writes_and_leaves_no_tmp(tmp_path: Path) -> None:
    from modules.infra.jsonl import atomic_write_json

    dest = tmp_path / "out.json"
    atomic_write_json(dest, {"a": 1, "b": "two"})

    assert json.loads(dest.read_text(encoding="utf-8")) == {"a": 1, "b": "two"}
    # No stray temp siblings survive a successful write.
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.unit
def test_atomic_write_json_preserves_prior_file_on_failure(tmp_path: Path) -> None:
    from modules.infra.jsonl import atomic_write_json

    dest = tmp_path / "out.json"
    dest.write_text(json.dumps({"kept": True}), encoding="utf-8")

    # object() is not JSON-serializable: the write fails before any replace().
    with pytest.raises(TypeError):
        atomic_write_json(dest, {"bad": object()})

    # The previously merged file is untouched, and no temp sibling is left.
    assert json.loads(dest.read_text(encoding="utf-8")) == {"kept": True}
    assert list(tmp_path.glob("*.tmp")) == []


# --------------------------------------------------------------------------- #
# FIX 2 -- sliced resume must not be falsely COMPLETE
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_sliced_resume_not_falsely_complete(tmp_path, config_loader, monkeypatch):
    """A prior --page-range run left records 5..7 in the output. A re-run with
    --resume --first-n-chunks 2 must process chunks 1 and 2, not skip the file
    because len(completed) >= len(sliced chunks)."""
    from modules.infra.chunking import ChunkSlice

    strategy = RecordingResumeStrategy()
    _patch(monkeypatch, strategy)

    input_file = tmp_path / "sliced.txt"
    input_file.write_text("\n".join(f"line {i}" for i in range(200)), encoding="utf-8")

    schema_paths = config_loader.get_schemas_paths()["TestSchema"]
    output_dir = Path(schema_paths["output"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prior output holds high absolute indices from a different slice; their
    # cardinality (3) exceeds the 2 chunks this run selects.
    prior = {
        "_chronominer_metadata": {"schema_name": "TestSchema", "total_chunks": 90},
        "records": [
            {"custom_id": "sliced-chunk-5", "chunk_index": 5, "response": {}},
            {"custom_id": "sliced-chunk-6", "chunk_index": 6, "response": {}},
            {"custom_id": "sliced-chunk-7", "chunk_index": 7, "response": {}},
        ],
    }
    (output_dir / "sliced_output.json").write_text(json.dumps(prior), encoding="utf-8")

    fp = _make_processor(config_loader)
    status = _run_process(
        fp,
        input_file,
        schema_paths,
        resume=True,
        chunk_slice=ChunkSlice(first_n=2),
    )

    assert status != "skipped", "membership, not cardinality, must gate the skip"
    assert strategy.calls >= 1
    # The strategy was asked to process chunks 1 and 2 (neither already done).
    assert strategy.first_completed == {5, 6, 7}

    data = json.loads((output_dir / "sliced_output.json").read_text(encoding="utf-8"))
    indices = {rec["chunk_index"] for rec in data["records"]}
    assert {1, 2}.issubset(indices), "the newly-selected chunks must be present"
    # Prior records survive the resume merge.
    assert {5, 6, 7}.issubset(indices)


# --------------------------------------------------------------------------- #
# FIX 3 -- recover a resumable temp left by a hard-killed run
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_resume_recovers_temp_jsonl_after_hard_crash(
    tmp_path, config_loader, monkeypatch
):
    """Run 1 durably appended chunk records to the temp JSONL, then the process
    was hard-killed before the output JSON was written. A --resume run must
    recover (skip) those indices instead of re-processing and truncating them."""
    strategy = RecordingResumeStrategy()
    _patch(monkeypatch, strategy)

    input_file = tmp_path / "crashed.txt"
    input_file.write_text("\n".join(f"line {i}" for i in range(200)), encoding="utf-8")

    schema_paths = config_loader.get_schemas_paths()["TestSchema"]
    output_dir = Path(schema_paths["output"])
    temp_dir = output_dir / "temp_jsonl"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / "crashed_temp.jsonl"

    # A resumable temp with records for chunks 1 and 2 and NO output JSON.
    lines = [json.dumps(build_temp_header())]
    for idx in (1, 2):
        lines.append(
            json.dumps(
                {
                    "custom_id": f"crashed-chunk-{idx}",
                    "chunk_index": idx,
                    "chunk_range": [idx, idx],
                    "response": {"body": {"output_text": "from-run-1"}},
                }
            )
        )
    temp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    fp = _make_processor(config_loader)
    status = _run_process(fp, input_file, schema_paths, resume=True)

    assert status in ("complete", "partial")
    # The indices already in the temp were recovered into the skip set, so the
    # strategy was told to skip them rather than re-processing (and truncating).
    assert strategy.first_completed is not None
    assert {1, 2}.issubset(strategy.first_completed)

    data = json.loads((output_dir / "crashed_output.json").read_text(encoding="utf-8"))
    indices = {rec["chunk_index"] for rec in data["records"]}
    assert {1, 2}.issubset(indices), "run-1 records must survive to the output"


# --------------------------------------------------------------------------- #
# FIX 4 -- batch output dedup by custom_id
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_build_unified_batch_output_dedups_custom_id() -> None:
    from modules.extract.batch_output import build_unified_batch_output

    # Temp record first, fresher download for the same id appended after it.
    responses = [
        {"custom_id": "doc-chunk-1", "response": "stale"},
        {"custom_id": "doc-chunk-2", "response": "two"},
        {"custom_id": "doc-chunk-1", "response": "fresh"},
    ]
    out = build_unified_batch_output(
        responses,
        tracking=[],
        schema_name="TestSchema",
    )

    records = out["records"]
    ids = [r["custom_id"] for r in records]
    assert ids.count("doc-chunk-1") == 1, "duplicate custom_id must collapse"
    # Last occurrence wins: the freshest text survives.
    chunk1 = next(r for r in records if r["custom_id"] == "doc-chunk-1")
    assert chunk1["response"]["output_text"] == "fresh"
    # total_chunks is not inflated by the duplicate.
    assert out["_chronominer_metadata"]["total_chunks"] == 2


# --------------------------------------------------------------------------- #
# FIX 5 -- a failed output write must keep the temp JSONL
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_cleanup_keeps_temp_when_output_not_written(tmp_path, config_loader) -> None:
    from modules.extract.file_processor import _MessagingAdapter

    fp = _make_processor(config_loader, retain_temp=False)
    temp_path = tmp_path / "doc_temp.jsonl"
    temp_path.write_text("records", encoding="utf-8")

    # wrote_output=False: the temp is the only copy and must be kept even though
    # retain_temporary_jsonl is False.
    fp._cleanup_temp_files(False, temp_path, _MessagingAdapter(), wrote_output=False)
    assert temp_path.exists()

    # wrote_output=True with retain False: the temp is safe to delete.
    fp._cleanup_temp_files(False, temp_path, _MessagingAdapter(), wrote_output=True)
    assert not temp_path.exists()


@pytest.mark.integration
def test_failed_output_write_preserves_temp(tmp_path, config_loader, monkeypatch):
    """When the final JSON write raises, wrote_output stays False and the temp
    JSONL (the run's only copy) survives even with retain disabled."""

    class _WritingStrategy:
        async def process_chunks(
            self, *, chunks, file_path, temp_jsonl_path, **_kwargs
        ):
            with temp_jsonl_path.open("w", encoding="utf-8") as f:
                for idx, _chunk in enumerate(chunks, 1):
                    f.write(
                        json.dumps(
                            {
                                "custom_id": f"{file_path.stem}-chunk-{idx}",
                                "chunk_index": idx,
                                "response": {"body": {"output_text": "ok"}},
                            }
                        )
                        + "\n"
                    )
            return [{"ok": True} for _ in chunks]

    _patch(monkeypatch, _WritingStrategy())

    def _boom(_path, _data):
        raise OSError("disk full")

    monkeypatch.setattr("modules.extract.file_processor._write_output_json", _boom)

    input_file = tmp_path / "keepme.txt"
    input_file.write_text("hello\nworld\n", encoding="utf-8")

    schema_paths = config_loader.get_schemas_paths()["TestSchema"]
    fp = _make_processor(config_loader, retain_temp=False)
    _run_process(fp, input_file, schema_paths)

    output_dir = Path(schema_paths["output"])
    temp_path = output_dir / "temp_jsonl" / "keepme_temp.jsonl"
    assert temp_path.exists(), "temp must survive a failed output write"
    assert not (output_dir / "keepme_output.json").exists()


# --------------------------------------------------------------------------- #
# FIX 6 -- load_line_ranges tolerates a UTF-8 BOM
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_load_line_ranges_tolerates_bom(tmp_path: Path) -> None:
    from modules.infra.chunking import load_line_ranges

    ranges_file = tmp_path / "doc_line_ranges.txt"
    # UTF-8 BOM (as written by Windows Notepad "UTF-8 with BOM").
    ranges_file.write_text("(1, 100)\n(101, 200)\n", encoding="utf-8-sig")

    ranges = load_line_ranges(ranges_file)
    assert ranges == [(1, 100), (101, 200)], "the first range must not be dropped"
