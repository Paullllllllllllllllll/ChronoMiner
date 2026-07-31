"""Regression tests for ``main.check_batches.process_all_batches``.

Covers the following finalization bugs:

1. A group containing completed AND terminally failed/expired batches was
   counted as "still processing" (the in-progress arithmetic included
   terminal failures), so the partial-output branch was never reached and
   completed, paid-for results were stranded forever.
2. The output identifier was derived with ``str.replace("_temp", "")``,
   which also strips internal ``_temp`` substrings (``oven_temperature`` ->
   ``ovenerature``), misfiling the finalized output under a garbage name.
3. Under ``retain_temporary_jsonl: true`` an already-finalized group was
   re-scanned on every later run: the remote provider files are deleted on
   finalization, so the re-download 404s and the group counted as failed
   forever (exit 1 for good).
4. A transient status-poll failure was read as "batch not found" and folded
   into a premature partial finalization.
5. ``order_index`` values are absolute document indices; re-basing them per
   part double-shifted the ordering of every part after the first.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from main.check_batches import process_all_batches
from modules.batch.backends import BatchStatus, BatchStatusInfo


def _write_temp_file(path, stem, batch_ids):
    lines = []
    for i in (1, 2):
        lines.append(
            json.dumps(
                {
                    "batch_request": {
                        "custom_id": f"{stem}-chunk-{i}",
                        "order_index": i,
                        "metadata": {"chunk_index": i, "total_chunks": 2},
                    }
                }
            )
        )
    for batch_id in batch_ids:
        lines.append(
            json.dumps({"batch_tracking": {"batch_id": batch_id, "provider": "openai"}})
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mock_backend(status_by_id):
    backend = MagicMock()

    def _get_status(handle):
        return BatchStatusInfo(status=status_by_id[handle.batch_id])

    backend.get_status.side_effect = _get_status
    return backend


@pytest.mark.unit
class TestPartialFinalization:
    def test_completed_plus_expired_writes_partial_output(self, tmp_path):
        """One COMPLETED + one EXPIRED batch must finalize partially, not
        report 'still processing' forever."""
        stem = "oven_temperature"
        temp_file = tmp_path / f"{stem}_temp.jsonl"
        _write_temp_file(temp_file, stem, ["b1", "b2"])

        backend = _mock_backend(
            {"b1": BatchStatus.COMPLETED, "b2": BatchStatus.EXPIRED}
        )
        responses = [{"custom_id": f"{stem}-chunk-1", "response": '{"entries": []}'}]
        agg: dict[str, int] = {}

        with (
            patch("main.check_batches.get_batch_backend", return_value=backend),
            patch(
                "main.check_batches.retrieve_responses_from_batch",
                return_value=responses,
            ),
            patch("main.check_batches.get_schema_handler", return_value=MagicMock()),
        ):
            process_all_batches(
                root_folder=tmp_path,
                processing_settings={"retain_temporary_jsonl": True},
                schema_name="TestSchema",
                schema_config={},
                ui=None,
                agg=agg,
            )

        # Regression 1: the group is finalized as failed/partial, not pending.
        assert agg.get("pending", 0) == 0
        assert agg.get("failed", 0) == 1

        # Regression 2: the output keeps the full stem (str.replace would
        # have produced ovenerature_output.json).
        final_path = tmp_path / f"{stem}_output.json"
        assert final_path.exists()
        assert not (tmp_path / "ovenerature_output.json").exists()

        data = json.loads(final_path.read_text(encoding="utf-8"))
        assert data["_chronominer_metadata"]["partial"] is True
        assert len(data["records"]) == 1

    def test_in_progress_batch_still_defers(self, tmp_path):
        """A genuinely in-progress batch must keep deferring finalization."""
        stem = "doc"
        temp_file = tmp_path / f"{stem}_temp.jsonl"
        _write_temp_file(temp_file, stem, ["b1", "b2"])

        backend = _mock_backend(
            {"b1": BatchStatus.COMPLETED, "b2": BatchStatus.IN_PROGRESS}
        )
        agg: dict[str, int] = {}

        with (
            patch("main.check_batches.get_batch_backend", return_value=backend),
            patch(
                "main.check_batches.retrieve_responses_from_batch",
                return_value=[],
            ),
            patch("main.check_batches.get_schema_handler", return_value=MagicMock()),
        ):
            process_all_batches(
                root_folder=tmp_path,
                processing_settings={"retain_temporary_jsonl": True},
                schema_name="TestSchema",
                schema_config={},
                ui=None,
                agg=agg,
            )

        assert agg.get("pending", 0) == 1
        assert not (tmp_path / f"{stem}_output.json").exists()


def _write_temp_file_indices(path, stem, indices, batch_ids, total=None):
    """Write a batch temp file with explicit absolute chunk indices."""
    total = total if total is not None else len(indices)
    lines = []
    for i in indices:
        lines.append(
            json.dumps(
                {
                    "batch_request": {
                        "custom_id": f"{stem}-chunk-{i}",
                        "order_index": i,
                        "metadata": {"chunk_index": i, "total_chunks": total},
                    }
                }
            )
        )
    for batch_id in batch_ids:
        lines.append(
            json.dumps({"batch_tracking": {"batch_id": batch_id, "provider": "openai"}})
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_final_output(path, stem, indices, *, fully_completed, partial):
    meta = {
        "schema_name": "TestSchema",
        "total_chunks": len(indices),
        "batch_tracking": {"fully_completed": fully_completed},
    }
    if partial:
        meta["partial"] = True
    payload = {
        "_chronominer_metadata": meta,
        "records": [
            {
                "custom_id": f"{stem}-chunk-{i}",
                "chunk_index": i,
                "response": {"output_text": "{}", "response_data": {}},
            }
            for i in indices
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.unit
class TestAlreadyFinalizedGroups:
    """Retained temp files must not be re-finalized (B1).

    With retain_temporary_jsonl: true the temp JSONL survives finalization
    while the remote provider files are deleted, so a re-scan 404s on download
    and marks the group failed on every subsequent run.
    """

    def test_finalized_group_is_skipped(self, tmp_path):
        stem = "doc"
        _write_temp_file(tmp_path / f"{stem}_temp.jsonl", stem, ["b1"])
        _write_final_output(
            tmp_path / f"{stem}_output.json",
            stem,
            [1, 2],
            fully_completed=True,
            partial=False,
        )

        backend = _mock_backend({"b1": BatchStatus.COMPLETED})
        retrieve = MagicMock(return_value=[])
        agg: dict[str, int] = {}

        with (
            patch("main.check_batches.get_batch_backend", return_value=backend),
            patch("main.check_batches.retrieve_responses_from_batch", retrieve),
            patch("main.check_batches.get_schema_handler", return_value=MagicMock()),
        ):
            process_all_batches(
                root_folder=tmp_path,
                processing_settings={"retain_temporary_jsonl": True},
                schema_name="TestSchema",
                schema_config={},
                ui=None,
                agg=agg,
            )

        assert backend.get_status.call_count == 0
        assert retrieve.call_count == 0
        assert agg.get("failed", 0) == 0
        assert agg.get("pending", 0) == 0
        assert agg.get("finalized", 0) == 1

    def test_partial_output_is_still_reprocessed(self, tmp_path):
        """A partial finalization must remain eligible for a top-up."""
        stem = "doc"
        _write_temp_file(tmp_path / f"{stem}_temp.jsonl", stem, ["b1"])
        _write_final_output(
            tmp_path / f"{stem}_output.json",
            stem,
            [1],
            fully_completed=False,
            partial=True,
        )

        backend = _mock_backend({"b1": BatchStatus.COMPLETED})
        responses = [{"custom_id": f"{stem}-chunk-2", "response": '{"entries": []}'}]

        with (
            patch("main.check_batches.get_batch_backend", return_value=backend),
            patch(
                "main.check_batches.retrieve_responses_from_batch",
                return_value=responses,
            ),
            patch("main.check_batches.get_schema_handler", return_value=MagicMock()),
        ):
            process_all_batches(
                root_folder=tmp_path,
                processing_settings={"retain_temporary_jsonl": True},
                schema_name="TestSchema",
                schema_config={},
                ui=None,
                agg={},
            )

        assert backend.get_status.call_count == 1
        data = json.loads(
            (tmp_path / f"{stem}_output.json").read_text(encoding="utf-8")
        )
        assert sorted(r["chunk_index"] for r in data["records"]) == [1, 2]


class _FlakyBackend:
    """Backend whose status poll fails ``failures`` times before succeeding."""

    def __init__(self, failures, final_status=BatchStatus.COMPLETED, error="boom"):
        self.remaining = failures
        self.final_status = final_status
        self.error = error
        self.calls = 0

    def get_status(self, handle):
        self.calls += 1
        if self.remaining > 0:
            self.remaining -= 1
            raise RuntimeError(self.error)
        return BatchStatusInfo(status=self.final_status)

    def cleanup(self, handle):
        return None


@pytest.mark.unit
class TestTransientStatusErrors:
    """A network blip during get_status must not trigger a premature partial
    finalization (B4)."""

    def test_transient_error_is_retried_then_succeeds(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "main.check_batches._STATUS_RETRY_DELAY_SECONDS", 0, raising=False
        )
        stem = "doc"
        _write_temp_file(tmp_path / f"{stem}_temp.jsonl", stem, ["b1"])
        backend = _FlakyBackend(failures=1, error="Connection reset by peer")
        responses = [
            {"custom_id": f"{stem}-chunk-{i}", "response": '{"entries": []}'}
            for i in (1, 2)
        ]
        agg: dict[str, int] = {}

        with (
            patch("main.check_batches.get_batch_backend", return_value=backend),
            patch(
                "main.check_batches.retrieve_responses_from_batch",
                return_value=responses,
            ),
            patch("main.check_batches.get_schema_handler", return_value=MagicMock()),
        ):
            process_all_batches(
                root_folder=tmp_path,
                processing_settings={"retain_temporary_jsonl": True},
                schema_name="TestSchema",
                schema_config={},
                ui=None,
                agg=agg,
            )

        assert backend.calls == 2
        assert agg.get("finalized", 0) == 1
        assert agg.get("failed", 0) == 0
        data = json.loads(
            (tmp_path / f"{stem}_output.json").read_text(encoding="utf-8")
        )
        assert data["_chronominer_metadata"].get("partial") is not True

    def test_persistent_transient_error_defers_as_pending(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "main.check_batches._STATUS_RETRY_DELAY_SECONDS", 0, raising=False
        )
        stem = "doc"
        _write_temp_file(tmp_path / f"{stem}_temp.jsonl", stem, ["b1"])
        backend = _FlakyBackend(failures=5, error="Connection reset by peer")
        agg: dict[str, int] = {}

        with (
            patch("main.check_batches.get_batch_backend", return_value=backend),
            patch(
                "main.check_batches.retrieve_responses_from_batch",
                return_value=[],
            ),
            patch("main.check_batches.get_schema_handler", return_value=MagicMock()),
        ):
            process_all_batches(
                root_folder=tmp_path,
                processing_settings={"retain_temporary_jsonl": True},
                schema_name="TestSchema",
                schema_config={},
                ui=None,
                agg=agg,
            )

        assert backend.calls == 2
        assert agg.get("pending", 0) == 1
        assert agg.get("failed", 0) == 0
        assert not (tmp_path / f"{stem}_output.json").exists()

    def test_explicit_not_found_still_counts_as_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "main.check_batches._STATUS_RETRY_DELAY_SECONDS", 0, raising=False
        )
        stem = "doc"
        _write_temp_file(tmp_path / f"{stem}_temp.jsonl", stem, ["b1"])
        backend = _FlakyBackend(failures=5, error="Error code: 404 - No such batch: b1")
        agg: dict[str, int] = {}

        with (
            patch("main.check_batches.get_batch_backend", return_value=backend),
            patch(
                "main.check_batches.retrieve_responses_from_batch",
                return_value=[],
            ),
            patch("main.check_batches.get_schema_handler", return_value=MagicMock()),
        ):
            process_all_batches(
                root_folder=tmp_path,
                processing_settings={"retain_temporary_jsonl": True},
                schema_name="TestSchema",
                schema_config={},
                ui=None,
                agg=agg,
            )

        assert agg.get("pending", 0) == 0
        # An explicit not-found keeps the missing-batch path (the group is
        # counted failed, not deferred as pending).
        assert agg.get("failed", 0) >= 1


@pytest.mark.unit
class TestMultiPartOrdering:
    """order_index values are absolute; parts must not be re-based (P4)."""

    def test_split_parts_are_ordered_by_absolute_index(self, tmp_path):
        stem = "doc"
        # Part 1 holds chunks 1 and 3, part 2 the gap-filling chunk 2: a
        # per-part offset would re-base chunk 2 behind chunk 3.
        _write_temp_file_indices(
            tmp_path / f"{stem}_temp_part1.jsonl", stem, [1, 3], ["b1"], total=3
        )
        _write_temp_file_indices(
            tmp_path / f"{stem}_temp_part2.jsonl", stem, [2], ["b2"], total=3
        )

        backend = _mock_backend(
            {"b1": BatchStatus.COMPLETED, "b2": BatchStatus.COMPLETED}
        )
        by_batch = {
            "b1": [
                {"custom_id": f"{stem}-chunk-1", "response": '{"a": 1}'},
                {"custom_id": f"{stem}-chunk-3", "response": '{"a": 3}'},
            ],
            "b2": [{"custom_id": f"{stem}-chunk-2", "response": '{"a": 2}'}],
        }

        with (
            patch("main.check_batches.get_batch_backend", return_value=backend),
            patch(
                "main.check_batches.retrieve_responses_from_batch",
                side_effect=lambda track: by_batch[track["batch_id"]],
            ),
            patch("main.check_batches.get_schema_handler", return_value=MagicMock()),
        ):
            process_all_batches(
                root_folder=tmp_path,
                processing_settings={"retain_temporary_jsonl": True},
                schema_name="TestSchema",
                schema_config={},
                ui=None,
                agg={},
            )

        data = json.loads(
            (tmp_path / f"{stem}_output.json").read_text(encoding="utf-8")
        )
        assert [r["chunk_index"] for r in data["records"]] == [1, 2, 3]
