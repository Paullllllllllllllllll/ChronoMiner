"""Regression tests for ``main.check_batches.process_all_batches``.

Covers two finalization bugs:

1. A group containing completed AND terminally failed/expired batches was
   counted as "still processing" (the in-progress arithmetic included
   terminal failures), so the partial-output branch was never reached and
   completed, paid-for results were stranded forever.
2. The output identifier was derived with ``str.replace("_temp", "")``,
   which also strips internal ``_temp`` substrings (``oven_temperature`` ->
   ``ovenerature``), misfiling the finalized output under a garbage name.
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
