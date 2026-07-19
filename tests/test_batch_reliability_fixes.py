"""Regression tests for the batch-reliability hardening pass.

Each test targets one verified failure mode in the batch submit/check/repair
pipeline. All provider clients and backends are mocked; nothing hits the
network.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from main.check_batches import process_all_batches
from main.repair_extractions import _discover_candidate_temp_files
from modules.batch.backends import BatchHandle, BatchStatus, BatchStatusInfo
from modules.batch.backends.base import BatchResultItem
from modules.batch.ops import (
    _group_temp_files_by_base,
    retrieve_responses_from_batch,
)
from modules.extract.batch_output import _to_unified_record
from modules.extract.processing_strategy import (
    BatchProcessingStrategy,
    _partition_batch_requests,
)
from modules.llm.openai_sdk_utils import list_all_batches


def _make_backend(*, max_bytes: int = 3000, submit_side_effect=None) -> MagicMock:
    backend = MagicMock()
    backend.max_batch_size = 50000
    backend.max_batch_bytes = max_bytes
    if submit_side_effect is not None:
        backend.submit_batch.side_effect = submit_side_effect
    return backend


def _run_batch_submit(backend, tmp_path):
    strategy = BatchProcessingStrategy()
    file_path = tmp_path / "doc.txt"
    temp_jsonl = tmp_path / "doc_temp.jsonl"
    with patch(
        "modules.extract.processing_strategy.get_batch_backend",
        return_value=backend,
    ):
        return asyncio.run(
            strategy.process_chunks(
                chunks=["chunk one text", "chunk two text"],
                handler=MagicMock(schema_name="TestSchema"),
                dev_message="SYS",
                model_config={
                    "extraction_model": {"provider": "openai", "name": "gpt-4o"}
                },
                schema={"type": "object"},
                file_path=file_path,
                temp_jsonl_path=temp_jsonl,
                console_print=lambda *a, **k: None,
            )
        )


# --- Item 1: orphaned batch ids -------------------------------------------


@pytest.mark.unit
def test_debug_artifact_written_before_crash_on_second_part(tmp_path):
    """A crash on the second part must still leave the recovery artifact with
    the first part's batch id and metadata on disk."""
    handle1 = BatchHandle(
        provider="openai", batch_id="batch-1", metadata={"input_file_id": "f1"}
    )
    backend = _make_backend(
        submit_side_effect=[handle1, RuntimeError("boom on part 2")]
    )

    with pytest.raises(RuntimeError):
        _run_batch_submit(backend, tmp_path)

    artifact = tmp_path / "doc_batch_submission_debug.json"
    assert artifact.exists(), "artifact must exist even though part 2 crashed"
    data = json.loads(artifact.read_text(encoding="utf-8"))
    assert data["batch_ids"] == ["batch-1"]
    assert data["provider"] == "openai"
    assert data["batch_metadata"] == {"batch-1": {"input_file_id": "f1"}}


# --- Item 3: stale part cleanup on resubmission ---------------------------


@pytest.mark.unit
def test_stale_part_files_removed_before_resubmission(tmp_path):
    """A prior submission's stale part and base temp files must be cleared so
    check_batches does not group them with the fresh parts."""
    stale_part = tmp_path / "doc_temp_part5.jsonl"
    stale_part.write_text("stale\n", encoding="utf-8")
    stale_base = tmp_path / "doc_temp.jsonl"
    stale_base.write_text("stale\n", encoding="utf-8")

    handles = [
        BatchHandle(provider="openai", batch_id="batch-1"),
        BatchHandle(provider="openai", batch_id="batch-2"),
    ]
    backend = _make_backend(submit_side_effect=handles)

    _run_batch_submit(backend, tmp_path)

    assert not stale_part.exists(), "stale _part5 must be removed"
    # Multi-part run never rewrites the base file, so the stale base is gone.
    assert not stale_base.exists()
    assert (tmp_path / "doc_temp_part1.jsonl").exists()
    assert (tmp_path / "doc_temp_part2.jsonl").exists()


# --- Item 4: repair discovers and groups part files -----------------------


def _write_part(path, stem, indices, batch_id):
    lines = []
    for i in indices:
        lines.append(
            json.dumps(
                {
                    "batch_request": {
                        "custom_id": f"{stem}-chunk-{i}",
                        "order_index": i,
                        "metadata": {"chunk_index": i, "total_chunks": 4},
                    }
                }
            )
        )
    lines.append(
        json.dumps({"batch_tracking": {"batch_id": batch_id, "provider": "openai"}})
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.mark.unit
def test_repair_discovers_and_groups_multipart(tmp_path):
    _write_part(tmp_path / "doc_temp_part1.jsonl", "doc", [1, 2], "batch-1")
    _write_part(tmp_path / "doc_temp_part2.jsonl", "doc", [3, 4], "batch-2")

    ui = MagicMock()
    candidates = _discover_candidate_temp_files([("TestSchema", tmp_path, {})], ui)

    assert len(candidates) == 1
    cand = candidates[0]
    assert cand["identifier"] == "doc"
    assert len(cand["temp_files"]) == 2
    assert cand["tracking_count"] == 2  # one tracking record per part


# --- Item 5: cross-directory group collision ------------------------------


@pytest.mark.unit
def test_group_keys_on_parent_dir(tmp_path):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    file_a = dir_a / "doc_temp.jsonl"
    file_b = dir_b / "doc_temp.jsonl"
    file_a.write_text("", encoding="utf-8")
    file_b.write_text("", encoding="utf-8")

    groups = _group_temp_files_by_base([file_a, file_b])
    # Two distinct groups despite identical stems.
    assert len(groups) == 2
    assert (dir_a, "doc_temp") in groups
    assert (dir_b, "doc_temp") in groups


# --- Item 6: unknown-status wedge -----------------------------------------


def _write_temp_file(path, stem, batch_ids, idx_range=(1, 2)):
    lines = []
    for i in range(idx_range[0], idx_range[1] + 1):
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


@pytest.mark.unit
def test_unknown_status_with_error_treated_as_missing(tmp_path):
    """A batch that returns UNKNOWN + error_message (aged out / deleted) must
    finalize partially, not report 'still processing' forever."""
    stem = "doc"
    _write_temp_file(tmp_path / f"{stem}_temp.jsonl", stem, ["b1", "b2"])

    infos = {
        "b1": BatchStatusInfo(status=BatchStatus.COMPLETED),
        "b2": BatchStatusInfo(
            status=BatchStatus.UNKNOWN, error_message="batch not found"
        ),
    }
    backend = MagicMock()
    backend.get_status.side_effect = lambda h: infos[h.batch_id]
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

    assert agg.get("pending", 0) == 0
    assert agg.get("failed", 0) == 1
    assert (tmp_path / f"{stem}_output.json").exists()


# --- Item 13: id-less tracking record wedge -------------------------------


@pytest.mark.unit
def test_idless_tracking_record_does_not_wedge_group(tmp_path):
    """An extra tracking record with no batch_id must not defer the group
    forever: with one completed and one expired batch it must finalize."""
    stem = "doc"
    path = tmp_path / f"{stem}_temp.jsonl"
    lines = [
        json.dumps(
            {
                "batch_request": {
                    "custom_id": f"{stem}-chunk-1",
                    "order_index": 1,
                    "metadata": {"chunk_index": 1, "total_chunks": 2},
                }
            }
        ),
        json.dumps({"batch_tracking": {"batch_id": "b1", "provider": "openai"}}),
        json.dumps({"batch_tracking": {"batch_id": "b2", "provider": "openai"}}),
        # id-less tracking record: previously counted in in_progress arithmetic.
        json.dumps({"batch_tracking": {"provider": "openai"}}),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    infos = {
        "b1": BatchStatusInfo(status=BatchStatus.COMPLETED),
        "b2": BatchStatusInfo(status=BatchStatus.EXPIRED),
    }
    backend = MagicMock()
    backend.get_status.side_effect = lambda h: infos[h.batch_id]
    backend.diagnose_failure.return_value = "expired"
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

    assert agg.get("pending", 0) == 0
    assert (tmp_path / f"{stem}_output.json").exists()


# --- Item 8: results_available for terminal OpenAI batches ----------------


@pytest.mark.unit
@patch("openai.OpenAI")
def test_openai_results_available_for_expired_with_output(mock_openai_class):
    from modules.batch.backends import clear_backend_cache, get_batch_backend

    clear_backend_cache()
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client

    mock_batch = MagicMock()
    mock_batch.status = "expired"
    mock_batch.output_file_id = "out-1"
    mock_batch.request_counts = MagicMock(total=2, completed=1, failed=1)
    mock_client.batches.retrieve.return_value = mock_batch

    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        backend = get_batch_backend("openai")
        info = backend.get_status(BatchHandle(provider="openai", batch_id="b1"))

    assert info.status == BatchStatus.EXPIRED
    assert info.results_available is True


@pytest.mark.unit
@patch("openai.OpenAI")
def test_openai_results_unavailable_without_output(mock_openai_class):
    from modules.batch.backends import clear_backend_cache, get_batch_backend

    clear_backend_cache()
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client

    mock_batch = MagicMock()
    mock_batch.status = "in_progress"
    mock_batch.output_file_id = None
    mock_batch.request_counts = MagicMock(total=2, completed=0, failed=0)
    mock_client.batches.retrieve.return_value = mock_batch

    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        backend = get_batch_backend("openai")
        info = backend.get_status(BatchHandle(provider="openai", batch_id="b1"))

    assert info.results_available is False


# --- Item 9: partition includes per-request prompt overhead ---------------


@pytest.mark.unit
def test_partition_accounts_for_prompt_overhead():
    from modules.batch.backends import BatchRequest

    reqs = [
        BatchRequest(custom_id="a", text="x"),
        BatchRequest(custom_id="b", text="x"),
    ]
    # Without overhead both requests fit one part.
    assert len(_partition_batch_requests(reqs, 50000, 4200)) == 1
    # A large per-request overhead forces a split into two parts.
    assert len(_partition_batch_requests(reqs, 50000, 4200, 200)) == 2


# --- Item 10: token usage carried through ---------------------------------


@pytest.mark.unit
def test_usage_propagated_from_backend_to_record():
    item = BatchResultItem(
        custom_id="doc-chunk-1",
        success=True,
        content='{"entries": []}',
        input_tokens=10,
        output_tokens=5,
    )
    backend = MagicMock()
    backend.download_results.return_value = iter([item])

    with patch("modules.batch.ops.get_batch_backend", return_value=backend):
        responses = retrieve_responses_from_batch(
            {"batch_id": "b1", "provider": "openai"}, None, {}
        )

    assert responses[0]["usage"] == {"input_tokens": 10, "output_tokens": 5}

    # _to_unified_record prefers the entry-level usage dict.
    record = _to_unified_record(responses[0], {})
    assert record["response"]["response_data"]["usage"] == {
        "input_tokens": 10,
        "output_tokens": 5,
    }


# --- Item 15: list_all_batches dict-page pagination -----------------------


@pytest.mark.unit
def test_list_all_batches_paginates_dict_pages():
    """Dict-shaped pages must advance past page 1 (getattr(dict, 'has_more')
    would silently return the default and stop early)."""
    mock_client = MagicMock()
    page1 = {"data": [], "has_more": True, "last_id": "b1"}
    page2 = {"data": [], "has_more": False, "last_id": None}
    mock_client.batches.list.side_effect = [page1, page2]

    list_all_batches(mock_client, limit=100)

    assert mock_client.batches.list.call_count == 2
    # The second call must advance the cursor to the first page's last_id.
    assert mock_client.batches.list.call_args_list[1].kwargs.get("after") == "b1"


@pytest.mark.unit
def test_list_all_batches_breaks_on_non_advancing_cursor():
    """A page whose last_id equals the current cursor must terminate the loop
    instead of looping forever."""
    mock_client = MagicMock()
    page1 = MagicMock()
    page1.data = []
    page1.has_more = True
    page1.last_id = "b1"
    page2 = MagicMock()
    page2.data = []
    page2.has_more = True
    page2.last_id = "b1"  # same cursor -> must break
    mock_client.batches.list.side_effect = [page1, page2]

    list_all_batches(mock_client, limit=100)

    assert mock_client.batches.list.call_count == 2
