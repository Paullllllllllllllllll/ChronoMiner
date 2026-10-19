"""OpenAI batch backend against a fake SDK client."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from chronominer.common.batch import (
    BatchHandle,
    BatchNotReadyError,
    BatchRequest,
    BatchStatus,
    BatchSubmitError,
)
from chronominer.common.batch_openai import (
    ENDPOINT,
    OpenAIBatchBackend,
    coerce_file_id,
)
from chronominer.common.usage import Usage


class FakeFiles:
    def __init__(self) -> None:
        self.uploads: dict[str, bytes] = {}
        self.contents: dict[str, bytes] = {}
        self.deleted: list[str] = []
        self.delete_error: Exception | None = None

    def create(self, *, file: tuple[str, bytes], purpose: str) -> Any:
        assert purpose == "batch"
        file_id = f"file-{len(self.uploads) + 1}"
        self.uploads[file_id] = file[1]
        return SimpleNamespace(id=file_id)

    def content(self, file_id: str) -> Any:
        data = self.contents[file_id]
        return SimpleNamespace(read=lambda: data)

    def delete(self, file_id: str) -> None:
        if self.delete_error is not None:
            raise self.delete_error
        self.deleted.append(file_id)


class FakeBatches:
    def __init__(self) -> None:
        self.records: dict[str, Any] = {}
        self.created: list[dict[str, Any]] = []
        self.cancelled: list[str] = []
        self.create_error: Exception | None = None
        self.list_limits: list[int] = []

    def create(self, **kwargs: Any) -> Any:
        if self.create_error is not None:
            raise self.create_error
        self.created.append(kwargs)
        batch = SimpleNamespace(id=f"batch_{len(self.created)}", status="validating")
        self.records[batch.id] = batch
        return batch

    def retrieve(self, batch_id: str) -> Any:
        if batch_id not in self.records:
            raise LookupError(f"no batch {batch_id}")
        return self.records[batch_id]

    def cancel(self, batch_id: str) -> None:
        if batch_id not in self.records:
            raise LookupError(batch_id)
        self.cancelled.append(batch_id)

    def list(self, *, limit: int) -> list[Any]:
        self.list_limits.append(limit)
        return list(self.records.values())


class FakeOpenAI:
    def __init__(self) -> None:
        self.files = FakeFiles()
        self.batches = FakeBatches()


def _body(request: BatchRequest) -> dict[str, Any]:
    return {"model": "gpt-test", "input": request.text, **request.metadata}


def _requests(n: int) -> list[BatchRequest]:
    return [BatchRequest(custom_id=f"p{i}", text=f"page {i}") for i in range(1, n + 1)]


def _lines(data: bytes) -> list[dict[str, Any]]:
    return [json.loads(line) for line in data.decode("utf-8").splitlines()]


def _success(custom_id: str, *texts: str, **body: Any) -> dict[str, Any]:
    content = [{"type": "output_text", "text": text} for text in texts]
    return {
        "custom_id": custom_id,
        "response": {
            "status_code": 200,
            "body": {"output": [{"type": "message", "content": content}], **body},
        },
        "error": None,
    }


def _jsonl(*records: Any) -> bytes:
    return "".join(
        (r if isinstance(r, str) else json.dumps(r)) + "\n" for r in records
    ).encode("utf-8")


@pytest.fixture
def client() -> FakeOpenAI:
    return FakeOpenAI()


def _finished(client: FakeOpenAI, batch_id: str, **fields: Any) -> BatchHandle:
    client.batches.records[batch_id] = SimpleNamespace(
        id=batch_id, status=fields.pop("status", "completed"), **fields
    )
    return BatchHandle("openai", batch_id, custom_ids=["p1", "p2", "p3"])


class TestSubmit:
    def test_splits_by_count_and_uploads_jsonl(self, client: FakeOpenAI) -> None:
        backend = OpenAIBatchBackend(client, max_requests=2)
        handles = backend.submit(_requests(5), _body, label="run")
        assert [h.custom_ids for h in handles] == [["p1", "p2"], ["p3", "p4"], ["p5"]]
        assert [h.batch_id for h in handles] == ["batch_1", "batch_2", "batch_3"]
        assert handles[0].metadata == {"input_file_id": "file-1", "request_count": 2}
        first = _lines(client.files.uploads["file-1"])
        assert first[0] == {
            "custom_id": "p1",
            "method": "POST",
            "url": ENDPOINT,
            "body": {"model": "gpt-test", "input": "page 1"},
        }
        assert client.batches.created[0] == {
            "input_file_id": "file-1",
            "endpoint": ENDPOINT,
            "completion_window": "24h",
            "metadata": {"description": "run"},
        }

    def test_splits_by_bytes(self, client: FakeOpenAI) -> None:
        backend = OpenAIBatchBackend(client, max_bytes=200)
        handles = backend.submit(_requests(3), _body)
        assert [len(h.custom_ids) for h in handles] == [1, 1, 1]
        for handle in handles:
            data = client.files.uploads[handle.metadata["input_file_id"]]
            assert len(data) <= 200

    def test_service_tier_rules(self, client: FakeOpenAI) -> None:
        backend = OpenAIBatchBackend(client)
        requests = [
            BatchRequest("a", metadata={"service_tier": "flex"}),
            BatchRequest("b", metadata={"service_tier": "turbo"}),
            BatchRequest("c", metadata={"service_tier": "priority"}),
        ]
        backend.submit(requests, _body)
        bodies = [line["body"] for line in _lines(client.files.uploads["file-1"])]
        assert bodies[0]["service_tier"] == "auto"
        assert "service_tier" not in bodies[1]
        assert bodies[2]["service_tier"] == "priority"

    def test_failed_create_removes_upload(self, client: FakeOpenAI) -> None:
        backend = OpenAIBatchBackend(client, max_requests=2)
        submitted: list[BatchHandle] = []

        def stop_after_first(handle: BatchHandle) -> None:
            submitted.append(handle)
            client.batches.create_error = RuntimeError("overloaded")

        with pytest.raises(BatchSubmitError) as raised:
            backend.submit(_requests(4), _body, on_submitted=stop_after_first)
        assert raised.value.handles == submitted
        assert client.files.deleted == ["file-2"]


class TestStatus:
    @pytest.mark.parametrize(
        ("provider_status", "expected"),
        [
            ("validating", BatchStatus.PENDING),
            ("in_progress", BatchStatus.IN_PROGRESS),
            ("finalizing", BatchStatus.IN_PROGRESS),
            ("cancelling", BatchStatus.IN_PROGRESS),
            ("completed", BatchStatus.COMPLETED),
            ("failed", BatchStatus.FAILED),
            ("cancelled", BatchStatus.CANCELLED),
            ("expired", BatchStatus.EXPIRED),
            ("archived", BatchStatus.UNKNOWN),
        ],
    )
    def test_status_mapping(
        self, client: FakeOpenAI, provider_status: str, expected: BatchStatus
    ) -> None:
        handle = _finished(client, "b1", status=provider_status)
        info = OpenAIBatchBackend(client).status(handle)
        assert info.status is expected
        assert info.provider_status == provider_status
        assert (info.error_message is not None) is (expected is BatchStatus.UNKNOWN)

    def test_counts_and_results_available(self, client: FakeOpenAI) -> None:
        handle = _finished(
            client,
            "b1",
            status="expired",
            output_file_id="out-1",
            request_counts=SimpleNamespace(total=5, completed=3, failed=1),
        )
        info = OpenAIBatchBackend(client).status(handle)
        assert (info.total_requests, info.completed_requests) == (5, 3)
        assert (info.failed_requests, info.pending_requests) == (1, 1)
        assert info.results_available is True
        assert info.output_file_id == "out-1"

    def test_error_file_alone_makes_results_available(self, client: FakeOpenAI) -> None:
        handle = _finished(client, "b1", error_file_id="err-1")
        assert OpenAIBatchBackend(client).status(handle).results_available is True

    def test_running_batch_has_no_results(self, client: FakeOpenAI) -> None:
        handle = _finished(client, "b1", status="in_progress", output_file_id="out-1")
        assert OpenAIBatchBackend(client).status(handle).results_available is False

    def test_failed_batch_reports_errors(self, client: FakeOpenAI) -> None:
        errors = SimpleNamespace(
            data=[SimpleNamespace(code="invalid_request", message="bad model", line=3)]
        )
        handle = _finished(client, "b1", status="failed", errors=errors)
        info = OpenAIBatchBackend(client).status(handle)
        assert info.error_message == "bad model (code=invalid_request) [line=3]"
        assert OpenAIBatchBackend(client).diagnose_failure(handle) == info.error_message

    def test_lookup_failure_raises(self, client: FakeOpenAI) -> None:
        with pytest.raises(LookupError, match="nope"):
            OpenAIBatchBackend(client).status(BatchHandle("openai", "nope"))


class TestResults:
    def test_output_error_file_and_missing_ids(self, client: FakeOpenAI) -> None:
        handle = _finished(client, "b1", output_file_id="out-1", error_file_id="err-1")
        usage = {
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 120,
            "input_tokens_details": {"cached_tokens": 40},
            "output_tokens_details": {"reasoning_tokens": 5},
        }
        client.files.contents["out-1"] = _jsonl(
            _success("p1", '{"a": 1', ", ", '"b": 2}', usage=usage),
            "not json",
        )
        client.files.contents["err-1"] = _jsonl(
            {
                "custom_id": "p2",
                "response": {
                    "status_code": 400,
                    "body": {"error": {"message": "image too large", "type": "bad"}},
                },
                "error": None,
            }
        )
        items = OpenAIBatchBackend(client).results(handle)
        assert [(i.custom_id, i.success, i.origin) for i in items] == [
            ("p1", True, "output"),
            ("p2", False, "error_file"),
            ("p3", False, "missing"),
            ("", False, "unmatched"),
        ]
        assert (items[3].raw, items[3].raw_kind) == ("not json", "line")
        assert "line 2" in (items[3].error or "")
        assert items[0].text == '{"a": 1, "b": 2}'
        assert items[0].usage == Usage(100, 20, 120, 40, 5)
        assert items[0].response["status_code"] == 200
        assert (items[1].error, items[1].error_code) == ("image too large", "bad")
        assert items[2].error_code == "missing"

    def test_top_level_error_line(self, client: FakeOpenAI) -> None:
        handle = _finished(client, "b1", error_file_id="err-1")
        client.files.contents["err-1"] = _jsonl(
            {
                "custom_id": "p1",
                "response": None,
                "error": {"code": "batch_expired", "message": "expired"},
            }
        )
        item = OpenAIBatchBackend(client).results(handle)[0]
        assert (item.error, item.error_code) == ("expired", "batch_expired")
        assert item.response == {
            "error": {"code": "batch_expired", "message": "expired"}
        }

    def test_non_200_in_output_file(self, client: FakeOpenAI) -> None:
        handle = _finished(client, "b1", output_file_id="out-1")
        client.files.contents["out-1"] = _jsonl(
            {"custom_id": "p1", "response": {"status_code": 500, "body": {}}}
        )
        item = OpenAIBatchBackend(client).results(handle)[0]
        assert (item.success, item.error) == (False, "HTTP 500")

    def test_response_without_text_fails(self, client: FakeOpenAI) -> None:
        handle = _finished(client, "b1", output_file_id="out-1")
        record = {
            "custom_id": "p1",
            "response": {
                "status_code": 200,
                "body": {"status": "incomplete", "output": [{"type": "reasoning"}]},
            },
        }
        client.files.contents["out-1"] = _jsonl(record)
        item = OpenAIBatchBackend(client).results(handle)[0]
        assert item.success is False
        assert "incomplete" in (item.error or "")

    def test_outcomes_as_a_synchronous_answer_reads_them(
        self, client: FakeOpenAI
    ) -> None:
        handle = _finished(client, "b1", output_file_id="out-1")
        refused = _success("p3")
        refused["response"]["body"]["output"][0]["content"] = [
            {"type": "refusal", "refusal": "No."}
        ]
        client.files.contents["out-1"] = _jsonl(
            _success(
                "p1",
                "cut",
                status="incomplete",
                incomplete_details={"reason": "max_output_tokens"},
            ),
            _success(
                "p2",
                "partial",
                status="incomplete",
                incomplete_details={"reason": "content_filter"},
            ),
            refused,
        )
        items = OpenAIBatchBackend(client).results(handle)
        assert [(i.success, i.outcome.kind, i.error_code) for i in items] == [
            (False, "truncated", "truncated"),
            (False, "content_filter", "content_filter"),
            (False, "refusal", "refusal"),
        ]
        assert items[0].text == "cut"
        assert items[2].error == "Model refused the request: No."

    def test_running_batch_is_not_ready(self, client: FakeOpenAI) -> None:
        handle = _finished(client, "b1", status="in_progress")
        with pytest.raises(BatchNotReadyError):
            OpenAIBatchBackend(client).results(handle)

    def test_finished_batch_without_files(self, client: FakeOpenAI) -> None:
        errors = SimpleNamespace(data=[SimpleNamespace(message="no access")])
        handle = _finished(client, "b1", status="failed", errors=errors)
        items = OpenAIBatchBackend(client).results(handle)
        assert [i.origin for i in items] == ["missing"] * 3
        assert {i.error for i in items} == {"no access"}

    def test_restored_handle_gives_the_same_results(self, client: FakeOpenAI) -> None:
        handle = _finished(client, "b1", output_file_id="out-1")
        client.files.contents["out-1"] = _jsonl(_success("p1", "text"))
        backend = OpenAIBatchBackend(client)
        restored = BatchHandle.from_dict(json.loads(json.dumps(handle.to_dict())))
        assert backend.results(restored) == backend.results(handle)


class TestCancelListCleanup:
    def test_cancel(self, client: FakeOpenAI) -> None:
        handle = _finished(client, "b1", status="in_progress")
        backend = OpenAIBatchBackend(client)
        assert backend.cancel(handle) is True
        assert client.batches.cancelled == ["b1"]
        assert backend.cancel(BatchHandle("openai", "missing")) is False

    def test_list_active_skips_terminal(self, client: FakeOpenAI) -> None:
        for batch_id, status in [
            ("b1", "in_progress"),
            ("b2", "completed"),
            ("b3", "validating"),
            ("b4", "cancelled"),
        ]:
            _finished(client, batch_id, status=status)
        active = OpenAIBatchBackend(client).list_active(page_size=50)
        assert [(s.batch_id, s.status) for s in active] == [
            ("b1", BatchStatus.IN_PROGRESS),
            ("b3", BatchStatus.PENDING),
        ]
        assert client.batches.list_limits == [50]

    def test_cleanup_deletes_all_remote_files(self, client: FakeOpenAI) -> None:
        handle = _finished(client, "b1", output_file_id="out-1", error_file_id="err-1")
        handle.metadata["input_file_id"] = "in-1"
        OpenAIBatchBackend(client).cleanup(handle)
        assert client.files.deleted == ["in-1", "out-1", "err-1"]

    def test_cleanup_never_raises(self, client: FakeOpenAI) -> None:
        handle = BatchHandle("openai", "gone", metadata={"input_file_id": "in-1"})
        client.files.delete_error = RuntimeError("delete failed")
        OpenAIBatchBackend(client).cleanup(handle)


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [
        ("file-1", "file-1"),
        ("", None),
        ({"id": "file-2"}, "file-2"),
        ({"file_id": "file-3"}, "file-3"),
        (["file-4"], "file-4"),
        ([{"id": "file-5"}], "file-5"),
        ([], None),
        (None, None),
    ],
)
def test_coerce_file_id(candidate: Any, expected: str | None) -> None:
    assert coerce_file_id(candidate) == expected
