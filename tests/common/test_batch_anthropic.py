"""Anthropic batch backend against a fake SDK client and real SDK models."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from anthropic.types.messages import MessageBatch, MessageBatchIndividualResponse

from chronominer.common.batch import (
    BatchHandle,
    BatchNotReadyError,
    BatchRequest,
    BatchStatus,
    BatchSubmitError,
)
from chronominer.common.batch_anthropic import AnthropicBatchBackend
from chronominer.common.usage import Usage


def _batch(batch_id: str, processing_status: str, **counts: int) -> MessageBatch:
    request_counts = {
        "processing": 0,
        "succeeded": 0,
        "errored": 0,
        "canceled": 0,
        "expired": 0,
        **counts,
    }
    return MessageBatch.model_validate(
        {
            "id": batch_id,
            "type": "message_batch",
            "processing_status": processing_status,
            "created_at": "2026-01-01T00:00:00Z",
            "expires_at": "2026-01-02T00:00:00Z",
            "archived_at": None,
            "cancel_initiated_at": None,
            "ended_at": None,
            "results_url": "https://results" if processing_status == "ended" else None,
            "request_counts": request_counts,
        }
    )


def _message(*blocks: dict[str, Any], stop_reason: str = "end_turn") -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-test",
        "content": list(blocks),
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 4,
            "cache_read_input_tokens": 50,
            "cache_creation_input_tokens": 20,
        },
    }


def _result(custom_id: str, result: dict[str, Any]) -> MessageBatchIndividualResponse:
    return MessageBatchIndividualResponse.model_validate(
        {"custom_id": custom_id, "result": result}
    )


class FakeMessageBatches:
    def __init__(self) -> None:
        self.records: dict[str, Any] = {}
        self.results_by_id: dict[str, list[Any]] = {}
        self.created: list[list[dict[str, Any]]] = []
        self.cancelled: list[str] = []
        self.create_error: Exception | None = None

    def create(self, *, requests: list[dict[str, Any]]) -> Any:
        if self.create_error is not None:
            raise self.create_error
        self.created.append(requests)
        batch_id = f"msgbatch_{len(self.created)}"
        self.records[batch_id] = _batch(batch_id, "in_progress")
        return self.records[batch_id]

    def retrieve(self, batch_id: str) -> Any:
        if batch_id not in self.records:
            raise LookupError(batch_id)
        return self.records[batch_id]

    def results(self, batch_id: str) -> Any:
        return iter(self.results_by_id.get(batch_id, []))

    def cancel(self, batch_id: str) -> None:
        if batch_id not in self.records:
            raise LookupError(batch_id)
        self.cancelled.append(batch_id)

    def list(self, *, limit: int) -> list[Any]:
        return list(self.records.values())


class FakeAnthropic:
    def __init__(self) -> None:
        self.messages = SimpleNamespace(batches=FakeMessageBatches())

    @property
    def batches(self) -> FakeMessageBatches:
        batches: FakeMessageBatches = self.messages.batches
        return batches


def _body(request: BatchRequest) -> dict[str, Any]:
    return {
        "model": "claude-test",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": request.text}],
    }


def _requests(n: int) -> list[BatchRequest]:
    return [BatchRequest(custom_id=f"p{i}", text=f"page {i}") for i in range(1, n + 1)]


@pytest.fixture
def client() -> FakeAnthropic:
    return FakeAnthropic()


class TestSubmit:
    def test_splits_by_count(self, client: FakeAnthropic) -> None:
        handles = AnthropicBatchBackend(client, max_requests=2).submit(
            _requests(3), _body
        )
        assert [h.custom_ids for h in handles] == [["p1", "p2"], ["p3"]]
        assert [h.batch_id for h in handles] == ["msgbatch_1", "msgbatch_2"]
        assert handles[1].metadata == {"request_count": 1}
        assert client.batches.created[0][0] == {
            "custom_id": "p1",
            "params": _body(_requests(1)[0]),
        }

    def test_failed_part(self, client: FakeAnthropic) -> None:
        client.batches.create_error = RuntimeError("overloaded")
        with pytest.raises(BatchSubmitError) as raised:
            AnthropicBatchBackend(client).submit(_requests(2), _body)
        assert raised.value.handles == []


class TestStatus:
    @pytest.mark.parametrize(
        ("processing_status", "counts", "expected"),
        [
            ("in_progress", {"processing": 2, "succeeded": 1}, BatchStatus.IN_PROGRESS),
            ("canceling", {"processing": 2}, BatchStatus.IN_PROGRESS),
            ("ended", {"succeeded": 2, "errored": 1}, BatchStatus.COMPLETED),
            ("ended", {"errored": 3}, BatchStatus.FAILED),
            ("ended", {"canceled": 3}, BatchStatus.CANCELLED),
            ("ended", {"expired": 3}, BatchStatus.EXPIRED),
        ],
    )
    def test_status_mapping(
        self,
        client: FakeAnthropic,
        processing_status: str,
        counts: dict[str, int],
        expected: BatchStatus,
    ) -> None:
        client.batches.records["b1"] = _batch("b1", processing_status, **counts)
        info = AnthropicBatchBackend(client).status(BatchHandle("anthropic", "b1"))
        assert info.status is expected
        assert info.provider_status == processing_status
        assert info.total_requests == sum(counts.values())
        assert info.results_available is (processing_status == "ended")

    def test_ended_without_counts_is_unknown(self, client: FakeAnthropic) -> None:
        client.batches.records["b1"] = SimpleNamespace(
            processing_status="ended", request_counts=None, results_url=None
        )
        info = AnthropicBatchBackend(client).status(BatchHandle("anthropic", "b1"))
        assert info.status is BatchStatus.UNKNOWN
        assert "no request counts" in (info.error_message or "")

    def test_counts(self, client: FakeAnthropic) -> None:
        client.batches.records["b1"] = _batch(
            "b1", "in_progress", processing=4, succeeded=3, errored=1
        )
        info = AnthropicBatchBackend(client).status(BatchHandle("anthropic", "b1"))
        assert (info.completed_requests, info.failed_requests) == (3, 1)
        assert info.pending_requests == 4


class TestResults:
    def _ended(self, client: FakeAnthropic, *results: Any) -> BatchHandle:
        client.batches.records["b1"] = _batch("b1", "ended", succeeded=1, errored=1)
        client.batches.results_by_id["b1"] = list(results)
        return BatchHandle("anthropic", "b1", custom_ids=["a", "b", "c", "d", "e"])

    def test_result_kinds_and_missing_ids(self, client: FakeAnthropic) -> None:
        handle = self._ended(
            client,
            _result(
                "a",
                {
                    "type": "succeeded",
                    "message": _message(
                        {"type": "thinking", "thinking": "...", "signature": "s"},
                        {"type": "text", "text": "Hello "},
                        {"type": "text", "text": "world"},
                    ),
                },
            ),
            _result(
                "b",
                {
                    "type": "errored",
                    "error": {
                        "type": "error",
                        "error": {"type": "invalid_request_error", "message": "bad"},
                    },
                },
            ),
            _result("c", {"type": "canceled"}),
            _result("d", {"type": "expired"}),
        )
        items = AnthropicBatchBackend(client).results(handle)
        by_id = {item.custom_id: item for item in items}
        assert [item.custom_id for item in items] == ["a", "b", "c", "d", "e"]
        assert by_id["a"].success is True
        assert by_id["a"].text == "Hello world"
        assert by_id["a"].usage == Usage(80, 4, 84, 50, 0)
        json.dumps(by_id["a"].response)
        assert by_id["a"].response["message"]["model"] == "claude-test"
        assert (by_id["b"].error, by_id["b"].error_code) == (
            "bad",
            "invalid_request_error",
        )
        assert by_id["c"].error_code == "canceled"
        assert by_id["d"].error_code == "expired"
        assert (by_id["e"].origin, by_id["e"].success) == ("missing", False)

    def test_message_without_text_fails(self, client: FakeAnthropic) -> None:
        handle = self._ended(
            client,
            _result(
                "a",
                {
                    "type": "succeeded",
                    "message": _message(
                        {"type": "thinking", "thinking": "...", "signature": "s"},
                        stop_reason="max_tokens",
                    ),
                },
            ),
        )
        item = AnthropicBatchBackend(client).results(handle)[0]
        assert item.success is False
        assert "max_tokens" in (item.error or "")

    def test_refusal_fails_and_truncation_is_marked(
        self, client: FakeAnthropic
    ) -> None:
        text = {"type": "text", "text": "Some text"}
        handle = self._ended(
            client,
            _result(
                "a",
                {"type": "succeeded", "message": _message(text, stop_reason="refusal")},
            ),
            _result(
                "b",
                {
                    "type": "succeeded",
                    "message": _message(text, stop_reason="max_tokens"),
                },
            ),
        )
        refused, cut = AnthropicBatchBackend(client).results(handle)[:2]
        assert (refused.success, refused.error_code) == (False, "refusal")
        assert refused.error == "Model refused the request: stop_reason=refusal"
        assert (cut.success, cut.outcome.kind) == (False, "truncated")
        assert cut.error_code == "truncated"

    def test_succeeded_without_message_fails(self, client: FakeAnthropic) -> None:
        result = SimpleNamespace(
            custom_id="a", result=SimpleNamespace(type="succeeded", message=None)
        )
        handle = self._ended(client, result)
        item = AnthropicBatchBackend(client).results(handle)[0]
        assert item.success is False
        assert item.error

    def test_running_batch_is_not_ready(self, client: FakeAnthropic) -> None:
        client.batches.records["b1"] = _batch("b1", "in_progress", processing=1)
        with pytest.raises(BatchNotReadyError):
            AnthropicBatchBackend(client).results(BatchHandle("anthropic", "b1"))


class TestCancelAndList:
    def test_cancel(self, client: FakeAnthropic) -> None:
        client.batches.records["b1"] = _batch("b1", "in_progress", processing=1)
        backend = AnthropicBatchBackend(client)
        assert backend.cancel(BatchHandle("anthropic", "b1")) is True
        assert backend.cancel(BatchHandle("anthropic", "nope")) is False
        assert client.batches.cancelled == ["b1"]

    def test_list_active(self, client: FakeAnthropic) -> None:
        client.batches.records["b1"] = _batch("b1", "in_progress", processing=1)
        client.batches.records["b2"] = _batch("b2", "ended", succeeded=1)
        client.batches.records["b3"] = _batch("b3", "canceling", processing=1)
        active = AnthropicBatchBackend(client).list_active()
        assert [(s.batch_id, s.status, s.provider_status) for s in active] == [
            ("b1", BatchStatus.IN_PROGRESS, "in_progress"),
            ("b3", BatchStatus.IN_PROGRESS, "canceling"),
        ]

    def test_cleanup_is_a_no_op(self, client: FakeAnthropic) -> None:
        AnthropicBatchBackend(client).cleanup(BatchHandle("anthropic", "b1"))
