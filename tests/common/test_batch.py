"""Shared batch types, request splitting, the factory and client constructors."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from chronominer.common import batch as batch_module
from chronominer.common.batch import (
    BatchBackend,
    BatchHandle,
    BatchRequest,
    BatchResultItem,
    BatchStatus,
    BatchStatusInfo,
    BatchSubmitError,
    BatchSummary,
    as_plain_dict,
    create_backend,
    create_client,
    split_by_limits,
    supports_batch,
)
from chronominer.common.batch_anthropic import AnthropicBatchBackend
from chronominer.common.batch_google import GoogleBatchBackend
from chronominer.common.batch_openai import OpenAIBatchBackend


class ScriptedBackend(BatchBackend):
    """Backend whose parts, records and status come from the test."""

    provider_name = "scripted"
    default_max_requests = 10
    default_max_bytes = 10_000

    def __init__(self, client: Any = None, **kwargs: Any) -> None:
        super().__init__(client, **kwargs)
        self.parts: list[list[dict[str, Any]]] = []
        self.fail_on_part: int | None = None
        self.records: list[BatchResultItem] = []
        self.batch_error: str | None = None
        self.cancel_error: Exception | None = None
        self.info = BatchStatusInfo(status=BatchStatus.FAILED)

    def _wrap(self, request: BatchRequest, body: dict[str, Any]) -> dict[str, Any]:
        return {"id": request.custom_id, "body": body}

    def _submit_part(self, entries: list[dict[str, Any]], *, label: str) -> BatchHandle:
        if self.fail_on_part == len(self.parts) + 1:
            raise RuntimeError("upload refused")
        self.parts.append(entries)
        return BatchHandle(
            provider=self.provider_name,
            batch_id=f"{label}-{len(self.parts)}",
            custom_ids=[entry["id"] for entry in entries],
        )

    def status(self, handle: BatchHandle) -> BatchStatusInfo:
        return self.info

    def list_active(self, *, page_size: int = 100) -> list[BatchSummary]:
        return []

    def _fetch_results(
        self, handle: BatchHandle
    ) -> tuple[list[BatchResultItem], str | None]:
        return list(self.records), self.batch_error

    def _cancel(self, handle: BatchHandle) -> None:
        if self.cancel_error is not None:
            raise self.cancel_error


def _requests(n: int, text: str = "x") -> list[BatchRequest]:
    return [BatchRequest(custom_id=f"r{i}", text=text) for i in range(1, n + 1)]


def _body(request: BatchRequest) -> dict[str, Any]:
    return {"input": request.text}


class TestTypes:
    def test_handle_round_trip(self) -> None:
        handle = BatchHandle(
            provider="openai",
            batch_id="batch_1",
            custom_ids=["a", "b"],
            metadata={"input_file_id": "file-1", "request_count": 2},
        )
        data = json.loads(json.dumps(handle.to_dict()))
        assert BatchHandle.from_dict(data) == handle

    def test_handle_from_partial_dict(self) -> None:
        restored = BatchHandle.from_dict({"batch_id": "b"})
        assert restored == BatchHandle(provider="", batch_id="b")

    def test_summary_handle(self) -> None:
        summary = BatchSummary("google", "batches/1", BatchStatus.PENDING)
        assert summary.handle() == BatchHandle(provider="google", batch_id="batches/1")

    def test_request_is_visual(self) -> None:
        assert BatchRequest(custom_id="a", text="t").is_visual is False
        image = BatchRequest(custom_id="b", image_base64="AA", mime_type="image/png")
        assert image.is_visual is True

    @pytest.mark.parametrize(
        ("status", "terminal"),
        [
            (BatchStatus.PENDING, False),
            (BatchStatus.IN_PROGRESS, False),
            (BatchStatus.UNKNOWN, False),
            (BatchStatus.COMPLETED, True),
            (BatchStatus.FAILED, True),
            (BatchStatus.CANCELLED, True),
            (BatchStatus.EXPIRED, True),
        ],
    )
    def test_status_info_terminal(self, status: BatchStatus, terminal: bool) -> None:
        assert BatchStatusInfo(status=status).is_terminal is terminal


class TestAsPlainDict:
    def test_pydantic_model(self) -> None:
        from anthropic.types import Usage as AnthropicUsage

        dumped = as_plain_dict(AnthropicUsage(input_tokens=1, output_tokens=2))
        assert dumped["input_tokens"] == 1
        json.dumps(dumped)

    def test_mapping_and_to_dict(self) -> None:
        assert as_plain_dict({"a": 1}) == {"a": 1}
        assert as_plain_dict(SimpleNamespace(to_dict=lambda: {"b": 2})) == {"b": 2}

    def test_unknown_object_falls_back_to_repr(self) -> None:
        assert as_plain_dict(object())["repr"].startswith("<object")


class TestSplitByLimits:
    @pytest.mark.parametrize(
        ("items", "sizes", "max_count", "expected"),
        [
            ([1, 2, 3, 4, 5], [1] * 5, 2, [[1, 2], [3, 4], [5]]),
            (["a", "b", "c"], [40, 40, 40], 9, [["a", "b"], ["c"]]),
            (["a", "big", "b"], [10, 500, 10], 9, [["a"], ["big"], ["b"]]),
            ([], [], 1, []),
        ],
        ids=["count", "bytes", "oversized-item-alone", "empty"],
    )
    def test_split(
        self,
        items: list[Any],
        sizes: list[int],
        max_count: int,
        expected: list[list[Any]],
    ) -> None:
        parts = split_by_limits(items, sizes, max_count=max_count, max_bytes=100)
        assert parts == expected

    def test_length_mismatch(self) -> None:
        with pytest.raises(ValueError):
            split_by_limits([1, 2], [1], max_count=1, max_bytes=1)


class TestSubmitAndResults:
    def test_submit_splits_and_reports_each_part(self) -> None:
        backend = ScriptedBackend(max_requests=2)
        seen: list[str] = []
        handles = backend.submit(
            _requests(5),
            _body,
            label="run",
            on_submitted=lambda h: seen.append(h.batch_id),
        )
        assert [h.custom_ids for h in handles] == [["r1", "r2"], ["r3", "r4"], ["r5"]]
        assert seen == ["run-1", "run-2", "run-3"]
        assert backend.parts[0][0] == {"id": "r1", "body": {"input": "x"}}

    def test_submit_splits_by_serialized_bytes(self) -> None:
        backend = ScriptedBackend(max_bytes=250)
        handles = backend.submit(_requests(3, text="y" * 100), _body)
        assert [len(h.custom_ids) for h in handles] == [1, 1, 1]

    def test_failed_part_carries_submitted_handles(self) -> None:
        backend = ScriptedBackend(max_requests=1)
        backend.fail_on_part = 2
        with pytest.raises(BatchSubmitError) as raised:
            backend.submit(_requests(3), _body)
        assert [h.batch_id for h in raised.value.handles] == ["batch-1"]
        assert "part 2 of 3" in str(raised.value)

    def test_duplicate_custom_ids_rejected(self) -> None:
        backend = ScriptedBackend()
        duplicate = [BatchRequest(custom_id="a"), BatchRequest(custom_id="a")]
        with pytest.raises(ValueError, match="unique"):
            backend.submit(duplicate, _body)

    def test_empty_submission(self) -> None:
        assert ScriptedBackend().submit([], _body) == []

    def test_results_dedupe_and_fill_missing_ids(self) -> None:
        backend = ScriptedBackend()
        backend.records = [
            BatchResultItem(custom_id="a", success=True, text="one"),
            BatchResultItem(custom_id="a", error="later", origin="error_file"),
            BatchResultItem(custom_id="", success=True, text="orphan"),
        ]
        handle = BatchHandle("scripted", "b1", custom_ids=["a", "b"])
        items = backend.results(handle)
        assert [(i.custom_id, i.success, i.origin) for i in items] == [
            ("a", True, "output"),
            ("b", False, "missing"),
        ]
        assert items[1].error == batch_module.MISSING_RESULT_ERROR
        assert items[1].error_code == "missing"

    def test_missing_items_carry_the_batch_error(self) -> None:
        backend = ScriptedBackend()
        backend.batch_error = "batch expired"
        items = backend.results(BatchHandle("scripted", "b1", custom_ids=["a"]))
        assert items[0].error == "batch expired"

    def test_cancel_reports_refusal(self) -> None:
        backend = ScriptedBackend()
        assert backend.cancel(BatchHandle("scripted", "b1")) is True
        backend.cancel_error = RuntimeError("gone")
        assert backend.cancel(BatchHandle("scripted", "b1")) is False

    def test_request_cancel_raises_the_provider_error(self) -> None:
        backend = ScriptedBackend()
        backend.request_cancel(BatchHandle("scripted", "b1"))
        backend.cancel_error = RuntimeError("gone")
        with pytest.raises(RuntimeError, match="gone"):
            backend.request_cancel(BatchHandle("scripted", "b1"))

    def test_diagnose_failure(self) -> None:
        backend = ScriptedBackend()
        handle = BatchHandle("scripted", "b1")
        assert backend.diagnose_failure(handle) == "Batch b1 has status failed"
        backend.info = BatchStatusInfo(status=BatchStatus.FAILED, error_message="bad")
        assert backend.diagnose_failure(handle) == "bad"


class TestFactory:
    @pytest.mark.parametrize(
        ("provider", "backend_class"),
        [
            ("openai", OpenAIBatchBackend),
            ("Anthropic", AnthropicBatchBackend),
            (" google ", GoogleBatchBackend),
        ],
    )
    def test_create_backend(
        self, provider: str, backend_class: type[BatchBackend]
    ) -> None:
        client = object()
        backend = create_backend(provider, client, max_requests=7, max_bytes=99)
        assert isinstance(backend, backend_class)
        assert backend.client is client
        assert (backend.max_requests, backend.max_bytes) == (7, 99)
        assert supports_batch(provider) is True

    @pytest.mark.parametrize("provider", ["openrouter", "custom", ""])
    def test_unsupported_provider(self, provider: str) -> None:
        assert supports_batch(provider) is False
        with pytest.raises(ValueError, match="no batch API"):
            create_backend(provider, object())


class TestClients:
    def test_openai_client(self) -> None:
        client = create_client("openai", "sk-test", timeout=90, connect_timeout=5)
        assert client.api_key == "sk-test"
        assert isinstance(client.timeout, httpx.Timeout)
        assert (client.timeout.read, client.timeout.connect) == (90, 5)

    def test_anthropic_client(self) -> None:
        client = create_client("anthropic", "ak-test", timeout=60)
        assert client.api_key == "ak-test"
        assert client.timeout.read == 60

    def test_client_without_timeout_keeps_sdk_default(self) -> None:
        from openai import OpenAI

        client = create_client("openai", "sk-test")
        assert client.timeout == OpenAI(api_key="sk-test").timeout

    def test_google_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from google import genai

        calls: list[dict[str, Any]] = []
        monkeypatch.setattr(genai, "Client", lambda **kwargs: calls.append(kwargs))
        create_client("google", "gk-test", timeout=2.5)
        create_client("google", "gk-test")
        http = httpx.Client()
        create_client("google", "gk-test", http_client=http)
        assert calls == [
            {"api_key": "gk-test", "http_options": {"timeout": 2500}},
            {"api_key": "gk-test"},
            {"api_key": "gk-test", "http_options": {"httpx_client": http}},
        ]

    @pytest.mark.parametrize("provider", ["openai", "anthropic"])
    def test_given_http_client_is_used(self, provider: str) -> None:
        http = httpx.Client()
        client = create_client(provider, "key", timeout=30, http_client=http)
        assert batch_module.sdk_http_client(client) is http

    def test_hooks_on_a_stand_in_client_change_nothing(self) -> None:
        assert batch_module.add_event_hooks(object(), {"response": [print]}) is False


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, datetime(1970, 1, 1, tzinfo=UTC)),
        ("2026-10-07T08:30:00Z", datetime(2026, 10, 7, 8, 30, tzinfo=UTC)),
        (datetime(2026, 10, 7, 8, 30), datetime(2026, 10, 7, 8, 30, tzinfo=UTC)),
        ("not a time", None),
        (None, None),
        (True, None),
    ],
)
def test_as_datetime(value: Any, expected: datetime | None) -> None:
    assert batch_module.as_datetime(value) == expected


@pytest.mark.parametrize(
    ("error", "status", "definite"),
    [
        (SimpleNamespace(status_code=404), 404, True),
        (SimpleNamespace(code=503), 503, False),
        (SimpleNamespace(code="INVALID"), None, False),
        (SimpleNamespace(response=SimpleNamespace(status_code=429)), 429, True),
        (RuntimeError("lost"), None, False),
        (SimpleNamespace(status_code=400), 400, True),
        (SimpleNamespace(status_code=422), 422, True),
        (SimpleNamespace(status_code=408), 408, False),
    ],
)
def test_status_code_and_definite_refusal(
    error: Any, status: int | None, definite: bool
) -> None:
    """A 4xx answer refuses the request for good; a timeout (408) or 5xx does not."""
    assert batch_module.status_code_of(error) == status
    assert batch_module.is_definite_refusal(error) is definite
