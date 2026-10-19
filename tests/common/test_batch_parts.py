"""Part API, reconciliation listing, raw records and hooked batch clients.

Fake SDK clients stand in for the three providers; the hooked-client tests
send real SDK requests into an ``httpx.MockTransport``.
"""

from __future__ import annotations

import asyncio
import io
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx
import openai
import pytest
from anthropic.types.messages import MessageBatch
from google.genai import errors as genai_errors
from google.genai import types

import chronominer.batch.clients as clients_module
from chronominer.batch.clients import BatchClients, status_of
from chronominer.common import batch_google
from chronominer.common.batch import (
    BatchBackend,
    BatchCreateError,
    BatchHandle,
    BatchPart,
    BatchRequest,
    BatchStatus,
    PartUpload,
    create_backend,
    create_client,
    sdk_http_client,
)
from chronominer.common.batch_anthropic import AnthropicBatchBackend
from chronominer.common.batch_google import GoogleBatchBackend
from chronominer.common.batch_openai import ENDPOINT, OpenAIBatchBackend
from chronominer.common.capture import (
    capture_attempt,
    capture_hooks,
    capture_response,
)
from chronominer.settings import Settings

SID = "6f1c2a9e-4b7d-4c1e-9a55-0d3e8f2b7c10"
T0 = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
REQUEST = httpx.Request("POST", "https://api.example.invalid/v1/batches")


def _requests(n: int) -> list[BatchRequest]:
    return [BatchRequest(custom_id=f"p{i}", text=f"page {i}") for i in range(1, n + 1)]


def _openai_body(request: BatchRequest) -> dict[str, Any]:
    return {"model": "gpt-test", "input": request.text}


def _anthropic_body(request: BatchRequest) -> dict[str, Any]:
    return {
        "model": "claude-test",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": request.text}],
    }


def _google_body(request: BatchRequest) -> dict[str, Any]:
    return {
        "model": request.metadata.get("model", "gemini-test"),
        "contents": [{"role": "user", "parts": [{"text": request.text}]}],
        "config": {"system_instruction": "Read the page.", "max_output_tokens": 64},
    }


def _pages(cursor: str | None, pages: list[list[Any]], key: str) -> int:
    """Return the index of the page after the one whose last item is *cursor*."""
    if cursor is None:
        return 0
    for index, page in enumerate(pages):
        if page and str(getattr(page[-1], key)) == cursor:
            return index + 1
    raise LookupError(cursor)


class FakeOpenAI:
    """OpenAI files and batches with an ordered event log and cursor pages."""

    def __init__(self) -> None:
        self.events: list[tuple[str, Any]] = []
        self.uploads: dict[str, bytes] = {}
        self.contents: dict[str, bytes] = {}
        self.deleted: list[str] = []
        self.created: list[dict[str, Any]] = []
        self.create_error: Exception | None = None
        self.records: dict[str, Any] = {}
        self.pages: list[list[Any]] = []
        self.list_calls: list[dict[str, Any]] = []
        self.options: list[dict[str, Any]] = []
        self.files = SimpleNamespace(
            create=self._upload, content=self._content, delete=self._delete
        )
        self.batches = SimpleNamespace(
            create=self._create, retrieve=self.records.__getitem__, list=self._list
        )

    def with_options(self, **kwargs: Any) -> FakeOpenAI:
        self.options.append(kwargs)
        return self

    def _upload(self, *, file: tuple[str, bytes], purpose: str) -> Any:
        file_id = f"file-{len(self.uploads) + 1}"
        self.uploads[file_id] = file[1]
        self.events.append(("files.create", file_id))
        return SimpleNamespace(id=file_id)

    def _content(self, file_id: str) -> Any:
        data = self.contents[file_id]
        return SimpleNamespace(read=lambda: data)

    def _delete(self, file_id: str) -> None:
        self.deleted.append(file_id)

    def _create(self, **kwargs: Any) -> Any:
        self.events.append(("batches.create", kwargs["input_file_id"]))
        if self.create_error is not None:
            raise self.create_error
        self.created.append(kwargs)
        return SimpleNamespace(id=f"batch_{len(self.created)}")

    def _list(self, *, limit: int, after: str | None = None) -> Any:
        self.list_calls.append({"limit": limit, "after": after})
        index = _pages(after, self.pages, "id")
        return SimpleNamespace(
            data=self.pages[index], has_more=index < len(self.pages) - 1
        )


class FakeAnthropic:
    """Anthropic message batches with cursor pages."""

    def __init__(self) -> None:
        self.created: list[list[dict[str, Any]]] = []
        self.create_error: Exception | None = None
        self.pages: list[list[Any]] = []
        self.list_calls: list[dict[str, Any]] = []
        self.options: list[dict[str, Any]] = []
        self.messages = SimpleNamespace(
            batches=SimpleNamespace(create=self._create, list=self._list)
        )

    def with_options(self, **kwargs: Any) -> FakeAnthropic:
        self.options.append(kwargs)
        return self

    def _create(self, *, requests: list[dict[str, Any]]) -> Any:
        if self.create_error is not None:
            raise self.create_error
        self.created.append(requests)
        return SimpleNamespace(id=f"msgbatch_{len(self.created)}")

    def _list(self, *, limit: int, after_id: str | None = None) -> Any:
        self.list_calls.append({"limit": limit, "after_id": after_id})
        index = _pages(after_id, self.pages, "id")
        page = self.pages[index]
        return SimpleNamespace(
            data=page,
            has_more=index < len(self.pages) - 1,
            last_id=page[-1].id if page else None,
        )


class FakeGenai:
    """Gemini files and batches with token pages shaped like the SDK pager."""

    def __init__(self) -> None:
        self.events: list[tuple[str, Any]] = []
        self.uploads: list[dict[str, Any]] = []
        self.downloads: dict[str, bytes] = {}
        self.deleted: list[str] = []
        self.created: list[dict[str, Any]] = []
        self.create_error: Exception | None = None
        self.jobs: dict[str, Any] = {}
        self.pages: list[list[Any]] = []
        self.list_configs: list[dict[str, Any]] = []
        self.files = SimpleNamespace(
            upload=self._upload, download=self._download, delete=self._delete
        )
        self.batches = SimpleNamespace(
            create=self._create, list=self._list, get=self._get
        )

    def _upload(self, *, file: io.BytesIO, config: dict[str, Any]) -> Any:
        name = f"files/input-{len(self.uploads) + 1}"
        self.uploads.append({"data": file.read(), "config": config})
        self.events.append(("files.upload", name))
        return SimpleNamespace(name=name)

    def _download(self, *, file: str) -> bytes:
        return self.downloads[file]

    def _delete(self, *, name: str) -> None:
        self.deleted.append(name)

    def _create(self, *, model: str, src: Any, config: dict[str, Any]) -> Any:
        self.events.append(("batches.create", src if isinstance(src, str) else None))
        if self.create_error is not None:
            raise self.create_error
        self.created.append({"model": model, "src": src, "config": config})
        return SimpleNamespace(name=f"batches/{len(self.created)}")

    def _get(self, *, name: str) -> Any:
        return self.jobs[name]

    def _list(self, *, config: dict[str, Any]) -> Any:
        self.list_configs.append(dict(config))
        token = config.get("page_token")
        index = int(token) if token else 0
        following = str(index + 1) if index < len(self.pages) - 1 else None
        return SimpleNamespace(
            page=self.pages[index], config={**config, "page_token": following}
        )


@pytest.fixture(params=["openai", "anthropic", "google"])
def provider(request: pytest.FixtureRequest) -> str:
    return str(request.param)


FAKES = {"openai": FakeOpenAI, "anthropic": FakeAnthropic, "google": FakeGenai}
BODIES = {
    "openai": _openai_body,
    "anthropic": _anthropic_body,
    "google": _google_body,
}


def _sent(provider: str, client: Any) -> list[Any]:
    """Return what each create call carried: lines, inline requests or source."""
    if provider == "openai":
        return [client.uploads[call["input_file_id"]] for call in client.created]
    if provider == "anthropic":
        return list(client.created)
    return [call["src"] for call in client.created]


class TestPlanParts:
    def test_planned_parts_match_submit(self, provider: str) -> None:
        requests, body = _requests(5), BODIES[provider]
        planned_client, submitted_client = FAKES[provider](), FAKES[provider]()
        planner = create_backend(provider, planned_client, max_requests=2)
        parts = planner.plan_parts(requests, body)
        handles = create_backend(provider, submitted_client, max_requests=2).submit(
            requests, body
        )
        assert [part.custom_ids for part in parts] == [h.custom_ids for h in handles]
        assert [(part.index, part.total) for part in parts] == [(1, 3), (2, 3), (3, 3)]
        for part in parts:
            planner.submit_part(part, submission_id=SID)
        assert _sent(provider, planned_client) == _sent(provider, submitted_client)

    def test_byte_split_matches_submit(self) -> None:
        backend = OpenAIBatchBackend(FakeOpenAI(), max_bytes=200)
        handles = OpenAIBatchBackend(FakeOpenAI(), max_bytes=200).submit(
            _requests(3), _openai_body
        )
        parts = backend.plan_parts(_requests(3), _openai_body)
        assert [p.custom_ids for p in parts] == [h.custom_ids for h in handles]

    def test_duplicate_ids_and_empty_input(self) -> None:
        backend = OpenAIBatchBackend(FakeOpenAI())
        assert backend.plan_parts([], _openai_body) == []
        with pytest.raises(ValueError, match="unique"):
            backend.plan_parts([BatchRequest("a"), BatchRequest("a")], _openai_body)


class TestUploadAndCreate:
    def test_openai_upload_reported_before_create(self) -> None:
        client = FakeOpenAI()
        backend = OpenAIBatchBackend(client)
        (part,) = backend.plan_parts(_requests(2), _openai_body)

        def on_uploaded(upload: PartUpload) -> None:
            client.events.append(("callback", upload.file_id))

        handle = backend.submit_part(
            part, submission_id=SID, label="run", on_uploaded=on_uploaded
        )
        assert client.events == [
            ("files.create", "file-1"),
            ("callback", "file-1"),
            ("batches.create", "file-1"),
        ]
        assert client.created[0] == {
            "input_file_id": "file-1",
            "endpoint": ENDPOINT,
            "completion_window": "24h",
            "metadata": {"description": "run", "submission_id": SID, "part": "1"},
        }
        assert client.options == [{"max_retries": 0}]
        assert handle.batch_id == "batch_1"
        assert handle.custom_ids == ["p1", "p2"]
        assert handle.metadata == {
            "input_file_id": "file-1",
            "request_count": 2,
            "submission_id": SID,
            "part": 1,
        }

    def test_openai_metadata_names_each_part(self) -> None:
        client = FakeOpenAI()
        backend = OpenAIBatchBackend(client, max_requests=1)
        for part in backend.plan_parts(_requests(2), _openai_body):
            backend.submit_part(part, submission_id=SID)
        assert [call["metadata"]["part"] for call in client.created] == ["1", "2"]
        assert {call["metadata"]["submission_id"] for call in client.created} == {SID}

    def test_google_inline_part_carries_the_id_in_its_display_name(self) -> None:
        client = FakeGenai()
        backend = GoogleBatchBackend(client)
        (part,) = backend.plan_parts(_requests(2), _google_body)
        uploads: list[PartUpload] = []
        handle = backend.submit_part(
            part, submission_id=SID, label="run", on_uploaded=uploads.append
        )
        assert uploads == [PartUpload("google", None)]
        assert client.uploads == []
        created = client.created[0]
        assert created["model"] == "models/gemini-test"
        assert SID in created["config"]["display_name"]
        assert created["src"][1]["metadata"] == {"key": "p2"}
        assert handle.metadata["input_file_name"] is None

    def test_google_file_part_uploads_before_create(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(batch_google, "MAX_INLINE_BYTES", 10)
        client = FakeGenai()
        backend = GoogleBatchBackend(client)
        (part,) = backend.plan_parts(_requests(2), _google_body)

        def on_uploaded(upload: PartUpload) -> None:
            client.events.append(("callback", upload.file_id))

        handle = backend.submit_part(
            part, submission_id=SID, label="run", on_uploaded=on_uploaded
        )
        assert client.events == [
            ("files.upload", "files/input-1"),
            ("callback", "files/input-1"),
            ("batches.create", "files/input-1"),
        ]
        assert SID in client.uploads[0]["config"]["display_name"]
        assert SID in client.created[0]["config"]["display_name"]
        assert handle.metadata["input_file_name"] == "files/input-1"

    def test_google_display_name_keeps_the_id_of_a_long_label(self) -> None:
        name = batch_google.display_name("x" * 300, SID, 12)
        assert len(name) == batch_google.MAX_DISPLAY_NAME
        assert name.endswith(f"-{SID}-p12")

    def test_google_mixed_models_fail_before_any_call(self) -> None:
        client = FakeGenai()
        backend = GoogleBatchBackend(client)
        requests = _requests(2)
        requests[1].metadata["model"] = "gemini-other"
        (part,) = backend.plan_parts(requests, _google_body)
        with pytest.raises(ValueError, match="same model"):
            backend.submit_part(part, submission_id=SID)
        assert client.events == []

    def test_anthropic_part_goes_inline_without_retries(self) -> None:
        client = FakeAnthropic()
        backend = AnthropicBatchBackend(client)
        (part,) = backend.plan_parts(_requests(2), _anthropic_body)
        upload = backend.upload_part(part, submission_id=SID)
        assert upload == PartUpload("anthropic", None)
        handle = backend.create_part(part, upload, submission_id=SID)
        assert client.created == [part.entries]
        assert client.options == [{"max_retries": 0}]
        assert handle.metadata == {"request_count": 2, "submission_id": SID, "part": 1}


def _status_error(cls: Any, status: int) -> Exception:
    """Return an SDK status error of *cls* with a real *status* response."""
    error: Exception = cls(
        "refused", response=httpx.Response(status, request=REQUEST), body=None
    )
    return error


AMBIGUOUS = [
    pytest.param(httpx.ReadTimeout("read timed out"), id="httpx-timeout"),
    pytest.param(httpx.ConnectError("connection refused"), id="httpx-connect"),
    pytest.param(openai.APITimeoutError(request=REQUEST), id="openai-timeout"),
    pytest.param(openai.APIConnectionError(request=REQUEST), id="openai-connection"),
    pytest.param(_status_error(openai.InternalServerError, 500), id="openai-500"),
    pytest.param(RuntimeError("response lost"), id="lost-response"),
    pytest.param(
        genai_errors.ServerError(503, {"error": {"code": 503}}), id="google-503"
    ),
]
GOOGLE_AMBIGUOUS = [
    param
    for param in AMBIGUOUS
    if param.id in {"httpx-timeout", "lost-response", "google-503"}
]
DEFINITE = [
    pytest.param(_status_error(openai.BadRequestError, 400), id="openai-400"),
    pytest.param(_status_error(openai.RateLimitError, 429), id="openai-429"),
    pytest.param(
        genai_errors.ClientError(400, {"error": {"code": 400, "message": "bad"}}),
        id="google-400",
    ),
]


class TestCreateFailures:
    @pytest.mark.parametrize("error", AMBIGUOUS)
    def test_openai_unknown_outcome_keeps_the_upload(self, error: Exception) -> None:
        client = FakeOpenAI()
        client.create_error = error
        backend = OpenAIBatchBackend(client)
        (part,) = backend.plan_parts(_requests(2), _openai_body)
        with pytest.raises(BatchCreateError) as raised:
            backend.submit_part(part, submission_id=SID)
        assert raised.value.ambiguous is True
        assert raised.value.definite is False
        assert raised.value.upload == PartUpload("openai", "file-1")
        assert raised.value.part == 1
        assert raised.value.__cause__ is error
        assert SID in str(raised.value)
        assert client.deleted == []
        assert "file-1" in client.uploads

    @pytest.mark.parametrize("error", DEFINITE)
    def test_openai_refusal_is_definite(self, error: Exception) -> None:
        client = FakeOpenAI()
        client.create_error = error
        backend = OpenAIBatchBackend(client)
        (part,) = backend.plan_parts(_requests(1), _openai_body)
        with pytest.raises(BatchCreateError) as raised:
            backend.submit_part(part, submission_id=SID)
        assert raised.value.definite is True
        assert "refused" in str(raised.value)
        assert client.deleted == []

    @pytest.mark.parametrize(
        ("error", "definite"),
        [
            (anthropic.APIConnectionError(request=REQUEST), False),
            (anthropic.APITimeoutError(request=REQUEST), False),
            (_status_error(anthropic.BadRequestError, 400), True),
            (_status_error(anthropic.InternalServerError, 529), False),
        ],
    )
    def test_anthropic_create(self, error: Exception, definite: bool) -> None:
        client = FakeAnthropic()
        client.create_error = error
        backend = AnthropicBatchBackend(client)
        (part,) = backend.plan_parts(_requests(1), _anthropic_body)
        with pytest.raises(BatchCreateError) as raised:
            backend.submit_part(part, submission_id=SID)
        assert raised.value.definite is definite
        assert raised.value.upload == PartUpload("anthropic", None)

    def test_google_file_part_keeps_its_upload(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(batch_google, "MAX_INLINE_BYTES", 10)
        client = FakeGenai()
        client.create_error = httpx.ReadTimeout("read timed out")
        backend = GoogleBatchBackend(client)
        (part,) = backend.plan_parts(_requests(1), _google_body)
        with pytest.raises(BatchCreateError) as raised:
            backend.submit_part(part, submission_id=SID)
        assert raised.value.ambiguous is True
        assert raised.value.upload == PartUpload("google", "files/input-1")
        assert client.deleted == []

    @pytest.mark.parametrize("error", GOOGLE_AMBIGUOUS)
    def test_google_inline_unknown_outcome_stays_unresolved(
        self, error: Exception
    ) -> None:
        client = FakeGenai()
        client.create_error = error
        backend = GoogleBatchBackend(client)
        (part,) = backend.plan_parts(_requests(2), _google_body)
        with pytest.raises(BatchCreateError) as raised:
            backend.submit_part(part, submission_id=SID)
        assert raised.value.ambiguous is True
        assert raised.value.definite is False
        assert raised.value.upload == PartUpload("google", None)
        assert SID in str(raised.value)
        assert client.uploads == []
        assert client.deleted == []

    def test_upload_error_propagates_unwrapped(self) -> None:
        client = FakeOpenAI()
        client.files.create = lambda **_: (_ for _ in ()).throw(OSError("disk"))
        backend = OpenAIBatchBackend(client)
        (part,) = backend.plan_parts(_requests(1), _openai_body)
        with pytest.raises(OSError, match="disk"):
            backend.submit_part(part, submission_id=SID)


class TestDeleteUpload:
    def test_openai_and_google_delete_their_files(self) -> None:
        openai_client, genai_client = FakeOpenAI(), FakeGenai()
        OpenAIBatchBackend(openai_client).delete_upload(PartUpload("openai", "f-1"))
        GoogleBatchBackend(genai_client).delete_upload(PartUpload("google", "files/x"))
        assert openai_client.deleted == ["f-1"]
        assert genai_client.deleted == ["files/x"]

    def test_inline_upload_has_nothing_to_delete(self) -> None:
        AnthropicBatchBackend(FakeAnthropic()).delete_upload(PartUpload("anthropic"))


def _openai_batch(number: int, created: datetime, **metadata: str) -> Any:
    return openai.types.Batch.model_validate(
        {
            "id": f"batch_{number}",
            "object": "batch",
            "endpoint": ENDPOINT,
            "input_file_id": f"file-{number}",
            "completion_window": "24h",
            "created_at": int(created.timestamp()),
            "status": "in_progress",
            "metadata": metadata or None,
            "request_counts": {"total": number, "completed": 0, "failed": 0},
        }
    )


def _anthropic_batch(number: int, created: datetime) -> MessageBatch:
    return MessageBatch.model_validate(
        {
            "id": f"msgbatch_{number}",
            "type": "message_batch",
            "processing_status": "in_progress",
            "created_at": created.isoformat(),
            "expires_at": (created + timedelta(days=1)).isoformat(),
            "archived_at": None,
            "cancel_initiated_at": None,
            "ended_at": None,
            "results_url": None,
            "request_counts": {
                "processing": number,
                "succeeded": 1,
                "errored": 0,
                "canceled": 0,
                "expired": 0,
            },
        }
    )


def _google_job(number: int, created: datetime, display: str) -> types.BatchJob:
    return types.BatchJob(
        name=f"batches/{number}",
        display_name=display,
        state=types.JobState.JOB_STATE_RUNNING,
        create_time=created,
    )


class TestListing:
    @pytest.fixture
    def openai_client(self) -> FakeOpenAI:
        client = FakeOpenAI()
        client.pages = [
            [
                _openai_batch(6, T0 + timedelta(hours=5), submission_id=SID, part="1"),
                _openai_batch(5, T0 + timedelta(hours=4)),
            ],
            [
                _openai_batch(4, T0 + timedelta(hours=3), submission_id="other"),
                _openai_batch(3, T0 + timedelta(hours=2)),
            ],
            [
                _openai_batch(2, T0 + timedelta(hours=1), submission_id=SID, part="2"),
                _openai_batch(1, T0),
            ],
        ]
        return client

    def test_openai_lists_every_page(self, openai_client: FakeOpenAI) -> None:
        listings = OpenAIBatchBackend(openai_client).list_batches(page_size=2)
        assert [item.batch_id for item in listings] == [
            f"batch_{n}" for n in range(6, 0, -1)
        ]
        assert openai_client.list_calls == [
            {"limit": 2, "after": None},
            {"limit": 2, "after": "batch_5"},
            {"limit": 2, "after": "batch_3"},
        ]
        first = listings[0]
        assert first.created_at == T0 + timedelta(hours=5)
        assert first.request_count == 6
        assert first.status is BatchStatus.IN_PROGRESS
        assert first.provider_status == "in_progress"
        assert first.metadata == {"submission_id": SID, "part": "1"}
        assert listings[1].metadata == {}

    def test_openai_filters_by_submission(self, openai_client: FakeOpenAI) -> None:
        listings = OpenAIBatchBackend(openai_client).list_batches(submission_id=SID)
        assert [(item.batch_id, item.metadata["part"]) for item in listings] == [
            ("batch_6", "1"),
            ("batch_2", "2"),
        ]
        assert listings[0].handle() == BatchHandle("openai", "batch_6")

    def test_openai_time_window_stops_at_older_pages(
        self, openai_client: FakeOpenAI
    ) -> None:
        listings = OpenAIBatchBackend(openai_client).list_batches(
            since=T0 + timedelta(hours=4), until=T0 + timedelta(hours=4, minutes=30)
        )
        assert [item.batch_id for item in listings] == ["batch_5"]
        assert len(openai_client.list_calls) == 2

    def test_naive_window_counts_as_utc(self, openai_client: FakeOpenAI) -> None:
        naive = (T0 + timedelta(hours=2)).replace(tzinfo=None)
        listings = OpenAIBatchBackend(openai_client).list_batches(
            since=naive, until=naive
        )
        assert [item.batch_id for item in listings] == ["batch_3"]

    def test_max_pages(self, openai_client: FakeOpenAI) -> None:
        listings = OpenAIBatchBackend(openai_client).list_batches(max_pages=1)
        assert len(listings) == 2

    def test_anthropic_lists_counts_and_times(self) -> None:
        client = FakeAnthropic()
        client.pages = [
            [_anthropic_batch(n, T0 + timedelta(hours=n))] for n in range(3, 0, -1)
        ]
        backend = AnthropicBatchBackend(client)
        listings = backend.list_batches(page_size=1)
        assert [(item.batch_id, item.request_count) for item in listings] == [
            ("msgbatch_3", 4),
            ("msgbatch_2", 3),
            ("msgbatch_1", 2),
        ]
        assert [call["after_id"] for call in client.list_calls] == [
            None,
            "msgbatch_3",
            "msgbatch_2",
        ]
        assert listings[0].created_at == T0 + timedelta(hours=3)
        assert listings[0].metadata == {}
        assert listings[0].display_name is None
        assert backend.list_batches(submission_id=SID) == []
        window = backend.list_batches(
            since=T0 + timedelta(hours=2), until=T0 + timedelta(hours=2)
        )
        assert [item.batch_id for item in window] == ["msgbatch_2"]

    def test_google_lists_display_names(self) -> None:
        client = FakeGenai()
        client.pages = [
            [_google_job(1, T0, f"run-{SID}-p1")],
            [_google_job(2, T0 + timedelta(hours=1), "run-other-p1")],
            [_google_job(3, T0 + timedelta(hours=2), f"run-{SID}-p2")],
        ]
        backend = GoogleBatchBackend(client)
        listings = backend.list_batches(page_size=1)
        assert [item.batch_id for item in listings] == [
            "batches/1",
            "batches/2",
            "batches/3",
        ]
        assert [config.get("page_token") for config in client.list_configs] == [
            None,
            "1",
            "2",
        ]
        assert listings[0].display_name == f"run-{SID}-p1"
        assert listings[0].created_at == T0
        assert listings[0].request_count is None
        assert listings[0].status is BatchStatus.IN_PROGRESS
        matched = backend.list_batches(submission_id=SID)
        assert [item.batch_id for item in matched] == ["batches/1", "batches/3"]
        window = backend.list_batches(since=T0 + timedelta(minutes=30))
        assert [item.batch_id for item in window] == ["batches/2", "batches/3"]

    def test_backend_without_listing_says_so(self) -> None:
        class Bare(BatchBackend):
            provider_name = "bare"
            default_max_requests = 1
            default_max_bytes = 1

            def status(self, handle: BatchHandle) -> Any:
                raise NotImplementedError

            def list_active(self, *, page_size: int = 100) -> list[Any]:
                return []

            def _wrap(self, request: BatchRequest, body: dict[str, Any]) -> Any:
                return body

            def _submit_part(self, entries: list[Any], *, label: str) -> Any:
                return BatchHandle("bare", "b1")

            def _fetch_results(self, handle: BatchHandle) -> Any:
                return [], None

            def _cancel(self, handle: BatchHandle) -> None:
                raise LookupError(handle.batch_id)

        backend = Bare(None)
        with pytest.raises(NotImplementedError, match="cannot list"):
            backend.list_batches()
        part = BatchPart(1, 1, [BatchRequest("a")], [{}])
        handle = backend.submit_part(part, submission_id=SID)
        assert handle.metadata == {"submission_id": SID, "part": 1}
        with pytest.raises(LookupError):
            backend.request_cancel(handle)
        assert backend.cancel(handle) is False


def _openai_line(custom_id: str, text: str, spacing: str = " ") -> str:
    message = {"type": "message", "content": [{"type": "output_text", "text": text}]}
    record = {
        "custom_id": custom_id,
        "response": {"status_code": 200, "body": {"output": [message]}},
    }
    separators = ("," + spacing, ":" + spacing)
    return json.dumps(record, ensure_ascii=False, separators=separators)


class TestRawRecords:
    def test_openai_file_lines_are_kept_verbatim(self) -> None:
        client = FakeOpenAI()
        client.records["b1"] = SimpleNamespace(
            id="b1", status="completed", output_file_id="out", error_file_id="err"
        )
        first = _openai_line("p1", "Zürich zweite Zeile")
        second = _openai_line("p2", "compact", spacing="")
        error = json.dumps(
            {
                "custom_id": "p3",
                "response": None,
                "error": {"code": "batch_expired", "message": "expired"},
            }
        )
        client.contents["out"] = f"{first}\r\n{second}\n".encode()
        client.contents["err"] = f"{error}\n".encode()
        handle = BatchHandle("openai", "b1", custom_ids=["p1", "p2", "p3", "p4"])
        items = OpenAIBatchBackend(client).results(handle)
        assert [(i.custom_id, i.raw_kind) for i in items] == [
            ("p1", "line"),
            ("p2", "line"),
            ("p3", "line"),
            ("p4", None),
        ]
        assert [i.raw for i in items] == [first, second, error, None]
        assert items[0].text == "Zürich zweite Zeile"
        for item, data in zip(
            items, [client.contents["out"]] * 2 + [client.contents["err"]], strict=False
        ):
            assert (item.raw or "").encode("utf-8") in data

    def test_google_file_lines_are_kept_verbatim(self) -> None:
        client = FakeGenai()
        dest = types.BatchJobDestination(file_name="files/out")
        client.jobs["batches/9"] = types.BatchJob(
            name="batches/9", state="JOB_STATE_SUCCEEDED", dest=dest
        )
        success = (
            '{"key":"p1","response":{"candidates":[{"content":'
            '{"parts":[{"text":"Grüße"}]},"finishReason":"STOP"}]}}'
        )
        failure = '{"key": "p2", "error": {"code": 400, "message": "unreadable"}}'
        client.downloads["files/out"] = f"{success}\n{failure}".encode()
        items = GoogleBatchBackend(client).results(
            BatchHandle("google", "batches/9", custom_ids=["p1", "p2"])
        )
        assert [(i.raw, i.raw_kind) for i in items] == [
            (success, "line"),
            (failure, "line"),
        ]
        assert items[0].text == "Grüße"

    def test_google_inline_items_without_a_capture_keep_the_sdk_dump(self) -> None:
        client = FakeGenai()
        response = types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=types.Content(role="model", parts=[types.Part(text="hi")])
                )
            ]
        )
        inlined = [
            types.InlinedResponse(metadata={"key": "p1"}, response=response),
            types.InlinedResponse(error=types.JobError(code=3, message="bad page")),
        ]
        client.jobs["batches/9"] = types.BatchJob(
            name="batches/9",
            state="JOB_STATE_SUCCEEDED",
            dest=types.BatchJobDestination(inlined_responses=inlined),
        )
        items = GoogleBatchBackend(client).results(
            BatchHandle("google", "batches/9", custom_ids=["p1", "p2"])
        )
        assert [i.raw_kind for i in items] == ["sdk", "sdk"]
        dumped = json.loads(items[0].raw or "")
        assert dumped["metadata"] == {"key": "p1"}
        assert json.loads(items[1].raw or "")["error"]["message"] == "bad page"

    def test_anthropic_sdk_records_without_a_capture_keep_the_sdk_dump(self) -> None:
        from anthropic.types.messages import MessageBatchIndividualResponse

        from chronominer.common.batch_anthropic import parse_result

        result = MessageBatchIndividualResponse.model_validate(
            {"custom_id": "a", "result": {"type": "canceled"}}
        )
        item = parse_result(result)
        assert item.raw_kind == "sdk"
        assert json.loads(item.raw or "") == {
            "custom_id": "a",
            "result": {"type": "canceled"},
        }


BODY = (
    b'{"id":  "batch_1", "object": "batch", "endpoint": "/v1/responses",'
    b' "input_file_id": "file-1", "completion_window": "24h",'
    b' "created_at": 1767225600, "status": "in_progress",'
    b' "note": "Z\xc3\xbcrich"}'
)


def _transport(seen: list[str]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(f"{request.method} {request.url.path}")
        return httpx.Response(
            200, content=BODY, headers={"content-type": "application/json"}
        )

    return httpx.MockTransport(handler)


class TestHookedClients:
    def test_given_httpx_client_captures_inside_a_scope(self) -> None:
        seen: list[str] = []
        client = create_client(
            "openai",
            "sk-test",
            http_client=httpx.Client(transport=_transport(seen)),
            event_hooks=capture_hooks(),
        )
        client.batches.retrieve("batch_1")
        with capture_attempt() as scope:
            batch = client.batches.retrieve("batch_1")
        assert batch.id == "batch_1"
        assert seen == ["GET /v1/batches/batch_1"] * 2
        assert [response.body for response in scope.responses] == [BODY]
        assert scope.responses[0].status == 200

    @pytest.mark.parametrize("provider", ["openai", "anthropic", "google"])
    def test_sdk_clients_get_the_hooks(self, provider: str) -> None:
        client = create_client(provider, "test-key", event_hooks=capture_hooks())
        http = sdk_http_client(client)
        assert http is not None
        assert http.event_hooks["response"].count(capture_response) == 1

    def test_hooks_are_added_once(self) -> None:
        http = httpx.Client(event_hooks=capture_hooks())
        create_client(
            "openai", "sk-test", http_client=http, event_hooks=capture_hooks()
        )
        assert http.event_hooks["response"] == [capture_response]

    def test_batch_clients_call_captures_the_provider_body(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[str] = []
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        def build(provider: str, api_key: str, **kwargs: Any) -> Any:
            http = httpx.Client(transport=_transport(seen))
            return create_client(provider, api_key, http_client=http, **kwargs)

        monkeypatch.setattr(clients_module, "create_client", build)
        clients = BatchClients(Settings(), "extract")
        scopes: list[Any] = []
        info = asyncio.run(
            clients.call(
                "openai",
                status_of(BatchHandle("openai", "batch_1")),
                retry=False,
                on_capture=scopes.append,
            )
        )
        assert info.status is BatchStatus.IN_PROGRESS
        assert seen == ["GET /v1/batches/batch_1"]
        assert len(scopes) == 1
        assert [response.body for response in scopes[0].responses] == [BODY]

    def test_batch_clients_reports_the_scope_of_a_failed_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        refusal = b'{"error": {"message": "no such batch", "type": "invalid"}}'

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, content=refusal)

        def build(provider: str, api_key: str, **kwargs: Any) -> Any:
            http = httpx.Client(transport=httpx.MockTransport(handler))
            return create_client(provider, api_key, http_client=http, **kwargs)

        monkeypatch.setattr(clients_module, "create_client", build)
        scopes: list[Any] = []
        with pytest.raises(openai.NotFoundError):
            asyncio.run(
                BatchClients(Settings(), "extract").call(
                    "openai",
                    status_of(BatchHandle("openai", "gone")),
                    retry=False,
                    on_capture=scopes.append,
                )
            )
        assert [response.body for response in scopes[0].responses] == [refusal]
