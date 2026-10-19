"""Google batch backend against a fake SDK client and real SDK models."""

from __future__ import annotations

import io
import json
from types import SimpleNamespace
from typing import Any

import pytest
from google.genai import types

from chronominer.common import batch_google
from chronominer.common.batch import (
    BatchHandle,
    BatchNotReadyError,
    BatchRequest,
    BatchStatus,
    BatchSubmitError,
)
from chronominer.common.batch_google import GoogleBatchBackend, file_request
from chronominer.common.usage import Usage


class FakeFiles:
    def __init__(self) -> None:
        self.uploads: list[dict[str, Any]] = []
        self.downloads: dict[str, bytes] = {}
        self.deleted: list[str] = []
        self.delete_error: Exception | None = None

    def upload(self, *, file: io.BytesIO, config: dict[str, Any]) -> Any:
        self.uploads.append({"data": file.read(), "config": config})
        return SimpleNamespace(name=f"files/input-{len(self.uploads)}")

    def download(self, *, file: str) -> bytes:
        return self.downloads[file]

    def delete(self, *, name: str) -> None:
        if self.delete_error is not None:
            raise self.delete_error
        self.deleted.append(name)


class FakeBatches:
    def __init__(self) -> None:
        self.jobs: dict[str, Any] = {}
        self.created: list[dict[str, Any]] = []
        self.cancelled: list[str] = []
        self.create_error: Exception | None = None
        self.list_configs: list[dict[str, Any]] = []

    def create(self, *, model: str, src: Any, config: dict[str, Any]) -> Any:
        if self.create_error is not None:
            raise self.create_error
        self.created.append({"model": model, "src": src, "config": config})
        name = f"batches/{len(self.created)}"
        self.jobs[name] = types.BatchJob(name=name, state="JOB_STATE_PENDING")
        return self.jobs[name]

    def get(self, *, name: str) -> Any:
        if name not in self.jobs:
            raise LookupError(name)
        return self.jobs[name]

    def cancel(self, *, name: str) -> None:
        if name not in self.jobs:
            raise LookupError(name)
        self.cancelled.append(name)

    def list(self, *, config: dict[str, Any]) -> list[Any]:
        self.list_configs.append(config)
        return list(self.jobs.values())


class FakeGenai:
    def __init__(self) -> None:
        self.files = FakeFiles()
        self.batches = FakeBatches()


def _body(request: BatchRequest) -> dict[str, Any]:
    return {
        "model": request.metadata.get("model", "gemini-test"),
        "contents": [{"role": "user", "parts": [{"text": request.text}]}],
        "config": {
            "system_instruction": "Read the page.",
            "max_output_tokens": 256,
            "response_mime_type": "application/json",
            "safety_settings": [{"category": "HARM_CATEGORY_HATE_SPEECH"}],
        },
    }


def _requests(n: int) -> list[BatchRequest]:
    return [BatchRequest(custom_id=f"p{i}", text=f"page {i}") for i in range(1, n + 1)]


def _job(client: FakeGenai, state: str, **fields: Any) -> BatchHandle:
    client.batches.jobs["batches/9"] = types.BatchJob(
        name="batches/9", state=state, **fields
    )
    return BatchHandle("google", "batches/9", custom_ids=["p1", "p2", "p3"])


@pytest.fixture
def client() -> FakeGenai:
    return FakeGenai()


class TestSubmit:
    def test_inline_part_matches_the_sdk_shape(self, client: FakeGenai) -> None:
        handles = GoogleBatchBackend(client).submit(_requests(2), _body, label="run")
        created = client.batches.created[0]
        assert created["model"] == "models/gemini-test"
        assert created["config"]["display_name"].startswith("run-")
        src = created["src"]
        assert set(src[0]) == {"contents", "config", "metadata"}
        assert src[1]["metadata"] == {"key": "p2"}
        source = types.BatchJobSource(inlined_requests=src)
        assert source.inlined_requests is not None
        assert source.inlined_requests[0].config is not None
        assert source.inlined_requests[0].config.system_instruction == "Read the page."
        assert handles[0].custom_ids == ["p1", "p2"]
        assert handles[0].metadata == {"request_count": 2, "input_file_name": None}
        assert client.files.uploads == []

    def test_large_part_goes_through_a_file(
        self, client: FakeGenai, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(batch_google, "MAX_INLINE_BYTES", 10)
        handles = GoogleBatchBackend(client).submit(_requests(2), _body)
        upload = client.files.uploads[0]
        assert upload["config"]["mime_type"] == "jsonl"
        lines = [json.loads(line) for line in upload["data"].decode().splitlines()]
        assert lines[0] == {
            "key": "p1",
            "request": {
                "contents": [{"role": "user", "parts": [{"text": "page 1"}]}],
                "system_instruction": {"parts": [{"text": "Read the page."}]},
                "safety_settings": [{"category": "HARM_CATEGORY_HATE_SPEECH"}],
                "generation_config": {
                    "max_output_tokens": 256,
                    "response_mime_type": "application/json",
                },
            },
        }
        assert client.batches.created[0]["src"] == "files/input-1"
        assert handles[0].metadata["input_file_name"] == "files/input-1"

    def test_failed_create_removes_upload(
        self, client: FakeGenai, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(batch_google, "MAX_INLINE_BYTES", 10)
        client.batches.create_error = RuntimeError("overloaded")
        with pytest.raises(BatchSubmitError):
            GoogleBatchBackend(client).submit(_requests(1), _body)
        assert client.files.deleted == ["files/input-1"]

    def test_mixed_models_are_rejected(self, client: FakeGenai) -> None:
        requests = _requests(2)
        requests[1].metadata["model"] = "gemini-other"
        with pytest.raises(BatchSubmitError, match="same model"):
            GoogleBatchBackend(client).submit(requests, _body)

    def test_body_without_model_is_rejected(self, client: FakeGenai) -> None:
        with pytest.raises(ValueError, match="no model"):
            GoogleBatchBackend(client).submit(_requests(1), lambda r: {"contents": []})

    @pytest.mark.parametrize("inline", [True, False], ids=["inline", "file"])
    def test_request_bodies_are_the_requests_as_submitted(
        self, client: FakeGenai, monkeypatch: pytest.MonkeyPatch, inline: bool
    ) -> None:
        if not inline:
            monkeypatch.setattr(batch_google, "MAX_INLINE_BYTES", 10)
        backend = GoogleBatchBackend(client)
        (part,) = backend.plan_parts(_requests(2), _body)

        bodies = backend.request_bodies(part)
        backend.submit(_requests(2), _body)

        if inline:
            sent = client.batches.created[0]["src"]
        else:
            data = client.files.uploads[0]["data"].decode()
            sent = [json.loads(line)["request"] for line in data.splitlines()]
        assert bodies == {"p1": sent[0], "p2": sent[1]}


def test_file_request_keeps_structured_instruction() -> None:
    instruction = {"parts": [{"text": "x"}]}
    rest = file_request({"contents": [], "config": {"system_instruction": instruction}})
    assert rest == {"contents": [], "system_instruction": instruction}


class TestStatus:
    @pytest.mark.parametrize(
        ("state", "expected"),
        [
            ("JOB_STATE_QUEUED", BatchStatus.PENDING),
            ("JOB_STATE_PENDING", BatchStatus.PENDING),
            ("JOB_STATE_RUNNING", BatchStatus.IN_PROGRESS),
            ("JOB_STATE_CANCELLING", BatchStatus.IN_PROGRESS),
            ("JOB_STATE_PAUSED", BatchStatus.IN_PROGRESS),
            ("JOB_STATE_UPDATING", BatchStatus.IN_PROGRESS),
            ("JOB_STATE_SUCCEEDED", BatchStatus.COMPLETED),
            ("JOB_STATE_PARTIALLY_SUCCEEDED", BatchStatus.COMPLETED),
            ("JOB_STATE_FAILED", BatchStatus.FAILED),
            ("JOB_STATE_CANCELLED", BatchStatus.CANCELLED),
            ("JOB_STATE_EXPIRED", BatchStatus.EXPIRED),
            ("JOB_STATE_UNSPECIFIED", BatchStatus.UNKNOWN),
        ],
    )
    def test_status_mapping(
        self, client: FakeGenai, state: str, expected: BatchStatus
    ) -> None:
        handle = _job(client, state)
        info = GoogleBatchBackend(client).status(handle)
        assert info.status is expected
        assert info.provider_status == state
        assert info.total_requests == 3
        assert (info.error_message is not None) is (expected is BatchStatus.UNKNOWN)

    def test_results_available(self, client: FakeGenai) -> None:
        dest = types.BatchJobDestination(file_name="files/out")
        handle = _job(client, "JOB_STATE_SUCCEEDED", dest=dest)
        info = GoogleBatchBackend(client).status(handle)
        assert (info.results_available, info.output_file_id) == (True, "files/out")

    def test_failed_job_reports_its_error(self, client: FakeGenai) -> None:
        error = types.JobError(code=3, message="invalid model")
        handle = _job(client, "JOB_STATE_FAILED", error=error)
        assert GoogleBatchBackend(client).status(handle).error_message == (
            "invalid model"
        )


class TestResults:
    def test_file_results(self, client: FakeGenai) -> None:
        dest = types.BatchJobDestination(file_name="files/out")
        handle = _job(client, "JOB_STATE_SUCCEEDED", dest=dest)
        records = [
            {
                "key": "p1",
                "response": {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {"text": "plan", "thought": True},
                                    {"text": "Hello"},
                                ]
                            },
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 10,
                        "candidatesTokenCount": 3,
                        "thoughtsTokenCount": 2,
                        "totalTokenCount": 15,
                        "cachedContentTokenCount": 4,
                    },
                },
            },
            {"key": "p2", "error": {"code": 400, "message": "image unreadable"}},
            {
                "key": "p3",
                "response": {
                    "candidates": [{"finishReason": "SAFETY", "content": {"parts": []}}]
                },
            },
        ]
        client.files.downloads["files/out"] = "\n".join(
            json.dumps(r) for r in records
        ).encode()
        items = GoogleBatchBackend(client).results(handle)
        assert [(i.custom_id, i.success) for i in items] == [
            ("p1", True),
            ("p2", False),
            ("p3", False),
        ]
        assert items[0].text == "Hello"
        assert items[0].usage == Usage(10, 5, 15, 4, 2)
        assert (items[1].error, items[1].error_code) == ("image unreadable", "400")
        assert "SAFETY" in (items[2].error or "")
        assert items[2].error_code == "content_filter"

    @pytest.mark.parametrize(
        ("finish", "success", "kind"),
        [
            ("STOP", True, "ok"),
            ("MAX_TOKENS", False, "truncated"),
            ("SAFETY", False, "content_filter"),
            ("RECITATION", False, "content_filter"),
            ("PROHIBITED_CONTENT", False, "content_filter"),
            ("BLOCKLIST", False, "content_filter"),
            ("SPII", False, "content_filter"),
        ],
    )
    def test_finish_reason_of_an_answer_with_text(
        self, finish: str, success: bool, kind: str
    ) -> None:
        candidate = {"content": {"parts": [{"text": "a"}]}, "finishReason": finish}
        item = batch_google.parse_response("p1", {"candidates": [candidate]})
        assert (item.success, item.outcome.kind, item.text) == (success, kind, "a")

    def test_a_blocked_prompt_without_candidates_is_a_content_filter(self) -> None:
        response = {"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"}}
        item = batch_google.parse_response("p1", response)
        assert (item.success, item.outcome.kind) == (False, "content_filter")
        assert "PROHIBITED_CONTENT" in (item.error or "")

    def test_inline_results(self, client: FakeGenai) -> None:
        def response(text: str) -> types.GenerateContentResponse:
            return types.GenerateContentResponse(
                candidates=[
                    types.Candidate(
                        content=types.Content(
                            role="model", parts=[types.Part(text=text)]
                        ),
                        finish_reason=types.FinishReason.STOP,
                    )
                ],
                usage_metadata=types.GenerateContentResponseUsageMetadata(
                    prompt_token_count=7, candidates_token_count=2
                ),
            )

        inlined = [
            types.InlinedResponse(metadata={"key": "p1"}, response=response("first")),
            types.InlinedResponse(response=response("by position")),
            types.InlinedResponse(error=types.JobError(code=3, message="bad page")),
        ]
        dest = types.BatchJobDestination(inlined_responses=inlined)
        handle = _job(client, "JOB_STATE_PARTIALLY_SUCCEEDED", dest=dest)
        handle.custom_ids.append("p4")
        items = GoogleBatchBackend(client).results(handle)
        assert [(i.custom_id, i.success, i.origin) for i in items] == [
            ("p1", True, "output"),
            ("p2", True, "output"),
            ("p3", False, "output"),
            ("p4", False, "missing"),
        ]
        assert items[0].text == "first"
        assert items[0].usage == Usage(7, 2, 9, 0, 0)
        json.dumps(items[0].response)
        assert items[1].text == "by position"
        assert (items[2].error, items[2].error_code) == ("bad page", "3")

    def test_inline_response_without_text_fails(self, client: FakeGenai) -> None:
        empty = types.GenerateContentResponse(
            candidates=[types.Candidate(finish_reason=types.FinishReason.MAX_TOKENS)]
        )
        dest = types.BatchJobDestination(
            inlined_responses=[types.InlinedResponse(response=empty)]
        )
        handle = _job(client, "JOB_STATE_SUCCEEDED", dest=dest)
        item = GoogleBatchBackend(client).results(handle)[0]
        assert item.success is False
        assert "MAX_TOKENS" in (item.error or "")

    def test_running_job_is_not_ready(self, client: FakeGenai) -> None:
        handle = _job(client, "JOB_STATE_RUNNING")
        with pytest.raises(BatchNotReadyError):
            GoogleBatchBackend(client).results(handle)

    def test_failed_job_without_results(self, client: FakeGenai) -> None:
        error = types.JobError(code=8, message="resource exhausted")
        handle = _job(client, "JOB_STATE_FAILED", error=error)
        items = GoogleBatchBackend(client).results(handle)
        assert [i.origin for i in items] == ["missing"] * 3
        assert {i.error for i in items} == {"resource exhausted"}


class TestCancelListCleanup:
    def test_cancel(self, client: FakeGenai) -> None:
        handle = _job(client, "JOB_STATE_RUNNING")
        backend = GoogleBatchBackend(client)
        assert backend.cancel(handle) is True
        assert backend.cancel(BatchHandle("google", "batches/x")) is False
        assert client.batches.cancelled == ["batches/9"]

    def test_list_active(self, client: FakeGenai) -> None:
        for name, state in [
            ("batches/1", "JOB_STATE_RUNNING"),
            ("batches/2", "JOB_STATE_SUCCEEDED"),
            ("batches/3", "JOB_STATE_QUEUED"),
        ]:
            client.batches.jobs[name] = types.BatchJob(name=name, state=state)
        active = GoogleBatchBackend(client).list_active(page_size=20)
        assert [(s.batch_id, s.status) for s in active] == [
            ("batches/1", BatchStatus.IN_PROGRESS),
            ("batches/3", BatchStatus.PENDING),
        ]
        assert client.batches.list_configs == [{"page_size": 20}]

    def test_cleanup_deletes_input_and_output(self, client: FakeGenai) -> None:
        dest = types.BatchJobDestination(file_name="files/out")
        handle = _job(client, "JOB_STATE_SUCCEEDED", dest=dest)
        handle.metadata["input_file_name"] = "files/in"
        GoogleBatchBackend(client).cleanup(handle)
        assert client.files.deleted == ["files/in", "files/out"]

    def test_cleanup_skips_a_result_file_the_files_api_cannot_delete(
        self, client: FakeGenai
    ) -> None:
        # Gemini names result files files/batch-<batch id>: a 42-character ID.
        result = "files/batch-6reziw5t20wy9apvj1vsbd4qwoddwtc4i19t"
        handle = _job(
            client,
            "JOB_STATE_SUCCEEDED",
            dest=types.BatchJobDestination(file_name=result),
        )
        handle.metadata["input_file_name"] = "files/" + "a" * 40
        GoogleBatchBackend(client).cleanup(handle)
        assert client.files.deleted == ["files/" + "a" * 40]

    def test_cleanup_never_raises(self, client: FakeGenai) -> None:
        handle = BatchHandle("google", "batches/x", metadata={"input_file_name": "f"})
        client.files.delete_error = RuntimeError("delete failed")
        GoogleBatchBackend(client).cleanup(handle)
