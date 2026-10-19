"""Batch results stored as downloaded, through real SDK clients.

Each test builds an anthropic, google-genai or openai client over an
``httpx.MockTransport`` with the capture hooks and downloads results inside
``capture_attempt()``, as :class:`~chronominer.batch.clients.
BatchClients` does; no request leaves the process.

Windows command::

    uv --directory <repo> run pytest tests/common/test_batch_verbatim.py -q

Oracle: every item's ``raw`` is its record as served (a line of the results
body, or the element of the served ``inlinedResponses`` array), with
``raw_kind`` ``'line'``; a malformed line comes back as an item without a
custom id that carries the line; without a capture scope Anthropic and
Google inline items keep ``'sdk'``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx

from chronominer.common.batch import BatchHandle, create_client
from chronominer.common.batch_anthropic import AnthropicBatchBackend
from chronominer.common.batch_google import GoogleBatchBackend
from chronominer.common.batch_openai import OpenAIBatchBackend
from chronominer.common.capture import capture_attempt, capture_hooks, sdk_dump

Handler = Callable[[httpx.Request], httpx.Response]

MALFORMED = '{"custom_id": "b", "result": {"type": "succ'


def _client(provider: str, handler: Handler, seen: list[str]) -> Any:
    def logged(request: httpx.Request) -> httpx.Response:
        seen.append(f"{request.method} {request.url.path}")
        return handler(request)

    return create_client(
        provider,
        "test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(logged)),
        event_hooks=capture_hooks(),
    )


def _json(data: Any) -> httpx.Response:
    return httpx.Response(200, json=data)


BATCH_ID = "msgbatch_01"
RESULTS_URL = f"https://api.anthropic.com/v1/messages/batches/{BATCH_ID}/results"


def _anthropic_batch() -> dict[str, Any]:
    return {
        "id": BATCH_ID,
        "type": "message_batch",
        "processing_status": "ended",
        "request_counts": {
            "processing": 0,
            "succeeded": 3,
            "errored": 1,
            "canceled": 0,
            "expired": 0,
        },
        "created_at": "2026-10-01T10:00:00Z",
        "ended_at": "2026-10-01T11:00:00Z",
        "expires_at": "2026-10-02T10:00:00Z",
        "archived_at": None,
        "cancel_initiated_at": None,
        "results_url": RESULTS_URL,
    }


def _message_line(custom_id: str, text: str) -> str:
    message = {
        "id": f"msg_{custom_id}",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-6",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    return json.dumps(
        {"custom_id": custom_id, "result": {"type": "succeeded", "message": message}},
        ensure_ascii=False,
    )


# "result" before "custom_id", odd spacing and a null field: the SDK model
# dump would reorder the keys and add the unset fields.
REORDERED = (
    '{"result":{"type":"succeeded","message":{"id":"msg_c","type":"message",'
    '"role":"assistant","model":"claude-sonnet-4-6","content":[{"type":"text",'
    '"text":"Grüße aus Zürich"}],"stop_reason":"end_turn","stop_sequence":null,'
    '"usage":{"input_tokens":7,"output_tokens":3}}},   "custom_id":"c"}'
)
ERRORED = (
    '{"custom_id": "d", "result": {"type": "errored", "error": {"type": "error",'
    ' "error": {"type": "invalid_request_error", "message": "bad image"}}}}'
)


def _anthropic_handler(body: bytes) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/results"):
            return httpx.Response(
                200, content=body, headers={"content-type": "application/binary"}
            )
        return _json(_anthropic_batch())

    return handler


ANTHROPIC_LINES = [
    REORDERED,
    _message_line("a", "erste Seite"),
    MALFORMED,
    ERRORED,
]
ANTHROPIC_BODY = ("\n".join(ANTHROPIC_LINES) + "\n").encode("utf-8")
ANTHROPIC_IDS = ["a", "b", "c", "d"]


def test_anthropic_items_keep_their_lines_and_pair_by_custom_id() -> None:
    seen: list[str] = []
    client = _client("anthropic", _anthropic_handler(ANTHROPIC_BODY), seen)
    handle = BatchHandle("anthropic", BATCH_ID, custom_ids=ANTHROPIC_IDS)

    with capture_attempt():
        items = AnthropicBatchBackend(client).results(handle)

    assert seen[-1] == f"GET /v1/messages/batches/{BATCH_ID}/results"
    by_id = {item.custom_id: item for item in items if item.custom_id}
    assert by_id["c"].raw == REORDERED
    assert by_id["c"].text == "Grüße aus Zürich"
    assert by_id["c"].usage.input_tokens == 7
    assert by_id["a"].raw == ANTHROPIC_LINES[1]
    assert by_id["a"].text == "erste Seite"
    assert by_id["d"].raw == ERRORED
    assert (by_id["d"].error, by_id["d"].error_code) == (
        "bad image",
        "invalid_request_error",
    )
    for custom_id in ("a", "c", "d"):
        assert by_id[custom_id].raw_kind == "line"
        assert (by_id[custom_id].raw or "").encode("utf-8") in ANTHROPIC_BODY
    assert by_id["b"].origin == "missing"
    assert by_id["b"].raw is None


def test_anthropic_line_differs_from_the_sdk_dump() -> None:
    from anthropic.types.messages import MessageBatchIndividualResponse

    model = MessageBatchIndividualResponse.model_validate(json.loads(REORDERED))
    assert sdk_dump(model) != REORDERED


def test_anthropic_malformed_line_is_kept_without_a_custom_id() -> None:
    client = _client("anthropic", _anthropic_handler(ANTHROPIC_BODY), [])
    handle = BatchHandle("anthropic", BATCH_ID, custom_ids=ANTHROPIC_IDS)

    with capture_attempt():
        items = AnthropicBatchBackend(client).results(handle)

    (unmatched,) = [item for item in items if not item.custom_id]
    assert items[-1] is unmatched
    assert unmatched.raw == MALFORMED
    assert unmatched.raw_kind == "line"
    assert unmatched.origin == "unmatched"
    assert not unmatched.success
    assert "line 3" in (unmatched.error or "")


def test_anthropic_without_a_scope_keeps_the_sdk_dump() -> None:
    body = ("\n".join(ANTHROPIC_LINES[:2]) + "\n").encode("utf-8")
    client = _client("anthropic", _anthropic_handler(body), [])
    handle = BatchHandle("anthropic", BATCH_ID, custom_ids=["a", "c"])

    items = AnthropicBatchBackend(client).results(handle)

    assert [(item.custom_id, item.raw_kind) for item in items] == [
        ("c", "sdk"),
        ("a", "sdk"),
    ]
    assert items[0].raw != REORDERED
    assert items[0].text == "Grüße aus Zürich"


def _openai_line(custom_id: str, text: str) -> str:
    message = {"type": "message", "content": [{"type": "output_text", "text": text}]}
    return json.dumps(
        {
            "custom_id": custom_id,
            "response": {"status_code": 200, "body": {"output": [message]}},
        },
        separators=(",", ":"),
        ensure_ascii=False,
    )


def test_openai_malformed_line_is_kept_without_a_custom_id() -> None:
    lines = [_openai_line("p1", "eins"), "not json at all", _openai_line("p2", "zwei")]
    output = ("\r\n".join(lines) + "\r\n").encode("utf-8")
    batch = {
        "id": "batch_1",
        "object": "batch",
        "endpoint": "/v1/responses",
        "input_file_id": "file-in",
        "completion_window": "24h",
        "created_at": 1767225600,
        "status": "completed",
        "output_file_id": "file-out",
        "error_file_id": None,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/files/file-out/content":
            return httpx.Response(200, content=output)
        return _json(batch)

    seen: list[str] = []
    client = _client("openai", handler, seen)
    handle = BatchHandle("openai", "batch_1", custom_ids=["p1", "p2"])

    with capture_attempt():
        items = OpenAIBatchBackend(client).results(handle)

    assert "GET /v1/files/file-out/content" in seen
    assert [(item.custom_id, item.raw) for item in items] == [
        ("p1", lines[0]),
        ("p2", lines[2]),
        ("", lines[1]),
    ]
    assert all(item.raw_kind == "line" for item in items)
    assert items[-1].origin == "unmatched"
    assert "line 2" in (items[-1].error or "")


GOOGLE_NAME = "batches/123"


def _google_job(output: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": GOOGLE_NAME,
        "metadata": {
            "@type": "type.googleapis.com/google.ai.generativelanguage.v1main."
            "GenerateContentBatch",
            "model": "models/gemini-3-flash-preview",
            "displayName": "batch-sid-p1",
            "state": "BATCH_STATE_SUCCEEDED",
            "output": output,
        },
    }


def _google_handler(job: dict[str, Any], download: bytes = b"") -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        if "download" in request.url.path or request.url.path.endswith(":download"):
            return httpx.Response(200, content=download)
        return _json(job)

    return handler


INLINED = [
    {
        "response": {
            "candidates": [
                {
                    "content": {"parts": [{"text": "Grüße"}], "role": "model"},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 2},
            "modelVersion": "gemini-3-flash-preview",
        },
        "metadata": {"key": "g1"},
    },
    {"error": {"code": 3, "message": "bad page"}, "metadata": {"key": "g2"}},
]


def test_google_inline_items_keep_their_wire_elements() -> None:
    job = _google_job({"inlinedResponses": {"inlinedResponses": INLINED}})
    seen: list[str] = []
    client = _client("google", _google_handler(job), seen)
    handle = BatchHandle("google", GOOGLE_NAME, custom_ids=["g1", "g2"])

    with capture_attempt():
        items = GoogleBatchBackend(client).results(handle)

    assert seen and all(entry.startswith("GET ") for entry in seen)
    assert [item.custom_id for item in items] == ["g1", "g2"]
    assert [item.raw_kind for item in items] == ["line", "line"]
    assert [json.loads(item.raw or "") for item in items] == INLINED
    assert '"usageMetadata"' in (items[0].raw or "")
    assert '"finishReason"' in (items[0].raw or "")
    assert "Grüße" in (items[0].raw or "")
    assert items[0].text == "Grüße"
    assert items[1].error == "bad page"


def test_google_inline_without_a_scope_keeps_the_sdk_dump() -> None:
    job = _google_job({"inlinedResponses": {"inlinedResponses": INLINED}})
    client = _client("google", _google_handler(job), [])
    handle = BatchHandle("google", GOOGLE_NAME, custom_ids=["g1", "g2"])

    items = GoogleBatchBackend(client).results(handle)

    assert [item.raw_kind for item in items] == ["sdk", "sdk"]
    assert "usage_metadata" in (items[0].raw or "")


def test_google_file_malformed_line_is_kept_without_a_custom_id() -> None:
    good = (
        '{"key":"g1","response":{"candidates":[{"content":{"parts":'
        '[{"text":"eins"}]},"finishReason":"STOP"}]}}'
    )
    bad = '{"key": "g2", "response": '
    job = _google_job({"responsesFile": "files/out-1"})
    seen: list[str] = []
    client = _client("google", _google_handler(job, f"{good}\n{bad}\n".encode()), seen)
    handle = BatchHandle("google", GOOGLE_NAME, custom_ids=["g1", "g2"])

    with capture_attempt():
        items = GoogleBatchBackend(client).results(handle)

    assert any("download" in entry for entry in seen)
    assert [(item.custom_id, item.origin, item.raw) for item in items] == [
        ("g1", "output", good),
        ("g2", "missing", None),
        ("", "unmatched", bad),
    ]
    assert items[-1].raw_kind == "line"
    assert "line 2" in (items[-1].error or "")
