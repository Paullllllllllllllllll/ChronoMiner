"""Transport capture: response hooks, attempt scopes and the recorder variable."""

from __future__ import annotations

import asyncio
import base64

import httpx

from chronominer.common.capture import (
    AttemptRecord,
    acapture_response,
    async_capture_hooks,
    capture_attempt,
    capture_hooks,
    capture_response,
    current_recorder,
    current_scope,
    install_capture,
    record_attempt,
    redact_payloads,
    sdk_dump,
    use_recorder,
)

BODY = b'{"id": "resp_1",  "output_text": "Gr\xc3\xbc\xc3\x9fe\\n", "x": [1, 2 ]}\n'


def _handler(request: httpx.Request) -> httpx.Response:
    """Answer with an unread stream, as a network transport would."""
    if request.url.path == "/fail":
        body = b'{"error": {"type": "rate_limit"}}'
        return httpx.Response(429, stream=httpx.ByteStream(body))
    return httpx.Response(
        200, stream=httpx.ByteStream(BODY + request.url.path.encode())
    )


def _client() -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(_handler),
        base_url="https://api.example.test",
        event_hooks=capture_hooks(),
    )


def _async_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.MockTransport(_handler),
        base_url="https://api.example.test",
        event_hooks=async_capture_hooks(),
    )


def test_sync_client_captures_bodies_only_inside_a_scope() -> None:
    with _client() as client:
        client.post("/outside")
        with capture_attempt() as scope:
            response = client.post("/v1/responses", json={"a": 1})
            failed = client.get("/fail")
        client.post("/after")

    assert response.content == BODY + b"/v1/responses"
    assert failed.status_code == 429
    first, second = scope.responses
    assert first.body == BODY + b"/v1/responses"
    assert first.status == 200
    assert first.method == "POST"
    assert first.url == "https://api.example.test/v1/responses"
    assert first.elapsed is not None and first.elapsed >= 0
    assert first.text == (BODY + b"/v1/responses").decode("utf-8")
    assert second.status == 429
    assert second.body == b'{"error": {"type": "rate_limit"}}'
    assert scope.last == second
    assert current_scope() is None


def test_async_client_captures_bodies_inside_a_scope() -> None:
    async def main() -> list[bytes]:
        async with _async_client() as client:
            await client.post("/outside")
            with capture_attempt() as scope:
                response = await client.post("/v1/messages")
                assert response.content == BODY + b"/v1/messages"
            await client.post("/after")
        return [captured.body for captured in scope.responses]

    assert asyncio.run(main()) == [BODY + b"/v1/messages"]


def test_scopes_of_gathered_tasks_do_not_mix() -> None:
    async def attempt(client: httpx.AsyncClient, name: str) -> list[str]:
        with capture_attempt() as scope:
            for index in range(3):
                await client.post(f"/{name}/{index}")
                await asyncio.sleep(0)
        return [captured.url.rsplit("/", 2)[-2] for captured in scope.responses]

    async def main() -> list[list[str]]:
        async with _async_client() as client:
            return await asyncio.gather(
                *(attempt(client, name) for name in ("a", "b", "c", "d"))
            )

    assert asyncio.run(main()) == [[name] * 3 for name in ("a", "b", "c", "d")]


def test_a_scope_follows_asyncio_to_thread() -> None:
    def call(client: httpx.Client) -> bytes:
        return client.post("/threaded").content

    async def main() -> tuple[bytes, list[bytes]]:
        with _client() as client, capture_attempt() as scope:
            content = await asyncio.to_thread(call, client)
        return content, [captured.body for captured in scope.responses]

    content, bodies = asyncio.run(main())

    assert bodies == [content] == [BODY + b"/threaded"]


def test_nested_scopes_capture_into_the_innermost() -> None:
    with _client() as client, capture_attempt() as outer:
        client.post("/one")
        with capture_attempt() as inner:
            client.post("/two")
        client.post("/three")

    assert [c.url[-4:] for c in outer.responses] == ["/one", "hree"]
    assert [c.url[-4:] for c in inner.responses] == ["/two"]


def test_install_capture_adds_the_matching_hook_once() -> None:
    sync_client = httpx.Client(transport=httpx.MockTransport(_handler))
    async_client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))

    install_capture(sync_client)
    install_capture(sync_client)
    install_capture(async_client)

    assert sync_client.event_hooks["response"] == [capture_response]
    assert async_client.event_hooks["response"] == [acapture_response]
    with capture_attempt() as scope:
        sync_client.get("https://api.example.test/x")
    assert [c.body for c in scope.responses] == [BODY + b"/x"]
    sync_client.close()
    asyncio.run(async_client.aclose())


class _Recorder:
    def __init__(self) -> None:
        self.records: list[AttemptRecord] = []

    async def record(self, record: AttemptRecord) -> int | None:
        self.records.append(record)
        return len(self.records)


def test_recorder_is_visible_to_nested_awaits_and_gone_after_the_block() -> None:
    recorder = _Recorder()
    record = AttemptRecord(
        provider="openai", model="m", role="transcription", attempt=1, label="p1"
    )

    async def nested() -> object:
        await asyncio.sleep(0)
        return current_recorder()

    async def main() -> tuple[object, object, int | None, object, int | None]:
        with use_recorder(recorder):
            seen = await nested()
            in_task = await asyncio.create_task(nested())
            stored = await record_attempt(record)
        return seen, in_task, stored, current_recorder(), await record_attempt(record)

    seen, in_task, stored, after, without = asyncio.run(main())

    assert seen is recorder
    assert in_task is recorder
    assert stored == 1
    assert after is None
    assert without is None
    assert recorder.records == [record]


def test_redact_payloads_drops_images_and_keeps_parameters() -> None:
    image = base64.b64encode(bytes(range(256)) * 8).decode("ascii")
    params = {
        "model": "m",
        "temperature": 0.2,
        "input": [
            {"type": "input_image", "image_url": f"data:image/jpeg;base64,{image}"},
            {"type": "image", "source": {"type": "base64", "data": image}},
            {"type": "text", "text": "Transcribe."},
        ],
        "audio": b"\x00\x01",
    }

    redacted = redact_payloads(params)

    assert redacted["model"] == "m"
    assert redacted["temperature"] == 0.2
    assert redacted["input"][2] == {"type": "text", "text": "Transcribe."}
    assert redacted["input"][0]["image_url"].startswith("<omitted data:image/jpeg")
    assert redacted["input"][1]["source"]["data"].startswith("<omitted base64")
    assert redacted["audio"] == "<omitted 2 bytes>"
    assert image not in repr(redacted)


def test_redact_payloads_drops_small_media_parts_and_url_safe_base64() -> None:
    tiny = base64.b64encode(b"\x89PNG tiny").decode("ascii")
    large = base64.urlsafe_b64encode(bytes(range(256)) * 8).decode("ascii")
    body = {
        "messages": [{"source": {"media_type": "image/png", "data": tiny}}],
        "contents": [{"parts": [{"inlineData": {"mimeType": "x", "data": tiny}}]}],
        "large": large,
        "schema": {"data": {"type": "string"}, "mime_type": {"type": "string"}},
    }

    redacted = redact_payloads(body)

    assert tiny not in repr(redacted) and large not in repr(redacted)
    assert redacted["messages"][0]["source"]["media_type"] == "image/png"
    assert redacted["schema"] == body["schema"]


def test_sdk_dump_prefers_the_model_dump() -> None:
    class Model:
        def model_dump_json(self) -> str:
            return '{"a": 1}'

    assert sdk_dump(Model()) == '{"a": 1}'
    assert sdk_dump({"b": "ü"}) == '{"b": "ü"}'
