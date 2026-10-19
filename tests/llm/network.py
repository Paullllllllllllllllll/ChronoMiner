"""A mocked network behind every httpx transport, and an attempt recorder.

:class:`MockNetwork` patches ``httpx.HTTPTransport`` and
``httpx.AsyncHTTPTransport`` so that every request of the provider SDKs is
answered by an ``httpx.MockTransport`` from a list of recorded bodies. When
request *n* arrives it checks that the recorder already holds the records of
the *n - 1* attempts before it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx
import pytest

from chronominer.common.capture import AttemptRecord, current_scope


class Recorder:
    """Keep every attempt record, as the store would."""

    def __init__(self) -> None:
        self.records: list[AttemptRecord] = []

    async def record(self, record: AttemptRecord) -> int | None:
        self.records.append(record)
        return len(self.records)


@dataclass
class MockNetwork:
    """Answer requests in order from *answers*; the last answer repeats."""

    recorder: Recorder
    answers: list[tuple[int, bytes]] = field(default_factory=list)
    requests: list[httpx.Request] = field(default_factory=list)
    scoped: list[bool] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)

    def handle(self, request: httpx.Request) -> httpx.Response:
        sent = len(self.requests)
        self.requests.append(request)
        self.scoped.append(current_scope() is not None)
        stored = len(self.recorder.records)
        if self.scoped[-1] and stored != sent:
            self.violations.append(
                f"request {sent + 1} arrived with {stored} records stored"
            )
        status, body = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        return httpx.Response(
            status,
            headers={"content-type": "application/json"},
            stream=httpx.ByteStream(body),
        )

    def install(self, monkeypatch: pytest.MonkeyPatch) -> MockNetwork:
        mock = httpx.MockTransport(self.handle)

        def handle_request(
            transport: httpx.HTTPTransport, request: httpx.Request
        ) -> httpx.Response:
            return mock.handle_request(request)

        async def handle_async_request(
            transport: httpx.AsyncHTTPTransport, request: httpx.Request
        ) -> httpx.Response:
            return await mock.handle_async_request(request)

        monkeypatch.setattr(httpx.HTTPTransport, "handle_request", handle_request)
        monkeypatch.setattr(
            httpx.AsyncHTTPTransport, "handle_async_request", handle_async_request
        )
        return self


def bodies(records: list[AttemptRecord]) -> list[bytes | None]:
    """Return the raw body of each record as bytes."""
    return [r.raw.encode("utf-8") if isinstance(r.raw, str) else r.raw for r in records]


__all__ = ["MockNetwork", "Recorder", "bodies"]
