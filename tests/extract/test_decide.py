"""The relevance pre-pass: question, thresholds, skips and repair selection."""

from __future__ import annotations

import asyncio
import base64
import io
import json
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest
from PIL import Image as PILImage

from chronominer.common.decision_runner import DecisionRunner
from chronominer.common.decisions import Image
from chronominer.common.settings_schema import Decisions
from chronominer.extract.chunks import TextChunk
from chronominer.extract.decide import (
    SKIPPED_OUTCOME,
    ChunkDecider,
    ChunkDecision,
    relevance_question,
)
from chronominer.extract.images import PageImage
from chronominer.extract.llm import Extractor
from chronominer.llm.caller import CallEnv
from chronominer.settings import Settings
from chronominer.spec import DecisionSpec, ResolvedSchema, resolve_repair

SCHEMA = ResolvedSchema(
    "WageRecords",
    Path("wage_records.json"),
    {"name": "WageRecords", "description": "Payrolls and wage books."},
)
SPEC = DecisionSpec("shadow", "typesafe/jev-1.14", None, None)


@pytest.fixture(autouse=True)
def _key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")


def _env(**block: Any) -> CallEnv:
    return CallEnv.create(Settings(decisions=replace(Decisions(), **block)))


async def _stream(chunks: list[TextChunk]) -> AsyncIterator[TextChunk]:
    for chunk in chunks:
        yield chunk


def test_the_question_names_the_schema_unless_the_wrapper_sets_one() -> None:
    built = relevance_question(SCHEMA)
    assert "WageRecords (Payrolls and wage books.)" in built.instructions
    custom = replace(SCHEMA, schema={"decision_question": "Lists a wage."})
    assert relevance_question(custom).instructions == "Lists a wage."
    with pytest.raises(ValueError, match="decision_question"):
        relevance_question(replace(SCHEMA, schema={"decision_question": " "}))


def test_each_chunk_is_one_request_with_the_document_and_schema() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        relevant = "wage" in body["state"][1]["text"]
        answer = {"type": "noul", "noul": 0.9 if relevant else 0.02}
        return httpx.Response(
            200,
            json={"answers": {"relevant": answer}, "usage": {"input_tokens": 50}},
        )

    env = _env()
    decider = ChunkDecider(SPEC, env, SCHEMA, concurrency=2)
    decider._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    chunks = [
        TextChunk(0, 1, 9, "a wage of 3 shillings"),
        TextChunk(1, 10, 19, "a sermon"),
        TextChunk(2, 20, 29, "wage book"),
    ]

    decisions = asyncio.run(decider.classify("ledger", _stream(chunks)))

    assert [(d.idx, d.relevance) for d in decisions] == [(0, 0.9), (1, 0.02), (2, 0.9)]
    assert json.loads(seen[0]["state"][0]["text"]) == {
        "document": "ledger",
        "schema": "WageRecords",
    }
    assert env.usage.get("decide").input_tokens == 150


def test_the_pre_pass_asks_through_the_shared_runner_as_before() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        refused = "sermon" in body["state"][1]["text"]
        answer = (
            {"type": "refusal", "reason": "unreadable"}
            if refused
            else {"type": "noul", "noul": 0.7}
        )
        usage = {"input_tokens": 5, "output_tokens": 1}
        return httpx.Response(
            200, json={"answers": {"relevant": answer}, "usage": usage}
        )

    env = _env()
    decider = ChunkDecider(SPEC, env, SCHEMA)
    decider._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    chunks = [TextChunk(0, 1, 9, "a wage"), TextChunk(1, 10, 19, "a sermon")]

    decisions = asyncio.run(decider.classify("ledger", _stream(chunks)))

    assert isinstance(decider.runner, DecisionRunner)
    assert [(d.idx, d.relevance) for d in decisions] == [(0, 0.7)]
    instructions = relevance_question(SCHEMA).instructions
    assert seen[0]["model"] == SPEC.model
    assert seen[0]["questions"] == {
        "relevant": {"type": "noul", "instructions": instructions}
    }
    assert [part["type"] for part in seen[0]["state"]] == ["text", "text"]
    assert seen[0]["state"][1]["text"] == "a wage"
    assert env.usage.get("decide").input_tokens == 10


@pytest.mark.parametrize("mode", ["RGB", "L"])
def test_page_images_go_out_as_the_earlier_downscale_sent_them(mode: str) -> None:
    buffer = io.BytesIO()
    PILImage.new(mode, (1200, 800), 180).save(buffer, "JPEG", quality=95)
    data = buffer.getvalue()
    with PILImage.open(io.BytesIO(data)) as opened:
        earlier = opened.convert("RGB")
    earlier.thumbnail((768, 768))
    expected = io.BytesIO()
    earlier.save(expected, "JPEG", quality=85)
    page = PageImage(0, "page_0001", base64=base64.b64encode(data).decode())

    part = ChunkDecider(SPEC, _env(), SCHEMA)._content(page)

    assert isinstance(part, Image)
    assert part.data == expected.getvalue()


def test_a_threshold_applies_only_to_its_request_fingerprint() -> None:
    decider = ChunkDecider(SPEC, _env(), SCHEMA)
    text = decider.fingerprint(None)
    assert decider.threshold(text) is None
    tuned = ChunkDecider(SPEC, _env(thresholds={text: 0.1}), SCHEMA)
    assert tuned.threshold(text) == 0.1
    assert decider.fingerprint({"dpi": 300}) != text
    other = ChunkDecider(
        SPEC, _env(), replace(SCHEMA, schema={"decision_question": "Other."})
    )
    assert other.fingerprint(None) != text


def test_a_missing_key_fails_before_any_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY")
    with pytest.raises(OSError, match="OPENROUTER_API_KEY"):
        ChunkDecider(SPEC, _env(), SCHEMA)


class _Store:
    def __init__(self) -> None:
        self.stored: list[Any] = []

    async def write(self, lease: object, func: Any) -> None:
        self.stored.append(func)


class _Handle:
    name = "ledger"
    doc_id = 1
    run_id = "run-1"
    lease = object()

    def __init__(self) -> None:
        self.writes: list[tuple[int, dict[str, Any]]] = []
        self.store = _Store()

    async def write(self, idx: int, **fields: Any) -> None:
        self.writes.append((idx, fields))


class _Decider:
    question_sha = "q"
    dialect = "openrouter"
    spec = replace(SPEC, mode="apply")

    def __init__(self, threshold: float | None) -> None:
        self._threshold = threshold

    async def classify(self, document: str, chunks: Any) -> list[ChunkDecision]:
        return [
            ChunkDecision(c.index, 0.01 if c.index == 0 else 0.8) async for c in chunks
        ]

    def fingerprint(self, upstream: Any = None) -> str:
        return "fp"

    def threshold(self, fingerprint: str) -> float | None:
        return self._threshold


def _run_decide(mode: str, threshold: float | None) -> tuple[list[int], _Handle, Any]:
    warnings: list[str] = []
    job = type("Job", (), {})()
    job.spec = type("Spec", (), {"decision": replace(SPEC, mode=mode)})()
    job.events = type("Events", (), {"warning": staticmethod(warnings.append)})()
    extractor = Extractor(job)
    extractor._decider = _Decider(threshold)  # type: ignore[assignment]
    handle = _Handle()

    async def source(indices: list[int]) -> AsyncIterator[TextChunk]:
        for idx in indices:
            yield TextChunk(idx, idx, idx, "text")

    needed = asyncio.run(extractor._decide(handle, [0, 1], source, None))  # type: ignore[arg-type]
    return needed, handle, warnings


def test_apply_skips_chunks_below_the_threshold_and_records_why() -> None:
    needed, handle, _ = _run_decide("apply", 0.05)
    assert needed == [1]
    assert len(handle.store.stored) == 1
    [(idx, fields)] = handle.writes
    assert (idx, fields["status"], fields["outcome"]) == (0, "done", SKIPPED_OUTCOME)
    assert fields["provenance"]["decision"]["fingerprint"] == "fp"
    assert json.loads(fields["content"]) == {
        "contains_no_content_of_requested_type": True,
        "entries": [],
    }


def test_a_failed_pre_pass_extracts_every_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def broken(self: Any, document: str, chunks: Any) -> list[ChunkDecision]:
        raise httpx.ConnectError("down")

    monkeypatch.setattr(_Decider, "classify", broken)
    needed, handle, warnings = _run_decide("apply", 0.05)
    assert (needed, handle.writes, handle.store.stored) == ([0, 1], [], [])
    assert "pre-pass failed" in warnings[0]


def test_shadow_and_apply_without_a_threshold_skip_nothing() -> None:
    needed, handle, _ = _run_decide("shadow", 0.05)
    assert (needed, handle.writes) == ([0, 1], [])
    needed, handle, warnings = _run_decide("apply", None)
    assert (needed, handle.writes) == ([0, 1], [])
    assert "No validated decision threshold" in warnings[0]


def test_repair_skipped_is_a_flag_of_the_spec() -> None:
    assert resolve_repair({"input": ".", "skipped": True}).spec.skipped
    assert not resolve_repair({"input": "."}).spec.skipped
