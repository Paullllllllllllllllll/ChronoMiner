"""Adjustment through the real ``ModelCaller`` against the characterization goldens.

A ``ScriptedChatModel`` answers every boundary request like the
characterization fake (the next record heading, or a scripted payload for a
heading's unit). The stored decisions must equal the per-range records of
the old ``_adjust_temp.jsonl`` (kept in ``data/adjust_golden_bodies.json``),
the rendered ``_line_ranges.txt`` the golden file, and the requests keep
the sizes the goldens record.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage

import chronominer.llm.caller as caller_module
from chronominer.common.sqlite import read_rows
from chronominer.common.store import Store
from chronominer.common.testing import FakeRequest, ScriptedChatModel
from chronominer.llm.caller import CallEnv, ModelCaller
from chronominer.output.store import DATABASE_NAME, KIND_RANGE, open_store
from chronominer.ranges.adjust import ROLE, AdjustOptions, adjust_ranges
from chronominer.settings import Settings
from chronominer.spec import ModelSpec
from tests.characterization.fakes import adjust_unit, boundary_payload
from tests.characterization.inputs import make_text_file, ranges_path, write_ranges
from tests.ranges.fakes import boundary

pytestmark = pytest.mark.integration

CHARACTERIZATION = Path(__file__).resolve().parents[1] / "characterization"
GOLDEN = CHARACTERIZATION / "golden"
SYSTEM_CHARS = 3076
"""The system prompt size of the goldens, whose runs send no adjust context."""
OLD_BODIES = Path(__file__).resolve().parent / "data" / "adjust_golden_bodies.json"
"""The per-range records of the old tool's ``_adjust_temp.jsonl``, per case."""

Case = tuple[Path, list[tuple[int, int]], dict[str, Any], set[tuple[str, int]]]
"""Golden folder, starting ranges, scripted answers by unit, and the golden
calls that retried a transient error (not made here)."""

CASES: Mapping[str, Case] = {
    "moved": (
        GOLDEN / "ranges" / "adjust" / "moved" / "adjust",
        [(1, 20), (21, 38), (39, 56), (57, 71)],
        {},
        set(),
    ),
    "merged": (
        GOLDEN / "ranges" / "adjust" / "merged" / "adjust",
        [(1, 4), (5, 20), (21, 38), (39, 56), (57, 71)],
        {adjust_unit(1): boundary(marker="Record 001.")},
        set(),
    ),
    "no_content": (
        GOLDEN / "ranges" / "adjust" / "no_content" / "adjust",
        [(1, 18), (19, 36), (37, 54), (55, 71)],
        {adjust_unit(6): boundary(no_content=True)},
        set(),
    ),
    "chunking": (
        GOLDEN / "chunking" / "adjust_line_ranges",
        [(1, 20), (21, 44), (45, 71)],
        {},
        {(adjust_unit(4), 2)},
    ),
}


class Model:
    """The scripted chat model the caller builds, and its requests."""

    def __init__(self, script: Mapping[str, Any]) -> None:
        self.script = script
        self.requests: list[FakeRequest] = []

    def respond(self, request: FakeRequest) -> AIMessage:
        unit, payload = boundary_payload(str(request.messages[-1].content))
        answer = self.script.get(unit or "", payload)
        return AIMessage(
            content=json.dumps(answer),
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
            },
        )

    def build(self, *_args: Any, **_kwargs: Any) -> ScriptedChatModel:
        return ScriptedChatModel(responder=self.respond, recorder=self.requests.append)


def _model_spec() -> ModelSpec:
    return ModelSpec(
        provider="openai",
        model="gpt-4o",
        endpoint=None,
        reasoning_effort="high",
        verbosity="medium",
        max_output_tokens=None,
        temperature=1.0,
        top_p=1.0,
        service_tier="flex",
        response_format=None,
    )


def _golden_bodies(case: str) -> list[dict[str, Any]]:
    """Return the per-range records of the case's old ``_adjust_temp.jsonl``."""
    bodies: dict[str, list[dict[str, Any]]] = json.loads(
        OLD_BODIES.read_text(encoding="utf-8")
    )
    return bodies[case]


@pytest.fixture
async def store(tmp_path: Path) -> AsyncIterator[Store]:
    async with open_store(tmp_path / "docs") as opened:
        await opened.begin_run("ranges adjust")
        yield opened


@pytest.mark.parametrize("case", sorted(CASES))
async def test_adjust_matches_the_golden_run(
    case: str, store: Store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder, starting, script, retried = CASES[case]
    model = Model(script)
    monkeypatch.setattr(caller_module, "build_chat_model", model.build)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    ledger = make_text_file(tmp_path / "docs" / "ledger.txt", 12)
    write_ranges(ledger, starting)
    caller = ModelCaller(ROLE, _model_spec(), CallEnv.create(Settings()))

    try:
        await adjust_ranges(
            store,
            ledger,
            caller=caller,
            model="gpt-4o",
            options=AdjustOptions(boundary_type="FlatEntries", context=None),
        )
    finally:
        await caller.aclose()

    expected = (folder / "outputs" / "docs" / "ledger_line_ranges.txt").read_text(
        encoding="utf-8"
    )
    assert ranges_path(ledger).read_text(encoding="utf-8") == expected
    rows = read_rows(ledger.parent / DATABASE_NAME, "units", order_by=["idx"])
    decisions = [
        json.loads(row["provenance"])["decision"]
        for row in rows
        if row["kind"] == KIND_RANGE
    ]
    assert decisions == _golden_bodies(case)
    golden = json.loads((folder / "requests.json").read_text(encoding="utf-8"))
    expected_sizes = sorted(
        tuple(
            block["chars"]
            for message in call["messages"]
            for block in message["blocks"]
        )
        for call in golden["calls"]
        if call["role"] == "adjust" and (call["unit"], call["attempt"]) not in retried
    )
    sizes = sorted(
        tuple(len(str(message.content)) for message in request.messages)
        for request in model.requests
    )
    assert sizes == expected_sizes
    assert all(size[0] == SYSTEM_CHARS for size in sizes)
    responses = read_rows(ledger.parent / DATABASE_NAME, "responses")
    assert len(responses) == len(model.requests)
    assert {response["role"] for response in responses} == {ROLE}
    assert all(response["kind"] == KIND_RANGE for response in responses)
