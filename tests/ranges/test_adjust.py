"""Line-range adjustment against the store with a scripted boundary caller.

The ledger has twelve records with headings on lines 1, 7, 13 ... 67 of 71;
the default answer moves a range start to the next heading.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from chronominer.common.selection import parse_page_range
from chronominer.common.sqlite import read_rows
from chronominer.common.store import Store
from chronominer.llm.prompts import boundary_prompt
from chronominer.output.store import DATABASE_NAME, KIND_RANGE, open_store
from chronominer.ranges.adjust import (
    BOUNDARY_SENTINEL,
    AdjustDegradedError,
    AdjustOptions,
    adjust_files,
    adjust_ranges,
    boundary_user_text,
)
from chronominer.ranges.decision import SEMANTIC_BOUNDARY_SCHEMA
from chronominer.ranges.lines import ranges_path, read_ranges
from chronominer.settings import PromptCaching
from tests.characterization.fakes import permanent_error, transient_error
from tests.characterization.inputs import make_text_file, write_ranges
from tests.ranges.fakes import FakeCaller, boundary, default_answer

pytestmark = pytest.mark.unit

MID_RECORD = [(1, 20), (21, 38), (39, 56), (57, 71)]
ADJUSTED = [(1, 24), (25, 42), (43, 60), (61, 71)]
OPTIONS = AdjustOptions(boundary_type="FlatEntries", context="Records start a unit.")


@pytest.fixture
def ledger(tmp_path: Path) -> Path:
    text = make_text_file(tmp_path / "ledger.txt", 12)
    write_ranges(text, MID_RECORD)
    return text


@pytest.fixture
async def store(tmp_path: Path) -> AsyncIterator[Store]:
    async with open_store(tmp_path) as opened:
        await opened.begin_run("ranges adjust")
        yield opened


async def _adjust(store: Store, text: Path, caller: FakeCaller, **changes: Any) -> Any:
    options = AdjustOptions(**{**vars(OPTIONS), **changes})
    return await adjust_ranges(
        store, text, caller=caller, model="gpt-4o", options=options
    )


def _units(text: Path) -> list[dict[str, Any]]:
    rows = read_rows(text.parent / DATABASE_NAME, "units", order_by=["idx"])
    return [row for row in rows if row["kind"] == KIND_RANGE]


def _decisions(text: Path) -> list[dict[str, Any] | None]:
    return [
        (json.loads(row["provenance"]) or {}).get("decision")
        if row["provenance"]
        else None
        for row in _units(text)
    ]


def test_the_system_prompt_names_the_type_and_carries_the_context() -> None:
    prompt = boundary_prompt(
        "FlatEntries", SEMANTIC_BOUNDARY_SCHEMA, "Records start a unit."
    )
    assert "schema name: FlatEntries" in prompt
    assert "Records start a unit." in prompt
    assert '"semantic_marker":{"type":"string"' in prompt
    bare = boundary_prompt("FlatEntries", SEMANTIC_BOUNDARY_SCHEMA, None)
    assert "{{CONTEXT}}" not in bare
    assert "Rules for this corpus and work:" not in bare


@pytest.mark.parametrize(
    ("provider", "model", "caching", "control"),
    [
        ("anthropic", "claude-sonnet-4-5", PromptCaching(), {"type": "ephemeral"}),
        (
            "anthropic",
            "claude-sonnet-4-5",
            PromptCaching(anthropic_ttl="1h"),
            {"type": "ephemeral", "ttl": "1h"},
        ),
        ("anthropic", "claude-sonnet-4-5", PromptCaching(enabled=False), None),
        (
            "openrouter",
            "anthropic/claude-sonnet-4.5",
            PromptCaching(),
            {"type": "ephemeral"},
        ),
        (
            "openrouter",
            "anthropic/claude-sonnet-4.5",
            PromptCaching(openrouter_cache_control=False),
            None,
        ),
        ("openrouter", "openai/gpt-5.4", PromptCaching(), None),
        ("openai", "gpt-4o", PromptCaching(), None),
    ],
)
async def test_the_system_prompt_is_cached_as_in_extraction(
    store: Store,
    ledger: Path,
    provider: str,
    model: str,
    caching: PromptCaching,
    control: dict[str, str] | None,
) -> None:
    caller = FakeCaller(provider=provider)
    options = AdjustOptions(**{**vars(OPTIONS), "prompt_caching": caching})

    await adjust_ranges(store, ledger, caller=caller, model=model, options=options)

    prompt = boundary_prompt(
        "FlatEntries", SEMANTIC_BOUNDARY_SCHEMA, "Records start a unit."
    )
    expected: Any = prompt
    if control is not None:
        expected = [{"type": "text", "text": prompt, "cache_control": control}]
    assert caller.systems
    assert all(content == expected for content in caller.systems)


def test_the_user_text_marks_the_range_start_and_failed_markers() -> None:
    lines = ["one", "two", "three", "four"]
    text = boundary_user_text(lines, 3, (2, 4), ["  bad marker ", ""])
    assert text.startswith(f"Input text:\n\ntwo\n{BOUNDARY_SENTINEL}\nthree\nfour\n")
    assert text.endswith("excludes these markers:\n- bad marker\n")
    assert BOUNDARY_SENTINEL not in boundary_user_text(lines, 1, (2, 4))


async def test_ranges_move_to_the_next_heading(store: Store, ledger: Path) -> None:
    caller = FakeCaller()

    report = await _adjust(store, ledger, caller)

    assert report.status == "adjusted"
    assert read_ranges(ranges_path(ledger)) == ADJUSTED
    assert list(report.ranges) == ADJUSTED
    assert (report.decided, report.calls) == (4, 4)
    units = _units(ledger)
    assert [row["status"] for row in units] == ["done"] * 4
    assert [row["outcome"] for row in units] == ["on_target", "moved", "moved", "moved"]
    assert [json.loads(row["content"]) for row in units] == [list(r) for r in ADJUSTED]
    assert [row["source_index"] for row in units] == ["1-20", "21-38", "39-56", "57-71"]
    decisions = _decisions(ledger)
    assert decisions[1] is not None
    assert decisions[1]["adjusted_range"] == [25, 38]
    assert decisions[1]["decision"]["semantic_marker"] == "Record 005."


async def test_a_range_without_content_is_deleted(store: Store, ledger: Path) -> None:
    write_ranges(ledger, [(1, 18), (19, 36), (37, 54), (55, 71)])

    def answer(text: str, label: str) -> Any:
        if "range 3 " in label:
            return boundary(no_content=True)
        return default_answer(text, label)

    report = await _adjust(store, ledger, FakeCaller(answer))

    assert read_ranges(ranges_path(ledger)) == [(1, 18), (19, 36), (55, 71)]
    assert report.calls == 5
    units = _units(ledger)
    assert units[2]["outcome"] == "deleted"
    assert units[2]["content"] is None


async def test_resume_continues_a_partial_run_and_force_redoes_it(
    store: Store, ledger: Path
) -> None:
    caller = FakeCaller()

    partial = await _adjust(store, ledger, caller, pages=parse_page_range("first:2"))
    assert len(caller.requests) == 2
    assert partial.decided == 2
    assert read_ranges(ranges_path(ledger)) == [(1, 24), (25, 38), (39, 56), (57, 71)]

    resumed = await _adjust(store, ledger, caller)
    assert len(caller.requests) == 4
    assert resumed.decided == 2
    assert read_ranges(ranges_path(ledger)) == ADJUSTED

    complete = await _adjust(store, ledger, caller)
    assert complete.status == "complete"
    assert len(caller.requests) == 4

    forced = await _adjust(store, ledger, caller, force=True)
    assert forced.decided == 4
    assert len(caller.requests) == 8
    assert read_ranges(ranges_path(ledger)) == ADJUSTED
    assert [row["source_index"] for row in _units(ledger)] == [
        f"{start}-{end}" for start, end in ADJUSTED
    ]


async def test_a_hand_edited_file_is_read_as_the_starting_ranges(
    store: Store, ledger: Path
) -> None:
    caller = FakeCaller()
    await _adjust(store, ledger, caller)

    write_ranges(ledger, [(1, 33), (34, 71)])
    report = await _adjust(store, ledger, caller)

    assert report.decided == 2
    assert read_ranges(ranges_path(ledger)) == [(1, 36), (37, 71)]
    assert [row["source_index"] for row in _units(ledger)] == ["1-33", "34-71"]


async def test_changed_settings_adjust_again(store: Store, ledger: Path) -> None:
    caller = FakeCaller()
    await _adjust(store, ledger, caller)

    report = await _adjust(store, ledger, caller, context="Another context.")

    assert report.decided == 4
    assert any("settings" in warning for warning in report.warnings)
    assert read_ranges(ranges_path(ledger)) == ADJUSTED


async def test_a_missing_file_is_generated_when_chunk_tokens_are_given(
    store: Store, tmp_path: Path
) -> None:
    text = make_text_file(tmp_path / "fresh.txt", 12)

    with pytest.raises(FileNotFoundError):
        await _adjust(store, text, FakeCaller())

    report = await _adjust(store, text, FakeCaller(), chunk_tokens=60)

    assert report.decided > 1
    pairs = read_ranges(ranges_path(text))
    assert pairs[0][0] == 1
    assert pairs[-1][1] == 71


async def test_a_non_utf8_text_is_read(store: Store, tmp_path: Path) -> None:
    text = tmp_path / "latin.txt"
    text.write_bytes("\n".join(f"Küche {i}" for i in range(1, 21)).encode("cp1252"))
    write_ranges(text, [(1, 10), (11, 20)])

    report = await _adjust(
        store, text, FakeCaller(lambda _t, _l: boundary(on_target=True))
    )

    assert report.decided == 2
    assert read_ranges(ranges_path(text)) == [(1, 10), (11, 20)]


async def test_a_permanent_error_fails_the_file_and_a_rerun_retries_it(
    store: Store, ledger: Path
) -> None:
    def failing(text: str, label: str) -> Any:
        if "range 2 " in label:
            return permanent_error()
        return default_answer(text, label)

    with pytest.raises(Exception, match="400"):
        await _adjust(store, ledger, FakeCaller(failing))

    assert read_ranges(ranges_path(ledger)) == MID_RECORD
    units = _units(ledger)
    assert [row["status"] for row in units] == ["done", "failed", "pending", "pending"]
    assert "400" in units[1]["error"]

    caller = FakeCaller()
    report = await _adjust(store, ledger, caller)

    assert report.decided == 3
    assert all("range 1 " not in label for _, _, label in caller.requests)
    assert read_ranges(ranges_path(ledger)) == ADJUSTED


async def test_partial_degradation_warns_and_succeeds(
    store: Store, ledger: Path
) -> None:
    def flaky(text: str, label: str) -> Any:
        if "range 2 window 0" in label:
            return transient_error()
        return default_answer(text, label)

    report = await _adjust(store, ledger, FakeCaller(flaky))

    assert report.degraded == 0
    assert any("treated as unsure" in warning for warning in report.warnings)
    assert read_ranges(ranges_path(ledger))[0] == (1, 24)


async def test_a_fully_degraded_file_raises_and_is_retried(
    store: Store, ledger: Path
) -> None:
    with pytest.raises(AdjustDegradedError) as raised:
        await _adjust(store, ledger, FakeCaller(lambda _t, _l: transient_error()))

    assert raised.value.degraded == raised.value.total > 0
    assert read_ranges(ranges_path(ledger)) == MID_RECORD
    assert {row["outcome"] for row in _units(ledger)} == {"degraded"}

    report = await _adjust(store, ledger, FakeCaller())
    assert report.decided == 4
    assert read_ranges(ranges_path(ledger)) == ADJUSTED


async def test_cancellation_propagates(store: Store, ledger: Path) -> None:
    started = asyncio.Event()

    class Hanging(FakeCaller):
        async def call(self, messages: Any, request: Any, *, label: str) -> Any:
            started.set()
            await asyncio.sleep(30)
            return await super().call(messages, request, label=label)

    task = asyncio.create_task(_adjust(store, ledger, Hanging()))
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


async def test_adjust_files_uses_the_store_of_each_folder(tmp_path: Path) -> None:
    first = make_text_file(tmp_path / "a" / "ledger.txt", 12)
    second = make_text_file(tmp_path / "b" / "ledger.txt", 12)
    missing = make_text_file(tmp_path / "b" / "other.txt", 2)
    write_ranges(first, MID_RECORD)
    write_ranges(second, MID_RECORD)

    outcomes = await adjust_files(
        [first, second, missing], caller=FakeCaller(), model="gpt-4o", options=OPTIONS
    )

    assert [type(outcome).__name__ for _, outcome in outcomes] == [
        "AdjustReport",
        "AdjustReport",
        "FileNotFoundError",
    ]
    for text in (first, second):
        assert read_ranges(ranges_path(text)) == ADJUSTED
        assert (text.parent / DATABASE_NAME).is_file()
    runs = read_rows(second.parent / DATABASE_NAME, "runs")
    assert [(run["command"], run["exit_code"]) for run in runs] == [
        ("ranges adjust", 1)
    ]
