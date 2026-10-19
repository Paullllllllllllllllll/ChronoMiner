"""Smoke cases of the characterization harness, one per command.

Every command runs through ``run_tool`` with exit code 0 and no network.
The golden case runs twice in fresh roots; both snapshots must match before
the run is compared with its golden files under ``golden/smoke/``.
"""

from __future__ import annotations

import inspect
import io
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from chronominer.cli.parser import build_parser
from tests.characterization.adapter import (
    DEFAULT_SCHEMA,
    RUN_TOOL_KEYWORDS,
    RunResult,
    build_argv,
    load_schema,
    run_tool,
)
from tests.characterization.conftest import CaseRoot, GoldenCheck
from tests.characterization.fakes import (
    FakeLLM,
    Reply,
    adjust_unit,
    example_payload,
    permanent_error,
    record_unit,
    transient_error,
)
from tests.characterization.fakes_batch import FakeBatchService
from tests.characterization.golden import compare, snapshot
from tests.characterization.inputs import heading_lines, ledger_text, make_text_file
from tests.characterization.seams import load_tool_modules, replace, tool_modules

pytestmark = pytest.mark.integration

RECORDS = 12
_RUN_TOOL_OWN = {"command", "llm", "root", "config", "env", "batch_service"}
_SAMPLES: dict[str, Any] = {
    "schema": DEFAULT_SCHEMA,
    "provider": "anthropic",
    "input": "docs/ledger.txt",
    "output": "out",
    "output_mode": "mirror",
    "input_type": "text",
    "chunking": "line_ranges",
    "chunk_size": 100,
    "first_n_chunks": 2,
    "last_n_chunks": 2,
    "page_range": "1-2",
    "batch": True,
    "resume": True,
    "force": True,
    "dry_run": True,
    "json": True,
    "quiet": True,
    "model": "gpt-4o",
    "reasoning_effort": "low",
    "verbosity": "low",
    "max_output_tokens": 512,
    "temperature": 0.5,
    "top_p": 0.9,
    "context": "none",
    "fallback_context": "grain ledgers",
    "context_image": True,
    "concurrency_limit": 2,
    "delay": 0.5,
    "image_detail": "high",
    "service_tier": "flex",
    "tokens": 100,
    "context_window": 4,
    "files": ["ledger_temp.jsonl"],
    "verbose": True,
}
_REQUIRED: dict[str, dict[str, Any]] = {
    "generate_line_ranges": {"input": "docs"},
    "adjust_line_ranges": {"input": "docs/ledger.txt"},
}


def run_twice(case_root: CaseRoot, build: Callable[[Path], RunResult]) -> RunResult:
    """Run *build* in two fresh roots and return the second result.

    Fails when the two snapshots differ, so a golden never records output
    that changes between identical runs.
    """
    first = snapshot(build(case_root("run1")))
    result = build(case_root("run2"))
    problems = compare(first, snapshot(result))
    assert not problems, "two identical runs differ:\n" + "\n".join(problems)
    return result


def _ledger(root: Path) -> Path:
    return make_text_file(root / "docs" / "ledger.txt", RECORDS)


def _parse(argv: list[str]) -> None:
    """Parse *argv* with the ``chronominer`` parser; a parse error raises."""
    build_parser().parse_args(argv)


def _records(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))["records"]


def test_every_run_tool_keyword_becomes_parsed_arguments(tmp_path: Path) -> None:
    params = set(inspect.signature(run_tool).parameters) - _RUN_TOOL_OWN
    assert params == set().union(*RUN_TOOL_KEYWORDS.values())
    assert params <= set(_SAMPLES)

    for command, names in RUN_TOOL_KEYWORDS.items():
        for name in sorted(names):
            keywords = {**_REQUIRED.get(command, {}), name: _SAMPLES[name]}
            argv, _ = build_argv(command, keywords, root=tmp_path)  # type: ignore[arg-type]
            _parse(argv)


def test_a_keyword_the_command_lacks_raises() -> None:
    with pytest.raises(TypeError, match="tokens"):
        build_argv("extract", {"tokens": 100})


def test_replace_swaps_the_object_in_every_tool_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import chronominer.settings as settings_module

    original = settings_module.Settings
    load_tool_modules()
    binders = [
        module
        for module in tool_modules()
        if any(value is original for value in vars(module).values())
    ]
    assert len(binders) > 1
    assert {module.__name__.split(".")[0] for module in binders} == {"chronominer"}
    stand_in = object()

    with monkeypatch.context() as mp:
        replaced = replace(mp, "chronominer.settings", "Settings", stand_in)

        assert replaced is original
        for module in binders:
            assert not any(value is original for value in vars(module).values())
            assert any(value is stand_in for value in vars(module).values())

    for module in binders:
        assert any(value is original for value in vars(module).values())


def test_openai_batch_fake_answers_schema_json_from_the_nested_body(
    fake_batch: FakeBatchService,
) -> None:
    schema = load_schema(DEFAULT_SCHEMA)
    assert schema is not None
    body = {
        "model": "gpt-4o",
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": "Extract the records."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": f"Input text:\n{ledger_text(2)}"}
                ],
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": schema["name"],
                "schema": schema["schema"],
                "strict": True,
            }
        },
    }
    line = {"custom_id": "ledger-chunk-1", "method": "POST", "url": "/v1/responses"}
    payload = json.dumps({**line, "body": body}) + "\n"
    client = fake_batch.client_class("openai")()

    uploaded = client.files.create(
        file=io.BytesIO(payload.encode("utf-8")), purpose="batch"
    )
    created = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window="24h",
    )
    finished = client.batches.retrieve(created.id)
    output = client.files.content(finished.output_file_id).read().decode("utf-8")
    result = json.loads(output.splitlines()[0])
    output_text = result["response"]["body"]["output"][0]["content"][0]["text"]

    assert finished.status == "completed"
    assert json.loads(output_text) == example_payload(schema["schema"], record_unit(0))


def test_extract_text_sync(
    fake_llm: FakeLLM, case_root: CaseRoot, golden: GoldenCheck
) -> None:
    def build(root: Path) -> RunResult:
        return run_tool(
            "extract",
            llm=fake_llm,
            root=root,
            input=_ledger(root),
            output=root / "out",
        )

    result = run_twice(case_root, build)

    assert result.exit_code == 0, result.stdout + result.stderr
    assert result.summary is not None
    assert (result.summary["items_complete"], result.summary["items_failed"]) == (
        1,
        0,
    )
    ledger = result.root / "docs" / "ledger.txt"
    assert result.output_json(ledger) in result.created
    assert result.output_json(ledger).with_suffix(".csv") in result.created
    assert result.database in result.created
    assert len(result.calls) > 1
    assert {call["role"] for call in result.calls} == {"extract"}
    assert result.calls[0]["unit"] == record_unit(0)
    assert result.calls[0]["structured"]["method"] == "json_schema"
    assert [model["constructor"] for model in result.models] == ["ChatOpenAI"]
    golden("smoke/extract_text", result)


@pytest.mark.parametrize(
    ("provider", "constructor"),
    [
        ("openai", "ChatOpenAI"),
        ("anthropic", "ChatAnthropic"),
        ("google", "ChatGoogleGenerativeAI"),
        ("openrouter", "ChatOpenAI"),
        ("custom", "ChatOpenAI"),
    ],
)
def test_extract_reaches_every_provider_class(
    provider: str,
    constructor: str,
    fake_llm: FakeLLM,
    case_root: CaseRoot,
) -> None:
    root = case_root()

    result = run_tool(
        "extract", llm=fake_llm, root=root, input=_ledger(root), provider=provider
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert {model["constructor"] for model in result.models} == {constructor}
    assert {model["provider"] for model in result.models} == {provider}
    assert result.calls[0]["unit"] == record_unit(0)


@pytest.mark.parametrize("provider", ["openai", "anthropic", "google"])
def test_extract_batch_submit_then_check_batches(
    provider: str,
    fake_llm: FakeLLM,
    fake_batch: FakeBatchService,
    case_root: CaseRoot,
) -> None:
    root = case_root()
    ledger = _ledger(root)

    submitted = run_tool(
        "extract",
        llm=fake_llm,
        root=root,
        input=ledger,
        provider=provider,
        batch=True,
        batch_service=fake_batch,
    )

    assert submitted.exit_code == 0, submitted.stdout + submitted.stderr
    assert submitted.calls == []
    assert [state["provider"] for state in fake_batch.batch_states()] == [provider]
    assert submitted.batch is not None
    assert submitted.batch["requests"][0]["unit"] == record_unit(0)

    checked = run_tool(
        "check_batches",
        llm=fake_llm,
        root=root,
        provider=provider,
        batch_service=fake_batch,
    )

    assert checked.exit_code == 0, checked.stdout + checked.stderr
    assert checked.summary is not None
    assert checked.summary["finalized"] == 1
    records = _records(checked.output_json(ledger))
    assert len(records) == len(submitted.batch["requests"])
    first = json.loads(records[0]["response"]["output_text"])
    assert first == example_payload(load_schema(DEFAULT_SCHEMA), record_unit(0))


def test_cancel_batches_cancels_a_pending_batch(
    fake_llm: FakeLLM, fake_batch: FakeBatchService, case_root: CaseRoot
) -> None:
    root = case_root()
    fake_batch.polls_until_complete = 99
    submitted = run_tool(
        "extract",
        llm=fake_llm,
        root=root,
        input=_ledger(root),
        batch=True,
        batch_service=fake_batch,
    )
    assert submitted.exit_code == 0, submitted.stdout + submitted.stderr

    cancelled = run_tool(
        "cancel_batches",
        llm=fake_llm,
        root=root,
        force=True,
        batch_service=fake_batch,
    )

    assert cancelled.exit_code == 0, cancelled.stdout + cancelled.stderr
    assert [state["state"] for state in fake_batch.batch_states()] == ["cancelled"]
    assert cancelled.batch is not None
    assert [call["call"] for call in cancelled.batch["calls"]][-1] == (
        "openai.batches.cancel"
    )


def test_repair_reruns_a_failed_chunk(fake_llm: FakeLLM, case_root: CaseRoot) -> None:
    root = case_root()
    ledger = _ledger(root)
    fake_llm.script(record_unit(3), always=Reply(permanent_error()))
    first = run_tool("extract", llm=fake_llm, root=root, input=ledger)
    assert first.exit_code == 1, first.console
    fake_llm.script(record_unit(3), always=None)

    repaired = run_tool("repair", llm=fake_llm, root=root)

    assert repaired.exit_code == 0, repaired.console
    assert repaired.summary is not None
    assert repaired.summary["repaired"] == 1
    assert "Repaired 1 chunk(s)" in repaired.stderr
    assert [call["unit"] for call in repaired.calls] == [record_unit(3)]
    records = _records(repaired.output_json(ledger))
    assert [record["chunk_index"] for record in records] == [1, 2, 3, 4]


def test_generate_line_ranges(fake_llm: FakeLLM, case_root: CaseRoot) -> None:
    root = case_root()
    ledger = _ledger(root)

    result = run_tool(
        "generate_line_ranges", llm=fake_llm, root=root, input=ledger, tokens=60
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    ranges = ledger.with_name("ledger_line_ranges.txt")
    assert ranges in result.created
    assert len(ranges.read_text(encoding="utf-8").splitlines()) > 1
    assert result.calls == []


def test_adjust_line_ranges_with_a_scripted_fake(
    fake_llm: FakeLLM, case_root: CaseRoot
) -> None:
    root = case_root()
    ledger = _ledger(root)
    generated = run_tool(
        "generate_line_ranges", llm=fake_llm, root=root, input=ledger, tokens=60
    )
    assert generated.exit_code == 0, generated.stdout + generated.stderr
    for index in range(RECORDS):
        fake_llm.reply(adjust_unit(index), 1, Reply(transient_error()))

    result = run_tool("adjust_line_ranges", llm=fake_llm, root=root, input=ledger)

    assert result.exit_code == 0, result.stdout + result.stderr
    assert result.calls
    assert {call["role"] for call in result.calls} == {"adjust"}
    retried = [call for call in result.calls if call["attempt"] == 2]
    assert retried
    starts = [
        int(line.strip("()").split(",")[0])
        for line in ledger.with_name("ledger_line_ranges.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert set(starts) - {1} <= set(heading_lines(ledger))
