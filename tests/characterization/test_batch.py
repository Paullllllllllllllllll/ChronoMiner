"""Characterization of batch extraction through the fake batch services.

Each case submits with ``extract --batch`` and finalizes with
``check_batches``; the real OpenAI, Anthropic and Google backends run
against the fake SDK clients of ``fakes_batch``. A batch completes on its
second status call, so the first ``check_batches`` finds it in progress, the
second finalizes it into the store, ``_output.json`` and CSV, and a third
finds nothing left to check. Every stage is snapshotted: the files the run
created, the files it changed (under ``changed/``), its stdout, the JSON
summary and ``requests.json``, whose ``batch`` entry holds the client
constructions, the API calls and the submitted request bodies. Each case
runs twice in fresh roots; both snapshots must match before they are
compared with the golden files under ``golden/batch/``.

The stage helpers here are shared with ``test_repair`` and ``test_cancel``.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from chronominer.common.testing import WriteGuard
from tests.characterization.adapter import (
    DEFAULT_SCHEMA,
    Command,
    RunResult,
    load_schema,
    run_tool,
)
from tests.characterization.fakes import (
    FakeLLM,
    Reply,
    example_payload,
    page_unit,
    permanent_error,
    record_unit,
)
from tests.characterization.fakes_batch import FakeBatchService, Provider
from tests.characterization.golden import normalizer_for, snapshot
from tests.characterization.inputs import make_scanned_pdf, make_text_file
from tests.characterization.stages import check_stages, current_files, run_twice

pytestmark = pytest.mark.integration

PROVIDERS: tuple[Provider, ...] = ("openai", "anthropic", "google")
RECORDS = 6
"""Ledger records; at 160 tokens per chunk they form two chunks."""

CHUNK_UNITS = (record_unit(0), record_unit(3))
"""Unit of each chunk of the ledger: its first record."""

PAGES = 2
CREATE_CALLS: dict[Provider, str] = {
    "openai": "openai.batches.create",
    "anthropic": "anthropic.messages.batches.create",
    "google": "google.batches.create",
}
"""The API call that creates a batch, per provider."""


@dataclasses.dataclass
class Stage:
    """One command of a case: its result, snapshot, changed files and calls."""

    result: RunResult
    files: dict[str, str]
    changed: list[Path]
    batch_calls: list[str]


def _contents(root: Path) -> dict[Path, bytes]:
    return {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}


def _safe_name(name: str) -> str:
    """Return a golden file name without the ``<>`` Windows forbids."""
    return name.replace("<", "{").replace(">", "}")


def _stage_files(result: RunResult, changed: list[Path]) -> dict[str, str]:
    files = snapshot(result)
    files.update(current_files(result, changed, "changed"))
    files["stderr.txt"] = normalizer_for(result).text(result.stderr)
    return {_safe_name(name): content for name, content in files.items()}


def run_stage(command: Command, service: FakeBatchService, **kwargs: Any) -> Stage:
    """Run one command and snapshot what it created and changed in the root."""
    root: Path = kwargs["root"]
    before = _contents(root)
    result = run_tool(command, batch_service=service, **kwargs)
    after = _contents(root)
    changed = sorted(
        path for path, data in after.items() if path in before and before[path] != data
    )
    calls = [call["call"] for call in (result.batch or {}).get("calls", [])]
    return Stage(result, _stage_files(result, changed), changed, calls)


def check_case(
    area: str, case: str, stages: Mapping[str, Stage], guard: WriteGuard
) -> None:
    """Compare every stage with ``golden/<area>/<case>/<stage>/``."""
    check_stages(f"{area}/{case}", stages, guard)


def records_of(path: Path) -> list[dict[str, Any]]:
    """Return the ``records`` of an ``_output.json`` file."""
    return json.loads(path.read_text(encoding="utf-8"))["records"]


def metadata_of(path: Path) -> dict[str, Any]:
    """Return the ``_chronominer_metadata`` of an ``_output.json`` file."""
    return json.loads(path.read_text(encoding="utf-8"))["_chronominer_metadata"]


def answers(path: Path) -> list[Any]:
    """Return the parsed ``output_text`` of every record of an output file."""
    return [
        json.loads(record["response"]["output_text"]) for record in records_of(path)
    ]


def store_rows(database: Path, sql: str) -> list[tuple[Any, ...]]:
    """Return the rows of a read-only query on a ``chronominer.sqlite``."""
    from chronominer.common.sqlite import read

    return read(database, lambda connection: connection.execute(sql).fetchall())


def ledger(root: Path, name: str = "ledger") -> Path:
    """Write the case's ledger to ``docs/<name>.txt`` and return it."""
    return make_text_file(root / "docs" / f"{name}.txt", RECORDS)


def scanned_pdf(root: Path) -> Path:
    """Write the case's two-page scanned PDF and return it."""
    return make_scanned_pdf(root / "docs" / "book.pdf", PAGES)


def submit(
    llm: FakeLLM,
    service: FakeBatchService,
    root: Path,
    provider: Provider,
    source: Path,
    **flags: Any,
) -> Stage:
    """Run ``extract --batch`` for *provider* on *source* into ``out/``."""
    return run_stage(
        "extract",
        service,
        llm=llm,
        root=root,
        input=source,
        output=root / "out",
        provider=provider,
        batch=True,
        **flags,
    )


def check(
    llm: FakeLLM, service: FakeBatchService, root: Path, provider: Provider
) -> Stage:
    """Run ``check_batches`` over the schema's output folder."""
    return run_stage("check_batches", service, llm=llm, root=root, provider=provider)


def round_trip(
    llm: FakeLLM,
    service: FakeBatchService,
    root: Path,
    provider: Provider,
    source: Path,
) -> dict[str, Stage]:
    """Submit, check while pending, finalize and check again."""
    stages = {"submit": submit(llm, service, root, provider, source)}
    stages["pending"] = check(llm, service, root, provider)
    stages["finalize"] = check(llm, service, root, provider)
    stages["recheck"] = check(llm, service, root, provider)
    return stages


@pytest.fixture
def two_polls(fake_batch: FakeBatchService) -> FakeBatchService:
    """Return the batch service with batches that complete on the second poll."""
    fake_batch.polls_until_complete = 2
    return fake_batch


def _summary(stage: Stage) -> dict[str, Any]:
    summary = stage.result.summary
    assert summary is not None, stage.result.stdout + stage.result.stderr
    return summary


def _counts(stage: Stage) -> tuple[int, int, int, int]:
    summary = _summary(stage)
    return (
        summary["finalized"],
        summary["pending"],
        summary["failed"],
        summary["unresolved"],
    )


def _assert_submitted(stage: Stage, provider: Provider, requests: int) -> None:
    result = stage.result
    assert result.exit_code == 0, result.stdout + result.stderr
    assert result.calls == []
    assert CREATE_CALLS[provider] in stage.batch_calls
    assert result.batch is not None
    assert len(result.batch["requests"]) == requests
    assert {request["provider"] for request in result.batch["requests"]} == {provider}
    assert not [p for p in result.created if p.name.endswith("_output.json")]
    assert result.database in result.created


def _submissions(stage: Stage) -> list[tuple[Any, ...]]:
    """Return provider, status and request count of every stored submission."""
    return store_rows(
        stage.result.database,
        "SELECT provider, status, request_count FROM submissions ORDER BY rowid",
    )


def _assert_round_trip(stages: Mapping[str, Stage], source: Path) -> None:
    pending, finalized, again = (
        stages["pending"],
        stages["finalize"],
        stages["recheck"],
    )
    output = finalized.result.output_json(source)
    assert pending.result.exit_code == 0, pending.result.console
    assert _counts(pending) == (0, 1, 0, 0)
    assert pending.result.created == []
    assert finalized.result.exit_code == 0, finalized.result.console
    assert _counts(finalized) == (1, 0, 0, 0)
    assert output in finalized.result.created
    assert output.with_suffix(".csv") in finalized.result.created
    assert metadata_of(output)["batch_tracking"]["fully_completed"] is True
    assert "partial" not in metadata_of(output)
    assert again.result.exit_code == 0
    assert _counts(again) == (0, 0, 0, 0)
    assert again.result.created == []
    assert again.changed == []
    assert again.batch_calls == []


@pytest.mark.parametrize("provider", PROVIDERS)
def test_text_round_trip(
    provider: Provider,
    fake_llm: FakeLLM,
    two_polls: FakeBatchService,
    case_root: Callable[[str], Path],
    write_guard: WriteGuard,
) -> None:
    def build(root: Path) -> dict[str, Stage]:
        two_polls.reset()
        return round_trip(fake_llm, two_polls, root, provider, ledger(root))

    stages = run_twice(case_root, build)

    submitted = stages["submit"]
    _assert_submitted(submitted, provider, len(CHUNK_UNITS))
    assert submitted.result.batch is not None
    units = [request["unit"] for request in submitted.result.batch["requests"]]
    assert units == list(CHUNK_UNITS)
    source = stages["finalize"].result.root / "docs" / "ledger.txt"
    _assert_round_trip(stages, source)
    assert _submissions(stages["finalize"]) == [
        (provider, "finalized", len(CHUNK_UNITS))
    ]
    schema = load_schema(DEFAULT_SCHEMA)
    output = stages["finalize"].result.output_json(source)
    assert answers(output) == [example_payload(schema, unit) for unit in CHUNK_UNITS]
    tracking = metadata_of(output)["batch_tracking"]["tracking"]
    assert [(t["provider"], t["schema_name"]) for t in tracking] == [
        (provider, DEFAULT_SCHEMA)
    ]
    check_case("batch", f"{provider}_text", stages, write_guard)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_pdf_round_trip(
    provider: Provider,
    fake_llm: FakeLLM,
    two_polls: FakeBatchService,
    case_root: Callable[[str], Path],
    write_guard: WriteGuard,
) -> None:
    def build(root: Path) -> dict[str, Stage]:
        two_polls.reset()
        return round_trip(fake_llm, two_polls, root, provider, scanned_pdf(root))

    stages = run_twice(case_root, build)

    submitted = stages["submit"]
    _assert_submitted(submitted, provider, PAGES)
    assert submitted.result.batch is not None
    units = [request["unit"] for request in submitted.result.batch["requests"]]
    assert units == [page_unit(index) for index in range(PAGES)]
    source = stages["finalize"].result.root / "docs" / "book.pdf"
    _assert_round_trip(stages, source)
    schema = load_schema(DEFAULT_SCHEMA)
    output = stages["finalize"].result.output_json(source)
    assert answers(output) == [
        example_payload(schema, page_unit(index)) for index in range(PAGES)
    ]
    assert metadata_of(output)["chunking_method"] == "pages"
    check_case("batch", f"{provider}_pdf", stages, write_guard)


def test_openai_answers_follow_the_schema_of_the_nested_body(
    fake_llm: FakeLLM,
    fake_batch: FakeBatchService,
    case_root: Callable[[str], Path],
) -> None:
    root = case_root("case")
    source = ledger(root)

    submitted = submit(fake_llm, fake_batch, root, "openai", source)
    lines = [
        json.loads(line)
        for line in fake_batch.files[next(iter(fake_batch.files))]
        .decode("utf-8")
        .splitlines()
    ]
    finalized = check(fake_llm, fake_batch, root, "openai")

    assert submitted.result.exit_code == 0, submitted.result.stdout
    assert finalized.result.exit_code == 0, finalized.result.stdout
    enforced = [line["body"]["text"]["format"] for line in lines]
    assert {fmt["type"] for fmt in enforced} == {"json_schema"}
    assert all("text" not in line for line in lines)
    schema = enforced[0]["schema"]
    output = finalized.result.output_json(source)
    assert answers(output) == [example_payload(schema, unit) for unit in CHUNK_UNITS]
    assert {"openai.files.content", "openai.files.delete"} <= set(finalized.batch_calls)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_finalization_stores_each_answer_as_text_and_entries(
    provider: Provider,
    fake_llm: FakeLLM,
    fake_batch: FakeBatchService,
    case_root: Callable[[str], Path],
) -> None:
    """Pin the answer text and the parsed entries a finalization stores.

    Each chunk unit keeps the answer as the batch returned it; the entries
    view holds the entries parsed from it, one row per entry.
    """
    root = case_root("case")
    source = ledger(root)

    submit(fake_llm, fake_batch, root, provider, source)
    finalized = check(fake_llm, fake_batch, root, provider)

    assert finalized.result.exit_code == 0, finalized.result.console
    schema = load_schema(DEFAULT_SCHEMA)
    payloads = [example_payload(schema, unit) for unit in CHUNK_UNITS]
    database = finalized.result.database
    contents = store_rows(
        database, "SELECT content FROM units WHERE kind = 'chunk' ORDER BY idx"
    )
    assert [json.loads(content) for (content,) in contents] == payloads
    entries = store_rows(database, "SELECT chunk_index, record_id, item FROM entries")
    assert entries == [
        (index, entry["record_id"], entry["item"])
        for index, payload in enumerate(payloads, 1)
        for entry in payload["entries"]
    ]


@pytest.mark.parametrize(
    ("provider", "error_form"),
    [
        ("openai", "error_file"),
        ("openai", "output"),
        ("anthropic", "errored"),
        ("google", "inline_error"),
    ],
)
def test_failed_request(
    provider: Provider,
    error_form: str,
    fake_llm: FakeLLM,
    fake_batch: FakeBatchService,
    case_root: Callable[[str], Path],
    write_guard: WriteGuard,
) -> None:
    fake_llm.script(CHUNK_UNITS[1], always=Reply(exception=permanent_error()))
    if error_form == "output":
        fake_batch.openai_error_form = "output"

    def build(root: Path) -> dict[str, Stage]:
        fake_batch.reset()
        source = ledger(root)
        stages = {"submit": submit(fake_llm, fake_batch, root, provider, source)}
        stages["finalize"] = check(fake_llm, fake_batch, root, provider)
        stages["recheck"] = check(fake_llm, fake_batch, root, provider)
        return stages

    stages = run_twice(case_root, build)

    finalized = stages["finalize"]
    source = finalized.result.root / "docs" / "ledger.txt"
    output = finalized.result.output_json(source)
    # A chunk failed in the batch, so the finalizing check exits 1.
    assert finalized.result.exit_code == 1, finalized.result.stdout
    assert _counts(finalized) == (1, 0, 0, 0)
    assert output in finalized.result.created
    assert [record["chunk_index"] for record in records_of(output)] == [1]
    metadata = metadata_of(output)
    assert metadata["partial"] is True
    assert metadata["failed_chunks"] == [2]
    assert metadata["batch_tracking"]["fully_completed"] is True
    check_case("batch", f"{provider}_failed_request_{error_form}", stages, write_guard)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_missing_result(
    provider: Provider,
    fake_llm: FakeLLM,
    fake_batch: FakeBatchService,
    case_root: Callable[[str], Path],
    write_guard: WriteGuard,
) -> None:
    fake_batch.drop_units = {CHUNK_UNITS[1]}

    def build(root: Path) -> dict[str, Stage]:
        fake_batch.reset()
        source = ledger(root)
        stages = {"submit": submit(fake_llm, fake_batch, root, provider, source)}
        stages["finalize"] = check(fake_llm, fake_batch, root, provider)
        return stages

    stages = run_twice(case_root, build)

    finalized = stages["finalize"]
    source = finalized.result.root / "docs" / "ledger.txt"
    output = finalized.result.output_json(source)
    assert finalized.result.exit_code == 1, finalized.result.stdout
    assert [record["chunk_index"] for record in records_of(output)] == [1]
    assert metadata_of(output)["partial"] is True
    check_case("batch", f"{provider}_missing_result", stages, write_guard)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_submission_failure(
    provider: Provider,
    fake_llm: FakeLLM,
    fake_batch: FakeBatchService,
    case_root: Callable[[str], Path],
    write_guard: WriteGuard,
) -> None:
    fake_batch.fail_on[CREATE_CALLS[provider]] = permanent_error()

    def build(root: Path) -> dict[str, Stage]:
        fake_batch.reset()
        source = ledger(root)
        stages = {"submit": submit(fake_llm, fake_batch, root, provider, source)}
        stages["check"] = check(fake_llm, fake_batch, root, provider)
        return stages

    stages = run_twice(case_root, build)

    submitted = stages["submit"]
    assert submitted.result.calls == []
    assert CREATE_CALLS[provider] in submitted.batch_calls
    assert fake_batch.batch_states() == []
    assert not [p for p in submitted.result.created if p.name.endswith("_output.json")]
    check_case("batch", f"{provider}_submission_failure", stages, write_guard)
