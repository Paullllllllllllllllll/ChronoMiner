"""Characterization of repairing failed chunks, synchronously and in batches.

The ledger has two chunks; the fake LLM fails the second (unit
``record-3``) with a permanent error, then the script is healed so every
request sent afterwards gets its answer. ``repair`` (the ``repair`` command) finds
the failed chunks in the store of the output folder, runs them again
(synchronously, or through the batch API with ``batch=True``), stores the
answers and renders ``_output.json`` and the CSV again. ``extract --batch``
on a partial document submits only its failed chunk, and ``check_batches``
merges the answer. ``repair --files`` limits the run to the named document.

Every stage is snapshotted like the batch cases (see ``test_batch``) and
compared with ``golden/repair/<case>/<stage>/``; each case runs twice in
fresh roots and both snapshots must match.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from chronominer.common.testing import WriteGuard
from tests.characterization.fakes import FakeLLM, Reply, permanent_error
from tests.characterization.fakes_batch import FakeBatchService, Provider
from tests.characterization.stages import run_twice
from tests.characterization.test_batch import (
    CHUNK_UNITS,
    CREATE_CALLS,
    PROVIDERS,
    Stage,
    check,
    check_case,
    ledger,
    metadata_of,
    records_of,
    run_stage,
    submit,
)

pytestmark = pytest.mark.integration

FAILED_UNIT = CHUNK_UNITS[1]


def _fail(llm: FakeLLM) -> None:
    llm.script(FAILED_UNIT, always=Reply(exception=permanent_error()))


def _heal(llm: FakeLLM) -> None:
    llm.script(FAILED_UNIT, always=None)


def extract(
    llm: FakeLLM,
    service: FakeBatchService,
    root: Path,
    provider: Provider,
    source: Path | None = None,
    **flags: Any,
) -> Stage:
    """Run ``extract`` synchronously on the case's ledger into ``out/``."""
    return run_stage(
        "extract",
        service,
        llm=llm,
        root=root,
        input=source or root / "docs" / "ledger.txt",
        output=root / "out",
        provider=provider,
        **flags,
    )


def repair(
    llm: FakeLLM,
    service: FakeBatchService,
    root: Path,
    provider: Provider = "openai",
    **flags: Any,
) -> Stage:
    """Run ``repair`` over the output folder with *provider*'s model."""
    return run_stage("repair", service, llm=llm, root=root, provider=provider, **flags)


def _output(stage: Stage, name: str = "ledger") -> Path:
    return stage.result.output_dir / f"{name}_output.json"


def _indices(path: Path) -> list[int]:
    return [record["chunk_index"] for record in records_of(path)]


def _units(stage: Stage) -> list[str | None]:
    return [call["unit"] for call in stage.result.calls]


def _repaired(stage: Stage) -> tuple[int, int]:
    summary = stage.result.summary
    assert summary is not None, stage.result.console
    return summary["repaired"], summary["failed"]


@pytest.mark.parametrize("provider", PROVIDERS)
def test_sync_failed_chunk(
    provider: Provider,
    fake_llm: FakeLLM,
    fake_batch: FakeBatchService,
    case_root: Callable[[str], Path],
    write_guard: WriteGuard,
) -> None:
    def build(root: Path) -> dict[str, Stage]:
        fake_batch.reset()
        ledger(root)
        _fail(fake_llm)
        stages = {"first": extract(fake_llm, fake_batch, root, provider)}
        _heal(fake_llm)
        stages["repair"] = repair(fake_llm, fake_batch, root, provider)
        stages["resume"] = extract(fake_llm, fake_batch, root, provider)
        return stages

    stages = run_twice(case_root, build)

    first = stages["first"]
    output = _output(first)
    assert first.result.exit_code == 1, first.result.console
    assert _units(first) == list(CHUNK_UNITS)
    repaired = stages["repair"]
    assert repaired.result.exit_code == 0, repaired.result.console
    assert _repaired(repaired) == (1, 0)
    assert _units(repaired) == [FAILED_UNIT]
    assert output in repaired.changed
    assert _indices(output) == [1, 2]
    assert "partial" not in metadata_of(output)
    resumed = stages["resume"]
    assert resumed.result.exit_code == 0, resumed.result.console
    assert resumed.result.calls == []
    assert resumed.changed == []
    check_case("repair", f"sync_{provider}", stages, write_guard)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_batch_failed_chunk(
    provider: Provider,
    fake_llm: FakeLLM,
    fake_batch: FakeBatchService,
    case_root: Callable[[str], Path],
    write_guard: WriteGuard,
) -> None:
    def build(root: Path) -> dict[str, Stage]:
        fake_batch.reset()
        source = ledger(root)
        _fail(fake_llm)
        stages = {"submit": submit(fake_llm, fake_batch, root, provider, source)}
        stages["finalize"] = check(fake_llm, fake_batch, root, provider)
        _heal(fake_llm)
        stages["resubmit"] = submit(fake_llm, fake_batch, root, provider, source)
        stages["refinalize"] = check(fake_llm, fake_batch, root, provider)
        return stages

    stages = run_twice(case_root, build)

    output = _output(stages["finalize"])
    resubmitted = stages["resubmit"]
    assert resubmitted.result.exit_code == 0, resubmitted.result.console
    assert resubmitted.result.batch is not None
    assert [r["unit"] for r in resubmitted.result.batch["requests"]] == [FAILED_UNIT]
    refinalized = stages["refinalize"]
    assert refinalized.result.exit_code == 0, refinalized.result.console
    assert _indices(output) == [1, 2]
    assert "partial" not in metadata_of(output)
    check_case("repair", f"batch_{provider}", stages, write_guard)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_batch_repair(
    provider: Provider,
    fake_llm: FakeLLM,
    fake_batch: FakeBatchService,
    case_root: Callable[[str], Path],
    write_guard: WriteGuard,
) -> None:
    def build(root: Path) -> dict[str, Stage]:
        fake_batch.reset()
        source = ledger(root)
        _fail(fake_llm)
        stages = {"submit": submit(fake_llm, fake_batch, root, provider, source)}
        stages["finalize"] = check(fake_llm, fake_batch, root, provider)
        _heal(fake_llm)
        stages["repair"] = repair(fake_llm, fake_batch, root, provider, batch=True)
        return stages

    stages = run_twice(case_root, build)

    finalized = stages["finalize"]
    output = _output(finalized)
    snapshot = json.loads(finalized.files["files/out/ledger_output.json"])
    assert snapshot["_chronominer_metadata"]["failed_chunks"] == [2]
    repaired = stages["repair"]
    assert repaired.result.exit_code == 0, repaired.result.console
    assert _repaired(repaired) == (1, 0)
    assert repaired.result.calls == []
    assert CREATE_CALLS[provider] in repaired.batch_calls
    assert repaired.result.batch is not None
    assert [r["unit"] for r in repaired.result.batch["requests"]] == [FAILED_UNIT]
    assert output in repaired.changed
    assert _indices(output) == [1, 2]
    check_case("repair", f"batch_repair_{provider}", stages, write_guard)


def test_files_selection(
    fake_llm: FakeLLM,
    fake_batch: FakeBatchService,
    case_root: Callable[[str], Path],
    write_guard: WriteGuard,
) -> None:
    def build(root: Path) -> dict[str, Stage]:
        fake_batch.reset()
        ledger(root)
        ledger(root, "annals")
        _fail(fake_llm)
        stages = {"first": extract(fake_llm, fake_batch, root, "openai", root / "docs")}
        _heal(fake_llm)
        stages["repair"] = repair(
            fake_llm, fake_batch, root, files=["annals_temp.jsonl"]
        )
        stages["unknown"] = repair(
            fake_llm, fake_batch, root, files=["missing_temp.jsonl"]
        )
        return stages

    stages = run_twice(case_root, build)

    repaired = stages["repair"]
    assert repaired.result.exit_code == 0, repaired.result.console
    assert _repaired(repaired) == (1, 0)
    assert _units(repaired) == [FAILED_UNIT]
    changed = {path.name for path in repaired.changed}
    assert {"annals_output.json", "annals_output.csv"} <= changed
    assert not changed & {"ledger_output.json", "ledger_output.csv"}
    assert "partial" in metadata_of(_output(repaired))
    unknown = stages["unknown"]
    assert unknown.result.exit_code == 2
    assert "missing_output.json" in unknown.result.stderr
    assert unknown.result.created == []
    check_case("repair", "files_selection", stages, write_guard)
