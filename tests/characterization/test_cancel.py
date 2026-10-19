"""Characterization of batch cancellation through the fake batch services.

Each case submits ledgers with ``extract --batch`` and runs
``cancel_batches`` (``batch cancel --input <output folder>``), which reads
the open submissions of the store in the output folder and cancels every
batch that has not ended. The snapshot of each stage (see ``test_batch``) adds
``batches.json``, the state of every submitted batch after the stage. Each
case runs twice in fresh roots and is compared with
``golden/cancel/<case>/<stage>/``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from chronominer.common.testing import WriteGuard
from tests.characterization.fakes import FakeLLM
from tests.characterization.fakes_batch import FakeBatchService, Provider
from tests.characterization.golden import normalizer_for
from tests.characterization.stages import run_twice
from tests.characterization.test_batch import (
    PROVIDERS,
    Stage,
    check_case,
    ledger,
    run_stage,
    submit,
)

pytestmark = pytest.mark.integration

NEVER_COMPLETES = 1_000
"""Status calls before a fake batch completes; more than any case makes."""

CANCEL_CALLS: dict[Provider, str] = {
    "openai": "openai.batches.cancel",
    "anthropic": "anthropic.messages.batches.cancel",
    "google": "google.batches.cancel",
}


@pytest.fixture
def pending_batches(fake_batch: FakeBatchService) -> FakeBatchService:
    """Return the batch service with batches that stay pending."""
    fake_batch.polls_until_complete = NEVER_COMPLETES
    return fake_batch


def cancel(llm: FakeLLM, service: FakeBatchService, root: Path) -> Stage:
    """Run ``cancel_batches``; add the batch states to the snapshot."""
    stage = run_stage("cancel_batches", service, llm=llm, root=root)
    states = normalizer_for(stage.result).data(service.batch_states())
    stage.files["batches.json"] = (
        json.dumps(states, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    return stage


def _states(service: FakeBatchService) -> list[tuple[str, str]]:
    return [(state["provider"], state["state"]) for state in service.batch_states()]


def _submitted(stage: Stage) -> None:
    assert stage.result.exit_code == 0, stage.result.console


def _cancelled(stage: Stage) -> int:
    summary = stage.result.summary
    assert summary is not None, stage.result.console
    return int(summary["cancelled"])


def check_cancel(case: str, stages: dict[str, Stage], guard: WriteGuard) -> None:
    """Compare the cancel stages with their goldens; ``test_batch`` pins submits."""
    kept = {name: stage for name, stage in stages.items() if "submit" not in name}
    check_case("cancel", case, kept, guard)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_cancel_pending(
    provider: Provider,
    fake_llm: FakeLLM,
    pending_batches: FakeBatchService,
    case_root: Callable[[str], Path],
    write_guard: WriteGuard,
) -> None:
    def build(root: Path) -> dict[str, Stage]:
        pending_batches.reset()
        source = ledger(root)
        stages: dict[str, Any] = {
            "submit": submit(fake_llm, pending_batches, root, provider, source)
        }
        _submitted(stages["submit"])
        stages["cancel"] = cancel(fake_llm, pending_batches, root)
        return stages

    stages = run_twice(case_root, build)

    cancelled = stages["cancel"]
    assert cancelled.result.exit_code == 0, cancelled.result.console
    assert cancelled.batch_calls[-1] == CANCEL_CALLS[provider]
    assert _cancelled(cancelled) == 1
    assert _states(pending_batches) == [(provider, "cancelled")]
    check_cancel(f"pending_{provider}", stages, write_guard)


def test_completed_batch_is_not_cancelled(
    fake_llm: FakeLLM,
    fake_batch: FakeBatchService,
    case_root: Callable[[str], Path],
    write_guard: WriteGuard,
) -> None:
    def build(root: Path) -> dict[str, Stage]:
        fake_batch.reset()
        source = ledger(root)
        stages = {"submit": submit(fake_llm, fake_batch, root, "openai", source)}
        _submitted(stages["submit"])
        stages["cancel"] = cancel(fake_llm, fake_batch, root)
        return stages

    stages = run_twice(case_root, build)

    cancelled = stages["cancel"]
    assert cancelled.result.exit_code == 0, cancelled.result.console
    assert _cancelled(cancelled) == 0
    assert "openai.batches.cancel" not in cancelled.batch_calls
    assert _states(fake_batch) == [("openai", "completed")]
    check_cancel("completed_openai", stages, write_guard)


def test_cancel_every_pending_batch(
    fake_llm: FakeLLM,
    pending_batches: FakeBatchService,
    case_root: Callable[[str], Path],
    write_guard: WriteGuard,
) -> None:
    def build(root: Path) -> dict[str, Stage]:
        pending_batches.reset()
        stages: dict[str, Stage] = {}
        for provider in PROVIDERS:
            source = ledger(root, f"{provider}_ledger")
            stage = submit(fake_llm, pending_batches, root, provider, source)
            _submitted(stage)
            stages[f"submit_{provider}"] = stage
        done = submit(
            fake_llm, pending_batches, root, "openai", ledger(root, "done_ledger")
        )
        _submitted(done)
        stages["submit_done"] = done
        finished = pending_batches.batches[
            pending_batches.batch_states()[-1]["batch_id"]
        ]
        pending_batches.polls_until_complete = 1
        pending_batches.poll(finished)
        pending_batches.polls_until_complete = NEVER_COMPLETES
        stages["cancel"] = cancel(fake_llm, pending_batches, root)
        return stages

    stages = run_twice(case_root, build)

    cancelled = stages["cancel"]
    assert cancelled.result.exit_code == 0, cancelled.result.console
    assert _cancelled(cancelled) == 3
    assert sorted(
        call for call in cancelled.batch_calls if call in CANCEL_CALLS.values()
    ) == sorted(CANCEL_CALLS.values())
    assert _states(pending_batches) == [
        ("openai", "cancelled"),
        ("anthropic", "cancelled"),
        ("google", "cancelled"),
        ("openai", "completed"),
    ]
    check_cancel("every_pending", stages, write_guard)
