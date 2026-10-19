"""Provider errors in batch calls: per-database isolation and the retry ladder."""

from __future__ import annotations

import functools
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import chronominer.batch.clients as clients_module
from chronominer.batch import cancel as cancel_module
from chronominer.batch import check as check_module
from chronominer.batch import submit as submit_module
from chronominer.batch.cancel import cancel_batches, list_batches
from chronominer.batch.check import CheckResult, check_submissions
from chronominer.batch.clients import BatchClients
from chronominer.batch.finalize import FAILED, FINALIZED_OUTCOME, PENDING
from chronominer.batch.submit import SubmitOutcome, submit_batch
from chronominer.batch.wait import wait_for_batches
from chronominer.common.batch import (
    BatchHandle,
    BatchStatus,
    create_backend,
)
from chronominer.output.store import DATABASE_NAME, open_store
from chronominer.settings import RetrySettings, Settings
from tests.batch.conftest import World, leased_document, page_images, run_spec
from tests.recording_events import RecordingEvents

OUTPUT = "doc_output.json"


class ApiError(Exception):
    """A provider error carrying an HTTP status."""

    def __init__(self, status_code: int, message: str = "provider error") -> None:
        super().__init__(f"Error code: {status_code} - {message}")
        self.status_code = status_code


class Sleeps:
    """Records the retry delays instead of waiting."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


def once() -> Settings:
    """Settings whose ladder gives up after the first attempt."""
    return Settings(retry=RetrySettings(max_attempts=1))


def no_waiting(monkeypatch: pytest.MonkeyPatch, module: Any) -> Sleeps:
    """Give the clients *module* builds a :class:`Sleeps`; return it."""
    sleeps = Sleeps()
    monkeypatch.setattr(
        module, "BatchClients", functools.partial(BatchClients, sleep=sleeps)
    )
    return sleeps


async def submit_item(root: Path, provider: str) -> SubmitOutcome:
    """Submit the two page chunks of document ``doc`` in *root*'s store."""
    root.mkdir(parents=True, exist_ok=True)
    async with open_store(root) as store:
        await store.begin_run("extract")
        handle = await leased_document(store, provider=provider)
        return await submit_batch(
            page_images(2),
            document=handle,
            spec=run_spec(root, provider),
            settings=Settings(),
        )


def by_root(result: CheckResult) -> dict[Path, Any]:
    """Map each database folder of *result* to its one submission check."""
    return {check.database.parent: check for check in result.submissions}


@pytest.mark.parametrize(
    ("status_code", "outcome"),
    [(503, PENDING), (429, PENDING), (401, FAILED), (404, FAILED)],
)
async def test_a_raising_status_in_one_database_does_not_stop_another(
    world: World, tmp_path: Path, status_code: int, outcome: str
) -> None:
    openai, anthropic = tmp_path / "out" / "a", tmp_path / "out" / "b"
    await submit_item(openai, "openai")
    await submit_item(anthropic, "anthropic")
    world.errors[("openai", "status")] = [ApiError(status_code)]

    result = await check_submissions(tmp_path / "out", once())

    checks = by_root(result)
    assert checks[anthropic].outcome == FINALIZED_OUTCOME
    assert (anthropic / OUTPUT).exists()
    assert checks[openai].outcome == outcome
    assert str(status_code) in (checks[openai].detail or "")
    assert not (openai / OUTPUT).exists()
    assert result.stats.finalized == 1
    assert result.stats.had_failure is (outcome == FAILED)


async def test_a_transient_status_error_is_retried_to_success(
    world: World,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await submit_item(tmp_path, "openai")
    world.errors[("openai", "status")] = [ApiError(503), ApiError(429)]
    sleeps = no_waiting(monkeypatch, check_module)

    with caplog.at_level(logging.WARNING):
        result = await check_submissions(tmp_path, Settings())

    (check,) = result.submissions
    assert check.outcome == FINALIZED_OUTCOME
    assert world.count("openai", "status") == 3
    assert len(sleeps.delays) == 2
    assert (tmp_path / OUTPUT).exists()
    assert (
        sum("Retrying status of batch" in r.getMessage() for r in caplog.records) == 2
    )


async def test_a_non_retryable_status_error_is_not_retried(
    world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await submit_item(tmp_path, "openai")
    world.errors[("openai", "status")] = [ApiError(401, "invalid api key")]
    sleeps = no_waiting(monkeypatch, check_module)

    result = await check_submissions(tmp_path, Settings())

    (check,) = result.submissions
    assert check.outcome == FAILED
    assert "invalid api key" in (check.detail or "")
    assert world.count("openai", "status") == 1
    assert sleeps.delays == []


class RaisingSDK:
    """The SDK client of every provider; each status lookup raises *error*."""

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.lookups = 0
        self.batches = self
        self.messages = SimpleNamespace(batches=self)

    def retrieve(self, batch_id: str) -> Any:
        return self._raise()

    def get(self, *, name: str) -> Any:
        return self._raise()

    def _raise(self) -> Any:
        self.lookups += 1
        raise self.error


def real_backends(monkeypatch: pytest.MonkeyPatch, error: Exception) -> RaisingSDK:
    """Build the shipped backends over a :class:`RaisingSDK`."""
    sdk = RaisingSDK(error)
    monkeypatch.setattr(clients_module, "create_client", lambda *_, **__: sdk)
    monkeypatch.setattr(clients_module, "create_backend", create_backend)
    return sdk


@pytest.mark.parametrize("provider", ["openai", "anthropic", "google"])
@pytest.mark.parametrize(
    ("status_code", "outcome", "lookups"), [(401, FAILED, 1), (503, PENDING, 2)]
)
async def test_a_status_error_of_a_shipped_backend_reaches_the_retry_ladder(
    world: World,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    status_code: int,
    outcome: str,
    lookups: int,
) -> None:
    await submit_item(tmp_path, provider)
    sdk = real_backends(monkeypatch, ApiError(status_code))
    no_waiting(monkeypatch, check_module)
    settings = Settings(retry=RetrySettings(max_attempts=2))

    result = await check_submissions(tmp_path, settings)

    (check,) = result.submissions
    assert check.outcome == outcome
    assert f"Error code: {status_code}" in (check.detail or "")
    assert sdk.lookups == lookups
    assert not (tmp_path / OUTPUT).exists()


@pytest.mark.parametrize("provider", ["openai", "anthropic", "google"])
async def test_a_rejected_key_stops_the_repair_wait(
    world: World, monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    real_backends(monkeypatch, ApiError(401, "invalid api key"))
    clients = BatchClients(Settings(), "extract", sleep=Sleeps())

    outcome = await wait_for_batches(
        [BatchHandle(provider, "b1")],
        clients,
        RecordingEvents(),
        poll_seconds=0,
        timeout_seconds=0,
    )

    assert outcome.running == []
    assert "invalid api key" in (outcome.error or "")


async def test_a_download_error_outlasting_the_retries_leaves_the_batch_pending(
    world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await submit_item(tmp_path, "openai")
    world.errors[("openai", "results")] = [ApiError(502), ApiError(502)]
    no_waiting(monkeypatch, check_module)
    settings = Settings(retry=RetrySettings(max_attempts=2))

    result = await check_submissions(tmp_path, settings)

    (check,) = result.submissions
    assert check.outcome == PENDING
    assert "not downloaded" in (check.detail or "")
    assert world.count("openai", "results") == 2
    assert not (tmp_path / OUTPUT).exists()


async def test_a_submission_is_attempted_once(
    world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world.errors[("openai", "submit")] = [ApiError(503)]
    sleeps = no_waiting(monkeypatch, submit_module)

    outcome = await submit_item(tmp_path, "openai")

    assert outcome.status == "unresolved"
    assert "503" in (outcome.detail or "")
    assert world.count("openai", "submit") == 1
    assert sleeps.delays == []
    assert world.batches == {}
    assert (tmp_path / DATABASE_NAME).is_file()


async def test_a_cancel_error_is_reported_for_its_batch(world: World) -> None:
    world.add("openai", status=BatchStatus.IN_PROGRESS, batch_id="batch_a")
    world.add("anthropic", status=BatchStatus.IN_PROGRESS, batch_id="msgbatch_b")
    world.errors[("openai", "cancel")] = [ApiError(403, "not allowed")]

    report = await cancel_batches(["batch_a", "msgbatch_b"], Settings())

    assert [(o.batch_id, o.cancelled) for o in report.outcomes] == [
        ("batch_a", False),
        ("msgbatch_b", True),
    ]
    assert "not allowed" in (report.outcomes[0].error or "")
    assert (report.cancelled, report.failed) == (1, 1)


async def test_a_transient_cancel_error_is_retried(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    world.add("anthropic", status=BatchStatus.IN_PROGRESS, batch_id="msgbatch_b")
    world.errors[("anthropic", "cancel")] = [ApiError(529, "overloaded")]
    sleeps = no_waiting(monkeypatch, cancel_module)

    report = await cancel_batches(["msgbatch_b"], Settings())

    assert report.cancelled == 1
    assert world.count("anthropic", "cancel") == 2
    assert len(sleeps.delays) == 1


async def test_a_list_error_is_reported_for_its_provider(world: World) -> None:
    world.add("openai", status=BatchStatus.IN_PROGRESS, batch_id="batch_a")
    world.errors[("anthropic", "list")] = [ApiError(500, "listing broke")]

    listing = await list_batches(once())

    assert [b.batch_id for b in listing.batches] == ["batch_a"]
    assert list(listing.errors) == ["anthropic"]
    assert "listing broke" in listing.errors["anthropic"]
