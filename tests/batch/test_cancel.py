"""Cancelling and listing the batches of the three providers."""

from __future__ import annotations

from pathlib import Path

import pytest

from chronominer.batch.cancel import (
    cancel_batches,
    cancel_submissions,
    cancel_unfinished,
    list_batches,
    provider_of,
)
from chronominer.batch.submit import submit_batch
from chronominer.common.batch import BatchStatus
from chronominer.common.sqlite import read
from chronominer.output.store import database_path, open_store
from chronominer.settings import Settings
from tests.batch.conftest import World, leased_document, page_images, run_spec


@pytest.mark.parametrize(
    ("batch_id", "provider"),
    [
        ("batch_abc", "openai"),
        ("msgbatch_abc", "anthropic"),
        ("batches/abc", "google"),
        ("anything-else", "openai"),
    ],
)
def test_the_provider_follows_the_id_shape(batch_id: str, provider: str) -> None:
    assert provider_of(batch_id) == provider


def seed(world: World) -> dict[str, str]:
    """One pending batch per provider and one completed OpenAI batch."""
    ids = {
        provider: world.add(provider, status=BatchStatus.IN_PROGRESS)
        for provider in ("openai", "anthropic", "google")
    }
    ids["done"] = world.add("openai", status=BatchStatus.COMPLETED)
    return ids


@pytest.mark.parametrize(
    ("batch_id", "provider"),
    [
        ("batch_1", "openai"),
        ("msgbatch_1", "anthropic"),
        ("batches/1", "google"),
    ],
)
async def test_cancel_by_id_uses_the_inferred_provider(
    world: World, batch_id: str, provider: str
) -> None:
    world.add(provider, status=BatchStatus.IN_PROGRESS, batch_id=batch_id)
    report = await cancel_batches([batch_id, "batch_unknown"], Settings())
    assert [(o.batch_id, o.provider, o.cancelled) for o in report.outcomes] == [
        (batch_id, provider, True),
        ("batch_unknown", "openai", False),
    ]
    assert (report.cancelled, report.failed) == (1, 1)
    assert world.batches[batch_id].cancelled


async def test_cancel_by_id_without_a_key_fails(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    report = await cancel_batches(["msgbatch_1"], Settings())
    (outcome,) = report.outcomes
    assert not outcome.cancelled
    assert "ANTHROPIC_API_KEY" in (outcome.error or "")
    assert world.calls == []


async def test_list_covers_every_provider_with_a_key(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = seed(world)
    monkeypatch.delenv("GOOGLE_API_KEY")
    listing = await list_batches(Settings())
    assert {(b.provider, b.batch_id) for b in listing.batches} == {
        ("openai", ids["openai"]),
        ("anthropic", ids["anthropic"]),
    }
    assert listing.skipped == ["google"]


async def test_cancel_all_cancels_every_unfinished_batch(world: World) -> None:
    ids = seed(world)
    report = await cancel_unfinished(Settings())
    assert report.cancelled == 3 and report.failed == 0
    assert {b.batch_id for b in world.batches.values() if b.cancelled} == {
        ids["openai"],
        ids["anthropic"],
        ids["google"],
    }
    assert not world.batches[ids["done"]].cancelled


async def _submit(root: Path, world: World, *, lose: bool = False) -> str:
    world.lose_create = lose
    async with open_store(root) as store:
        await store.begin_run("extract")
        handle = await leased_document(store)
        outcome = await submit_batch(
            page_images(2),
            document=handle,
            spec=run_spec(root),
            settings=Settings(),
        )
    world.lose_create = False
    (submission_id,) = outcome.submission_ids
    return submission_id


def _status(root: Path) -> str:
    return str(
        read(
            database_path(root),
            lambda connection: connection.execute(
                "SELECT status FROM submissions"
            ).fetchone()[0],
        )
    )


async def test_cancel_dir_cancels_the_submitted_batches_below_a_folder(
    world: World, tmp_path: Path
) -> None:
    world.status = BatchStatus.IN_PROGRESS
    first, second, lost = (tmp_path / "jobs" / name for name in ("a", "b/c", "d"))
    for root in (first, second, lost):
        root.mkdir(parents=True)
    await _submit(first, world)
    await _submit(second, world)
    unresolved = await _submit(lost, world, lose=True)

    report = await cancel_submissions(tmp_path / "jobs", Settings())

    assert (report.cancelled, report.failed) == (2, 0)
    assert {outcome.batch_id for outcome in report.outcomes} == {
        "openai-batch-1",
        "openai-batch-2",
    }
    assert all(world.batches[f"openai-batch-{n}"].cancelled for n in (1, 2))
    assert not world.batches["openai-batch-3"].cancelled
    assert [_status(root) for root in (first, second, lost)] == [
        "cancelled",
        "cancelled",
        "unresolved",
    ]
    (skipped,) = report.skipped
    assert unresolved in skipped
    async with open_store(first) as store:
        await store.begin_run("resume")
        assert (await leased_document(store)).todo() == [0, 1]

    again = await cancel_submissions(tmp_path / "jobs", Settings())
    assert again.outcomes == []


async def test_cancel_dir_leaves_an_ended_batch_for_batch_check(
    world: World, tmp_path: Path
) -> None:
    world.status = BatchStatus.COMPLETED
    root = tmp_path / "jobs"
    root.mkdir()
    submission_id = await _submit(root, world)

    report = await cancel_submissions(tmp_path, Settings())

    assert report.outcomes == []
    (skipped,) = report.skipped
    assert submission_id in skipped
    assert "has already ended" in skipped
    assert not world.batches["openai-batch-1"].cancelled
    assert _status(root) == "submitted"


async def test_a_failed_listing_counts_as_a_failure(world: World) -> None:
    seed(world)
    world.fail_list = {"anthropic"}
    report = await cancel_unfinished(Settings())
    assert report.listing is not None
    assert list(report.listing.errors) == ["anthropic"]
    assert (report.cancelled, report.failed) == (2, 1)
