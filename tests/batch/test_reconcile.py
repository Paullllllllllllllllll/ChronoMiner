"""Unresolved submissions: reconciliation per provider, adopt and abandon.

Run on Windows with:

    uv --directory <repo> run pytest tests/batch/test_reconcile.py -q

The shipped backends run over fake SDK clients (``sdk`` fixture) whose
create records the batch and then raises a read timeout, so the provider
holds a batch the store never learned about.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from chronominer.batch.check import (
    ABANDONED,
    UNRESOLVED,
    CheckTargetError,
    check_submissions,
)
from chronominer.batch.finalize import FAILED, FINALIZED_OUTCOME
from chronominer.batch.submit import submit_batch
from chronominer.common.sqlite import read
from chronominer.output.store import database_path, open_store
from chronominer.settings import Settings
from tests.batch.conftest import leased_document, page_images, run_spec

PAGES = 3
OUTPUT = "doc_output.json"


async def submit_lost(root: Path, provider: str, sdk: dict[str, Any]) -> str:
    """Submit with a create whose response is lost; return the submission id."""
    sdk[provider].lose_create = True
    async with open_store(root) as store:
        await store.begin_run("extract")
        handle = await leased_document(store, provider=provider)
        outcome = await submit_batch(
            page_images(PAGES),
            document=handle,
            spec=run_spec(root, provider),
            settings=Settings(),
        )
    sdk[provider].lose_create = False
    assert outcome.status == "unresolved"
    (submission_id,) = outcome.submission_ids
    return submission_id


def row(root: Path, submission_id: str) -> tuple[str, str | None]:
    def query(connection: sqlite3.Connection) -> tuple[str, str | None]:
        status, batch_id = connection.execute(
            "SELECT status, batch_id FROM submissions WHERE submission_id = ?",
            (submission_id,),
        ).fetchone()
        return str(status), batch_id

    return read(database_path(root), query)


def unit_statuses(root: Path) -> list[str]:
    return [
        str(status)
        for (status,) in read(
            database_path(root),
            lambda connection: connection.execute(
                "SELECT status FROM units ORDER BY idx"
            ).fetchall(),
        )
    ]


async def todo(root: Path, provider: str) -> list[int]:
    async with open_store(root) as store:
        await store.begin_run("resume")
        return (await leased_document(store, provider=provider)).todo()


@pytest.mark.parametrize("provider", ["openai", "anthropic", "google"])
async def test_an_unknown_create_is_never_resubmitted(
    sdk: dict[str, Any], tmp_path: Path, provider: str
) -> None:
    submission_id = await submit_lost(tmp_path, provider, sdk)
    assert row(tmp_path, submission_id) == ("unresolved", None)
    assert sdk[provider].creates == 1
    assert await todo(tmp_path, provider) == []

    await check_submissions(tmp_path, Settings())
    await check_submissions(tmp_path, Settings())

    assert sdk[provider].creates == 1
    assert unit_statuses(tmp_path) == ["pending"] * PAGES
    assert await todo(tmp_path, provider) == []


async def test_openai_adopts_its_batch_from_the_third_listing_page(
    sdk: dict[str, Any], tmp_path: Path
) -> None:
    openai = sdk["openai"]
    submission_id = await submit_lost(tmp_path, "openai", sdk)
    (lost,) = openai.records
    for _ in range(5):
        openai.add(submission_id="another-submission")
    openai.complete(lost.id)

    result = await check_submissions(tmp_path, Settings())

    (check,) = result.submissions
    assert check.outcome == FINALIZED_OUTCOME
    assert "adopted batch batch_1" in (check.detail or "")
    assert row(tmp_path, submission_id) == ("finalized", "batch_1")
    assert unit_statuses(tmp_path) == ["done"] * PAGES
    assert (tmp_path / OUTPUT).is_file()


async def test_openai_reports_several_batches_naming_the_submission(
    sdk: dict[str, Any], tmp_path: Path
) -> None:
    openai = sdk["openai"]
    submission_id = await submit_lost(tmp_path, "openai", sdk)
    openai.add(submission_id=submission_id, part="1")

    result = await check_submissions(tmp_path, Settings())

    (check,) = result.submissions
    assert check.outcome == UNRESOLVED
    assert "2 batches name submission" in (check.detail or "")
    assert row(tmp_path, submission_id) == ("unresolved", None)
    assert result.stats.unresolved == 1


async def test_google_adopts_its_batch_from_the_third_listing_page(
    sdk: dict[str, Any], tmp_path: Path
) -> None:
    genai = sdk["google"]
    for index in range(4):
        genai.add(f"batch-extraction-someone-else-{index}-p1")
    submission_id = await submit_lost(tmp_path, "google", sdk)
    lost = genai.records[-1]
    assert genai.records.index(lost) // genai.page_size == 2
    assert submission_id in lost.display_name
    genai.add("batch-extraction-later-p1")
    genai.complete(lost.name)

    result = await check_submissions(tmp_path, Settings())

    (check,) = result.submissions
    assert check.outcome == FINALIZED_OUTCOME
    assert "adopted batch" in (check.detail or "")
    assert row(tmp_path, submission_id) == ("finalized", lost.name)
    assert unit_statuses(tmp_path) == ["done"] * PAGES


async def test_google_reports_several_batches_naming_the_submission(
    sdk: dict[str, Any], tmp_path: Path
) -> None:
    genai = sdk["google"]
    submission_id = await submit_lost(tmp_path, "google", sdk)
    genai.add(genai.records[-1].display_name)

    result = await check_submissions(tmp_path, Settings())

    (check,) = result.submissions
    assert check.outcome == UNRESOLVED
    assert "2 batches name submission" in (check.detail or "")
    assert row(tmp_path, submission_id) == ("unresolved", None)


async def test_anthropic_lists_candidates_and_adopts_on_request(
    sdk: dict[str, Any], tmp_path: Path
) -> None:
    anthropic = sdk["anthropic"]
    submission_id = await submit_lost(tmp_path, "anthropic", sdk)
    (lost,) = anthropic.records
    other = anthropic.add(PAGES + 4)

    first = await check_submissions(tmp_path, Settings())

    (check,) = first.submissions
    assert check.outcome == UNRESOLVED
    assert [c.batch_id for c in check.candidates] == [lost.id, other.id]
    assert [c.request_count for c in check.candidates] == [PAGES, PAGES + 4]
    assert "--adopt BATCH_ID" in (check.detail or "")
    assert row(tmp_path, submission_id) == ("unresolved", None)

    anthropic.complete(lost.id)
    adopted = await check_submissions(tmp_path, Settings(), adopt=lost.id)

    (check,) = adopted.submissions
    assert check.outcome == FINALIZED_OUTCOME
    assert row(tmp_path, submission_id) == ("finalized", lost.id)
    assert unit_statuses(tmp_path) == ["done"] * PAGES
    assert anthropic.creates == 1


async def test_an_unknown_batch_is_not_adopted(
    sdk: dict[str, Any], tmp_path: Path
) -> None:
    submission_id = await submit_lost(tmp_path, "anthropic", sdk)

    result = await check_submissions(tmp_path, Settings(), adopt="msgbatch_404")

    (check,) = result.submissions
    assert check.outcome == FAILED
    assert "msgbatch_404 not adopted" in (check.detail or "")
    assert row(tmp_path, submission_id) == ("unresolved", None)


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
async def test_abandon_returns_the_units_to_pending(
    sdk: dict[str, Any], tmp_path: Path, provider: str
) -> None:
    submission_id = await submit_lost(tmp_path, provider, sdk)

    result = await check_submissions(tmp_path, Settings(), abandon=submission_id)

    (check,) = result.submissions
    assert check.outcome == ABANDONED
    assert result.stats.abandoned == 1
    assert row(tmp_path, submission_id) == ("abandoned", None)
    assert unit_statuses(tmp_path) == ["pending"] * PAGES
    assert await todo(tmp_path, provider) == list(range(PAGES))
    if provider == "openai":
        assert sdk["openai"].deleted == ["file-1"]


async def test_targets_that_fit_nothing_are_refused(
    sdk: dict[str, Any], tmp_path: Path
) -> None:
    submission_id = await submit_lost(tmp_path, "anthropic", sdk)

    with pytest.raises(CheckTargetError, match="No submission"):
        await check_submissions(tmp_path, Settings(), abandon="no-such-id")
    with pytest.raises(CheckTargetError, match="No unresolved openai submission"):
        await check_submissions(tmp_path, Settings(), adopt="batch_9")
    await check_submissions(tmp_path, Settings(), abandon=submission_id)
    with pytest.raises(CheckTargetError, match="only an unresolved or submitted"):
        await check_submissions(tmp_path, Settings(), abandon=submission_id)


async def test_abandon_gives_up_a_submitted_batch(
    sdk: dict[str, Any], tmp_path: Path
) -> None:
    async with open_store(tmp_path) as store:
        await store.begin_run("extract")
        handle = await leased_document(store, provider="openai")
        outcome = await submit_batch(
            page_images(PAGES),
            document=handle,
            spec=run_spec(tmp_path, "openai"),
            settings=Settings(),
        )
    (submission_id,) = outcome.submission_ids
    assert row(tmp_path, submission_id) == ("submitted", "batch_1")

    result = await check_submissions(
        tmp_path, Settings(), batch_ids=("batch_9",), abandon=submission_id
    )

    (check,) = result.submissions
    assert check.outcome == ABANDONED
    assert row(tmp_path, submission_id) == ("abandoned", "batch_1")
    assert unit_statuses(tmp_path) == ["pending"] * PAGES
    assert await todo(tmp_path, "openai") == list(range(PAGES))
    assert sdk["openai"].deleted == ["file-1"]
