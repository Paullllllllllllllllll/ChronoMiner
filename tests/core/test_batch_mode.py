"""Batch mode through the core: submission, dry runs and submission outcomes.

``chronominer.batch.submit.submit_batch`` is replaced by a stub with the
signature of the real function; each outcome (submitted, refused,
unresolved) and an unbuildable submission drive the extractor's reaction.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import chronominer.batch.submit as submit_module
import chronominer.extract.llm as llm_module
import chronominer.run as core
from chronominer import __main__ as entry
from chronominer import ext
from chronominer.common.contract import ExitCode
from chronominer.common.sqlite import read_rows
from chronominer.extract.units import DocumentHandle
from chronominer.output.store import DATABASE_NAME
from chronominer.spec import to_argv
from tests.characterization.inputs import make_scanned_pdf, make_text_file
from tests.conftest import make_settings, make_spec


@pytest.fixture(autouse=True)
def openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set the key variable of the default provider."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


@dataclass(frozen=True)
class Outcome:
    """A submission outcome as ``submit_batch`` reports it."""

    status: str
    submission_ids: tuple[str, ...] = ()
    detail: str | None = None


class FakeSubmit:
    """Stand in for ``submit_batch``; record each document and its payloads.

    *outcome* is returned, or raised when it is an exception.
    """

    def __init__(self, outcome: Outcome | Exception | None = None) -> None:
        self.outcome = outcome or Outcome("submitted", ("sub-1",))
        self.items: list[tuple[str, int]] = []
        self.payloads: list[Any] = []
        self.schema_paths: list[Path | None] = []

    async def __call__(
        self,
        payloads: Sequence[Any],
        *,
        document: DocumentHandle,
        spec: Any,
        settings: Any,
        schema_path: Path | None = None,
        context: str | None = None,
        context_image: Path | None = None,
    ) -> Outcome:
        self.items.append((Path(document.name).name, len(payloads)))
        self.payloads.extend(payloads)
        self.schema_paths.append(schema_path)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome

    def install(self, monkeypatch: pytest.MonkeyPatch) -> FakeSubmit:
        monkeypatch.setattr(submit_module, "submit_batch", self, raising=False)
        return self


def _no_client(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("batch mode builds no extraction client")


def _statuses(out: Path) -> list[str]:
    rows = read_rows(out / DATABASE_NAME, "units", order_by=["doc_id", "idx"])
    return [row["status"] for row in rows]


def _scan_spec(tmp_path: Path, out: Path, **flags: Any) -> Any:
    return make_spec(
        make_scanned_pdf(tmp_path / "in" / "scan.pdf", pages=2),
        type="pdf",
        batch=True,
        output=out,
        **flags,
    )


def test_a_batch_run_submits_the_pages_and_leaves_them_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    submitted = FakeSubmit().install(monkeypatch)
    monkeypatch.setattr(llm_module, "ChunkExtractor", _no_client)
    out = tmp_path / "out"
    run_plan = core.plan(_scan_spec(tmp_path, out), make_settings())

    result = asyncio.run(core.run(run_plan))

    assert run_plan.batch
    assert submitted.items == [("scan", 2)]
    assert [payload.custom_id for payload in submitted.payloads] == [
        "scan-page-1",
        "scan-page-2",
    ]
    assert all(path is not None and path.is_file() for path in submitted.schema_paths)
    assert [(item.name, item.success) for item in result.items] == [("scan.pdf", True)]
    assert _statuses(out) == ["pending", "pending"]
    assert not list(out.rglob("*_output.json"))


def test_a_text_item_submits_its_chunks_as_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    submitted = FakeSubmit().install(monkeypatch)
    out = tmp_path / "out"
    text = make_text_file(tmp_path / "in" / "ledger.txt", records=6)
    spec = make_spec(text, type="text", batch=True, output=out, chunk_tokens=200)

    asyncio.run(core.run(core.plan(spec, make_settings())))

    ((name, count),) = submitted.items
    assert name == "ledger" and count >= 1
    assert all(isinstance(payload.text, str) for payload in submitted.payloads)
    assert set(_statuses(out)) == {"pending"}


def test_a_refused_batch_fails_the_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ext, "batch_allowed", lambda spec, **_: False)

    with pytest.raises(core.ConfigurationError, match="Batch submission") as raised:
        core.plan(_scan_spec(tmp_path, tmp_path / "out"), make_settings())

    assert raised.value.exit_code == ExitCode.USAGE
    assert raised.value.items_total == 1


@pytest.mark.parametrize("batch", [True, False])
def test_a_dry_run_reports_batch_mode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], batch: bool
) -> None:
    out = tmp_path / "out"
    spec = make_spec(
        make_scanned_pdf(tmp_path / "in" / "scan.pdf", pages=1),
        type="pdf",
        batch=batch,
        output=out,
    )

    assert entry.main(["extract", *to_argv(spec), "--dry-run"]) == 0

    err = capsys.readouterr().err
    note = "Batch mode: 1 item(s) would be submitted to the openai batch API."
    assert (note in err) is batch
    assert not out.exists()


@pytest.mark.parametrize(
    "outcome",
    [
        Outcome("refused", detail="400 invalid request"),
        submit_module.BatchSubmissionError("400 invalid request"),
    ],
    ids=["refused", "unbuildable"],
)
def test_a_refused_submission_fails_the_item(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: Outcome | Exception
) -> None:
    FakeSubmit(outcome).install(monkeypatch)
    monkeypatch.setattr(llm_module, "ChunkExtractor", _no_client)
    out = tmp_path / "out"

    result = asyncio.run(
        core.run(core.plan(_scan_spec(tmp_path, out), make_settings()))
    )

    ((success, error),) = [(item.success, item.error) for item in result.items]
    assert not success
    assert error is not None and "400 invalid request" in error
    assert _statuses(out) == ["pending", "pending"]
    assert not list(out.rglob("*_output.json"))


def test_an_unresolved_submission_fails_the_item(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeSubmit(Outcome("unresolved", ("sub-1",), "timeout")).install(monkeypatch)
    out = tmp_path / "out"

    result = asyncio.run(
        core.run(core.plan(_scan_spec(tmp_path, out), make_settings()))
    )

    ((success, error),) = [(item.success, item.error) for item in result.items]
    assert not success
    assert error is not None
    assert "unresolved" in error and "chronominer batch check" in error
    assert _statuses(out) == ["pending", "pending"]
