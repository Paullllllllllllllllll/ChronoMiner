"""Scripted extract sessions through the wizard and the task's run."""

from __future__ import annotations

import argparse
import asyncio
import io
import sqlite3
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from chronominer.common.testing import KEEP, ScriptedPrompter, WriteGuard
from chronominer.common.wizard import Decision, Outcome, State
from chronominer.spec import OPTIONS, RunSpec, parse
from chronominer.ui.extract import flow
from chronominer.ui.review import PROGRAM
from chronominer.ui.session import RunRequest, Session, build_wizard
from tests.characterization.conftest import CONTEXT_DIR
from tests.characterization.fakes import PROVIDER_CLASSES, FakeLLM
from tests.characterization.fakes_batch import FakeBatchService
from tests.characterization.seams import replace
from tests.ui.conftest import DUMMY_KEYS

SCHEMA = "WageRecords"

_ASYNC_SLEEP = asyncio.sleep
HEARTBEAT_TASK = "lease-heartbeat"


async def _yield_only(delay: float, result: Any = None) -> Any:
    """Only yield, except in the lease heartbeat, which keeps its interval."""
    task = asyncio.current_task()
    if task is not None and task.get_name() == HEARTBEAT_TASK:
        return await _ASYNC_SLEEP(delay, result)
    await _ASYNC_SLEEP(0)
    return result


@pytest.fixture
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> FakeLLM:
    """Install the characterization FakeLLM, the dummy keys and instant sleeps."""
    import chronominer.llm.prompts as prompts_module

    controller = FakeLLM()
    for module_name, name in PROVIDER_CLASSES:
        replace(monkeypatch, module_name, name, controller.constructor(name))
    for name, value in DUMMY_KEYS.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(prompts_module, "CONTEXT_DIR", CONTEXT_DIR)
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(asyncio, "sleep", _yield_only)
    return controller


@pytest.fixture
def fake_batch(
    fake_llm: FakeLLM, monkeypatch: pytest.MonkeyPatch
) -> Iterator[FakeBatchService]:
    """Install the batch fakes of OpenAI, Anthropic and Google."""
    service = FakeBatchService(fake_llm)
    service.install(monkeypatch)
    yield service
    service.reset()


def _model_and_response() -> list[Any]:
    return [
        *[KEEP] * 4,  # provider, model, reasoning, model parameters
        KEEP,  # response format
        SCHEMA,
        *[KEEP] * 3,  # context, fallback context, context image
    ]


def _run_answers(flags: Sequence[str], end: str, *, batch: bool = False) -> list[Any]:
    concurrency = [] if batch else [KEEP]  # not asked for a batch run
    return [batch, list(flags), *concurrency, end]  # submission, run options


def _text_answers(
    text: Path, out: Path, chunking: Sequence[Any], end: str = "run"
) -> list[Any]:
    """Answers of a text extraction into *out*; *chunking* answers its step."""
    return [
        KEEP,  # task
        str(text),
        "text",
        *chunking,
        KEEP,  # pages: all
        "folder",
        str(out),
        ["csv"],
        *_model_and_response(),
        *_run_answers([], end),
    ]


def _pdf_answers(pdf: Path, out: Path, flags: Sequence[str], end: str) -> list[Any]:
    """Answers of a PDF extraction into *out*."""
    return [
        KEEP,  # task
        str(pdf),
        KEEP,  # input type: auto
        KEEP,  # pages: all
        "folder",
        str(out),
        [],  # formats
        *_model_and_response(),
        KEEP,  # page images
        *_run_answers(flags, end),
    ]


def _image_answers(folder: Path, out: Path, *, batch: bool) -> list[Any]:
    """Answers of an image-folder extraction into *out*."""
    return [
        KEEP,  # task
        str(folder),
        KEEP,  # subfolders: no
        "images",
        "all",  # which items
        "folder",
        str(out),
        "mirror",
        ["csv"],
        *_model_and_response(),
        KEEP,  # page images
        *_run_answers([], "run", batch=batch),
    ]


def _session_run(
    session: Session, answers: Sequence[Any]
) -> tuple[Outcome, ScriptedPrompter]:
    prompter = ScriptedPrompter(answers)
    with prompter:
        outcome = build_wizard(flow(), session, prompter).run()
    return outcome, prompter


def _execute(session: Session, outcome: Outcome) -> tuple[int, str]:
    screen = io.StringIO()
    console = Console(file=screen, width=120)
    request = RunRequest(session, outcome.answers, console, argparse.Namespace())
    return flow().run(request), screen.getvalue()


def _printed_spec(session: Session, outcome: Outcome) -> RunSpec:
    """Assert the printed command parses to the wizard's spec; return the spec."""
    assert outcome.command is not None
    words = list(outcome.command)
    assert tuple(words[:2]) == PROGRAM
    state = State(OPTIONS, session.settings.defaults, outcome.answers)
    spec: RunSpec = flow().resolve(state, session).spec
    assert parse(words[2:], session.settings.defaults) == spec
    return spec


def _files(*folders: Path) -> dict[Path, bytes]:
    return {
        path: path.read_bytes()
        for folder in folders
        if folder.exists()
        for path in folder.rglob("*")
        if path.is_file()
    }


def test_a_session_extracts_a_text_file_with_automatic_chunks(
    session: Session, text_file: Path, tmp_path: Path, fake_llm: FakeLLM
) -> None:
    out = tmp_path / "out"
    answers = _text_answers(text_file, out, [KEEP, KEEP])  # mode, tokens per chunk
    outcome, _prompter = _session_run(session, answers)
    assert outcome.action is Decision.RUN
    spec = _printed_spec(session, outcome)
    assert spec.chunking.mode == "auto"
    assert spec.output.formats == ("csv",)
    code, screen = _execute(session, outcome)
    assert code == 0, screen
    assert fake_llm.calls
    assert (out / "chronominer.sqlite").is_file()
    assert (out / "ledger_output.json").is_file()
    assert (out / "ledger_output.csv").is_file()
    assert "1 complete, 0 failed" in screen

    again = _execute(session, outcome)
    assert again[0] == 0
    assert "Nothing to do: 1 item already complete." in again[1]


def test_a_plan_with_only_a_refused_item_exits_1_naming_it_failed(
    session: Session, text_file: Path, tmp_path: Path, fake_llm: FakeLLM
) -> None:
    out = tmp_path / "out"
    outcome, _prompter = _session_run(
        session, _text_answers(text_file, out, [KEEP, KEEP])
    )
    assert _execute(session, outcome)[0] == 0
    text_file.write_text(text_file.read_text(encoding="utf-8") + "\nEdited.\n", "utf-8")
    calls = len(fake_llm.calls)
    code, screen = _execute(session, outcome)
    assert code == 1, screen
    assert len(fake_llm.calls) == calls
    assert "Nothing to do" not in screen
    assert "changed since" in screen
    assert "0 complete, 1 failed" in screen


def test_a_plan_with_only_a_blocked_item_exits_1(
    session: Session, text_file: Path, tmp_path: Path, fake_llm: FakeLLM
) -> None:
    out = tmp_path / "out"
    outcome, _prompter = _session_run(
        session, _text_answers(text_file, out, [KEEP, KEEP])
    )
    assert _execute(session, outcome)[0] == 0
    connection = sqlite3.connect(out / "chronominer.sqlite")
    try:
        connection.execute(
            "UPDATE documents SET lease_run_id = 'other-run', lease_until = ?",
            (time.time() + 600,),
        )
        connection.commit()
    finally:
        connection.close()

    code, screen = _execute(session, outcome)

    assert code == 1, screen


def test_a_session_extracts_a_text_file_by_its_line_ranges(
    session: Session, text_with_ranges: Path, tmp_path: Path, fake_llm: FakeLLM
) -> None:
    out = tmp_path / "out"
    answers = _text_answers(text_with_ranges, out, ["line-ranges"])
    outcome, prompter = _session_run(session, answers)
    assert outcome.action is Decision.RUN
    assert "? Which line ranges?" in prompter.transcript
    assert "? Tokens per chunk:" not in prompter.transcript
    spec = _printed_spec(session, outcome)
    assert spec.chunking.mode == "line-ranges"
    code, screen = _execute(session, outcome)
    assert code == 0, screen
    assert len(fake_llm.calls) == 2
    assert (out / "ledger_output.json").is_file()


def test_a_session_writes_missing_line_ranges_before_extracting(
    session: Session, text_file: Path, tmp_path: Path, fake_llm: FakeLLM
) -> None:
    out = tmp_path / "out"
    answers = _text_answers(text_file, out, ["line-ranges", KEEP])
    outcome, prompter = _session_run(session, answers)
    assert outcome.action is Decision.RUN
    assert "? Tokens per chunk:" in prompter.transcript
    spec = _printed_spec(session, outcome)
    assert spec.chunking.mode == "line-ranges"
    code, screen = _execute(session, outcome)
    assert code == 0, screen
    assert fake_llm.calls
    assert text_file.with_name("ledger_line_ranges.txt").is_file()
    assert (out / "ledger_output.json").is_file()


def test_a_session_extracts_a_pdf(
    session: Session, scanned_pdf: Path, tmp_path: Path, fake_llm: FakeLLM
) -> None:
    out = tmp_path / "out"
    outcome, prompter = _session_run(session, _pdf_answers(scanned_pdf, out, [], "run"))
    assert outcome.action is Decision.RUN
    assert "Plan: 1 to process, 0 complete" in prompter.transcript
    assert "? Which pages?" in prompter.transcript
    assert "? Page images:" in prompter.transcript
    spec = _printed_spec(session, outcome)
    assert spec.input.type == "auto"
    code, screen = _execute(session, outcome)
    assert code == 0, screen
    assert len(fake_llm.calls) == 3
    assert (out / "book_output.json").is_file()
    assert not (out / "book_output.csv").exists()
    assert "Run summary" in screen


def test_a_session_extracts_an_image_folder(
    session: Session, image_folder: Path, tmp_path: Path, fake_llm: FakeLLM
) -> None:
    out = tmp_path / "out"
    answers = _image_answers(image_folder, out, batch=False)
    outcome, prompter = _session_run(session, answers)
    assert outcome.action is Decision.RUN
    assert "? Which pages?" not in prompter.transcript
    spec = _printed_spec(session, outcome)
    assert spec.input.type == "images"
    assert spec.input.all is True
    assert spec.output.layout == "mirror"
    code, screen = _execute(session, outcome)
    assert code == 0, screen
    assert len(fake_llm.calls) == 3
    assert "3 complete, 0 failed" in screen


def test_a_session_submits_an_image_folder_to_the_batch_api(
    session: Session,
    image_folder: Path,
    tmp_path: Path,
    fake_batch: FakeBatchService,
) -> None:
    out = tmp_path / "out"
    answers = _image_answers(image_folder, out, batch=True)
    outcome, prompter = _session_run(session, answers)
    assert outcome.action is Decision.RUN
    assert "? Submission:" in prompter.transcript
    assert "Parallel requests:" not in prompter.transcript
    spec = _printed_spec(session, outcome)
    assert spec.run.batch is True
    assert "--batch" in list(outcome.command or ())
    code, screen = _execute(session, outcome)
    assert code == 0, screen
    assert fake_batch.batches
    assert fake_batch.llm.calls == []


@pytest.mark.usefixtures("fake_llm")
def test_a_dry_run_session_writes_nothing(
    session: Session,
    scanned_pdf: Path,
    tmp_path: Path,
    write_guard: WriteGuard,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    out = tmp_path / "out"
    outcome, _prompter = _session_run(
        session, _pdf_answers(scanned_pdf, out, ["dry_run"], "run")
    )
    assert outcome.answers["dry_run"] is True
    before = _files(scanned_pdf.parent, out)
    write_guard.start([tmp_path / "home", tmp_path / "tmp"])
    try:
        code, screen = _execute(session, outcome)
        stray = write_guard.consume()
    finally:
        write_guard.start([tmp_path_factory.getbasetemp()])
    assert code == 0, screen
    assert stray == []
    assert _files(scanned_pdf.parent, out) == before
    assert not out.exists()
    assert "Plan: 1 to process, 0 already complete" in screen


@pytest.mark.usefixtures("fake_llm")
def test_copy_returns_the_command_of_the_session(
    session: Session, scanned_pdf: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    outcome, prompter = _session_run(
        session, _pdf_answers(scanned_pdf, out, [], "copy")
    )
    assert outcome.action is Decision.COPY
    assert "Equivalent command:" in prompter.transcript
    _printed_spec(session, outcome)
    words = list(outcome.command or ())
    assert words[words.index("--schema") + 1] == SCHEMA
    assert words[words.index("--output") + 1] == str(out)
