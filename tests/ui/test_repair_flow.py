"""The repair task of the wizard: steps, review, command and run.

Run on Windows with:

    uv --directory <repo> run pytest tests/ui/test_repair_flow.py -q

Each session is scripted through :func:`build_wizard`; the stored documents
come from the helpers of ``tests/batch/test_repair_store.py`` and the model
is their fake extractor. No request leaves the machine.
"""

from __future__ import annotations

import argparse
import asyncio
import io
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from chronominer import ext
from chronominer.batch import repair as repair_module
from chronominer.common.testing import KEEP, ScriptedPrompter
from chronominer.common.wizard import (
    BACK,
    SETTINGS_LEGEND,
    Decision,
    Outcome,
    State,
    StepContext,
    render_review,
)
from chronominer.spec import REPAIR_OPTIONS, CommandSpec, parse_repair
from chronominer.ui import repair as repair_ui
from chronominer.ui.session import (
    RunRequest,
    Session,
    absolute,
    build_wizard,
    new_state,
)
from tests.batch.conftest import answer
from tests.batch.test_repair_store import (
    FakeExtractor,
    Stored,
    chunk_label,
    page_name,
    store_document,
    unit_rows,
)
from tests.ui.conftest import SessionMaker

MODEL_KEEPS = [KEEP] * 4
"""Provider, model, reasoning effort and the recommended tuning."""
RESPONSE_KEEPS = [KEEP]
"""Response format."""
IMAGE_KEEPS = [KEEP]
"""The recommended page image settings."""
RUN_KEEPS = [KEEP, KEEP]
"""Run mode and dry run."""


def stored(tmp_path: Path, outcomes: Sequence[str], **options: Any) -> Stored:
    return asyncio.run(store_document(tmp_path, outcomes, **options))


@pytest.fixture
def extractor(monkeypatch: pytest.MonkeyPatch) -> type[FakeExtractor]:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(repair_module, "ChunkExtractor", FakeExtractor)
    FakeExtractor.texts = {}
    FakeExtractor.calls = []
    FakeExtractor.created = []
    FakeExtractor.context_images = []
    return FakeExtractor


def session_of(
    session: Session, answers: Sequence[Any]
) -> tuple[Outcome, State, ScriptedPrompter]:
    prompter = ScriptedPrompter(answers)
    wizard = build_wizard(repair_ui.flow(), session, prompter, width=100)
    with prompter:
        outcome = wizard.run()
    return outcome, wizard.state, prompter


def run(session: Session, answers: Any) -> tuple[int, str]:
    console = Console(file=io.StringIO(), width=200)
    request = RunRequest(session, answers, console, argparse.Namespace(command="ui"))
    code = repair_ui.flow().run(request)
    output = console.file.getvalue()  # type: ignore[attr-defined]
    return code, output


def script(
    path: Path, *documents: Any, pages: bool = True, start: str = "run"
) -> list[Any]:
    return [
        KEEP,
        str(path),
        *(documents or (KEEP, KEEP)),
        *MODEL_KEEPS,
        *RESPONSE_KEEPS,
        *(IMAGE_KEEPS if pages else []),
        *RUN_KEEPS,
        start,
    ]


def settings_words(session: Session) -> tuple[str, str]:
    return ("--settings", str(absolute(session.settings_path or "")))


def test_the_steps_set_every_repair_option_but_json_and_settings(
    session: Session,
) -> None:
    sets = {name for step in repair_ui.flow().steps(session) for name in step.sets}
    assert sets == set(REPAIR_OPTIONS.names) - {"json", "settings"}


def test_a_scripted_session_repairs_the_failed_pages(
    tmp_path: Path, session: Session, extractor: type[FakeExtractor]
) -> None:
    document = stored(tmp_path, ["ok", "failed", "ok", "failed"])
    extractor.texts = {page_name(1): "Fixed one", page_name(3): "Fixed three"}
    answers = [
        KEEP,
        str(document.output),
        BACK,
        str(document.output),
        KEEP,
        "",
        *MODEL_KEEPS,
        *RESPONSE_KEEPS,
        *IMAGE_KEEPS,
        *RUN_KEEPS,
        "run",
    ]

    outcome, state, prompter = session_of(session, answers)

    assert outcome.action is Decision.RUN
    assert outcome.command == (
        "chronominer",
        "repair",
        "--input",
        str(document.output),
        *settings_words(session),
    )
    assert "? Page images:" in prompter.lines
    assert "Chunks to repair: 2 in 1 document" in prompter.transcript
    assert "'doc' in" in prompter.transcript
    spec = repair_ui.flow().resolve(state, session).spec
    assert parse_repair(outcome.command[2:], session.settings.defaults) == spec

    code, output = run(session, outcome.answers)

    assert code == 0
    assert [unit[3] for unit in unit_rows(document.db)] == [
        answer("Chunk 0"),
        answer("Fixed one"),
        answer("Chunk 2"),
        answer("Fixed three"),
    ]
    assert "Repair of doc_output.json" in output


def test_text_chunks_skip_the_page_image_step(
    tmp_path: Path, session: Session, extractor: type[FakeExtractor]
) -> None:
    document = stored(tmp_path, ["ok", "failed"], kind="text")
    extractor.texts = {chunk_label(1): "Fixed text"}

    outcome, state, prompter = session_of(session, script(document.output, pages=False))

    assert "? Page images:" not in prompter.lines
    assert not any(line.startswith("  images") for line in prompter.lines)
    assert outcome.command is not None
    spec = repair_ui.flow().resolve(state, session).spec
    assert parse_repair(outcome.command[2:], session.settings.defaults) == spec
    assert run(session, outcome.answers)[0] == 0
    assert unit_rows(document.db)[1][3] == answer("Fixed text")


def test_the_documents_step_spells_schema_and_indices(
    tmp_path: Path, session: Session, extractor: type[FakeExtractor]
) -> None:
    document = stored(tmp_path, ["failed", "ok", "failed"])
    schema = "WageRecords"

    outcome, state, prompter = session_of(
        session, script(tmp_path, schema, "2", start="quit")
    )

    spec = repair_ui.flow().resolve(state, session).spec
    assert (spec.schema, spec.indices) == (schema, (2,))
    assert outcome.command is not None
    assert "--schema" in outcome.command and "--indices" in outcome.command
    assert parse_repair(outcome.command[2:], session.settings.defaults) == spec
    text = render_review(repair_ui.flow().review(state, session))
    assert f"schema {schema} | indices 2" in text
    assert "Chunks to repair: 1 in 1 document" in text
    assert str(document.db) in text


def test_the_review_marks_settings_defaults_and_previews_the_plan(
    tmp_path: Path, make_session: SessionMaker, extractor: type[FakeExtractor]
) -> None:
    document = stored(tmp_path, ["failed", "ok"])
    session = make_session({"reasoning_effort": "low"})

    outcome, state, _prompter = session_of(
        session, script(document.output, start="quit")
    )

    review = repair_ui.flow().review(state, session)
    text = render_review(review)
    assert SETTINGS_LEGEND in text
    assert "reasoning low (s)" in text
    lines = {line.split()[0]: line.split(None, 1)[1] for line in text.splitlines()[1:6]}
    assert lines["documents"].startswith(str(document.output))
    assert lines["documents"].endswith("every schema | every failed chunk")
    assert lines["run"] == "sync"
    assert review.withheld is None
    assert review.notes[0] == "Chunks to repair: 1 in 1 document"
    assert outcome.command is not None
    words = outcome.command[2:]
    assert words[words.index("--reasoning-effort") + 1] == "low"
    spec = repair_ui.flow().resolve(state, session).spec
    assert parse_repair(words, session.settings.defaults) == spec


def test_the_review_shows_the_condition_warnings(
    tmp_path: Path, make_session: SessionMaker, extractor: type[FakeExtractor]
) -> None:
    document = stored(tmp_path, ["failed"], kind="text")
    session = make_session({"service_tier": "priority"})
    state = new_state(session, REPAIR_OPTIONS)
    state.set("input", document.output)
    state.set("model", "claude-sonnet-4-6")

    review = repair_ui.flow().review(state, session)

    warnings = [note for note in review.notes if note.startswith("Warning: ")]
    assert len(warnings) == 1
    assert "the settings default service_tier (--service-tier)" in warnings[0]
    words = repair_ui.flow().command(state, session)
    assert "--service-tier" not in words


def ask_run(state: State, session: Session, answers: Sequence[Any]) -> list[str]:
    prompter = ScriptedPrompter(answers)
    (step,) = [s for s in repair_ui.flow().steps(session) if s.name == "run"]
    with prompter:
        step.ask(StepContext(state, prompter))
    return prompter.lines


def test_batch_is_offered_for_a_provider_with_a_batch_api(
    session: Session,
) -> None:
    state = new_state(session, REPAIR_OPTIONS)

    lines = ask_run(state, session, [True, KEEP])

    assert "? Run mode:" in lines
    assert state.value("batch") is True


@pytest.mark.parametrize(
    "answers",
    [
        {"provider": "openrouter", "model": "openai/gpt-5.6-luna"},
        {"endpoint": "local", "model": "local-model"},
    ],
    ids=["openrouter", "endpoint"],
)
def test_batch_is_not_offered_without_a_batch_api(
    session: Session, answers: dict[str, Any]
) -> None:
    state = new_state(session, REPAIR_OPTIONS)
    state.set("batch", True)
    for name, value in answers.items():
        state.set(name, value)

    lines = ask_run(state, session, [KEEP])

    assert "? Run mode:" not in lines
    assert "? Dry run:" in lines
    assert state.value("batch") is False


def test_batch_is_not_offered_when_the_extension_refuses_it(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ext, "batch_allowed", lambda spec, **_: False)
    state = new_state(session, REPAIR_OPTIONS)

    lines = ask_run(state, session, [KEEP])

    assert "? Run mode:" not in lines


def test_run_is_withheld_when_the_repair_cannot_be_planned(
    tmp_path: Path, session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    document = stored(tmp_path, ["failed"])

    _outcome, state, prompter = session_of(
        session, script(document.output, start="quit")
    )

    review = repair_ui.flow().review(state, session)
    assert review.withheld is not None and "OPENAI_API_KEY" in review.withheld
    assert "   Run" not in prompter.lines
    assert "Run unavailable:" in prompter.transcript


def test_a_plan_error_exits_2_without_startup(
    tmp_path: Path, session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[CommandSpec] = []
    monkeypatch.setattr(ext, "startup", lambda spec, **_: calls.append(spec))
    code, output = run(session, {"input": tmp_path / "missing.json"})
    assert code == 2
    assert "Input path does not exist" in output
    assert calls == []


def test_a_dry_run_starts_the_extension_and_writes_nothing(
    tmp_path: Path,
    session: Session,
    extractor: type[FakeExtractor],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = stored(tmp_path, ["failed", "ok"])
    calls: list[CommandSpec] = []
    monkeypatch.setattr(ext, "startup", lambda spec, **_: calls.append(spec))
    before = unit_rows(document.db)

    code, output = run(session, {"input": document.output, "dry_run": True})

    assert code == 0
    assert len(calls) == 1
    assert "[DRY RUN] 1 chunk(s) would be repaired." in output
    assert unit_rows(document.db) == before
    assert extractor.calls == []


def test_a_chunk_that_fails_again_exits_1(
    tmp_path: Path, session: Session, extractor: type[FakeExtractor]
) -> None:
    document = stored(tmp_path, ["failed"])
    extractor.texts = {page_name(0): None}
    code, _output = run(session, {"input": document.output})
    assert code == 1
