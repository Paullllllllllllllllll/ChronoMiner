"""The extract review: lines, markers, warnings, the plan, Force and the command."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from chronominer import run as core
from chronominer.common.wizard import Part, Review, State, render_review
from chronominer.settings import resolve
from chronominer.spec import OPTIONS, parse
from chronominer.ui.review import (
    FORCE_ACTION,
    PROGRAM,
    build_review,
    equivalent_command,
)
from chronominer.ui.session import Session, absolute
from tests.characterization.fakes import PROVIDER_CLASSES, FakeLLM
from tests.characterization.seams import replace
from tests.ui.conftest import DUMMY_KEYS, SessionMaker

SCHEMA = "WageRecords"


@pytest.fixture
def keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set the dummy provider keys the plan checks."""
    for name, value in DUMMY_KEYS.items():
        monkeypatch.setenv(name, value)


def _state(session: Session, **answers: Any) -> State:
    answers.setdefault("schema", SCHEMA)
    return State(OPTIONS, session.settings.defaults, answers)


def _lines(review: Review) -> dict[str, list[str]]:
    return {
        line.label: [
            part.text if isinstance(part, Part) else part for part in line.parts
        ]
        for line in review.lines
    }


def _marked(review: Review) -> set[str]:
    return {
        part.text
        for line in review.lines
        for part in line.parts
        if isinstance(part, Part) and part.from_settings
    }


@pytest.mark.usefixtures("keys")
def test_the_lines_of_a_pdf_run(
    make_session: SessionMaker, scanned_pdf: Path, tmp_path: Path
) -> None:
    session = make_session({"reasoning_effort": "medium", "formats": ["csv"]})
    out = tmp_path / "out"
    state = _state(
        session,
        input=scanned_pdf,
        type="pdf",
        output=out,
        pages="2-3",
        concurrency=4,
    )
    review = build_review(state, session)
    lines = _lines(review)
    assert list(lines) == [
        "input",
        "output",
        "schema",
        "model",
        "access",
        "response",
        "images",
        "run",
    ]
    assert lines["access"] == ["API key OPENAI_API_KEY"]
    assert lines["input"] == [
        f"{scanned_pdf}  (1 PDF, 3 pages)",
        "type pdf",
        "pages 2-3",
    ]
    assert lines["output"] == [
        str(out),
        "layout flat",
        "formats _output.json, csv (+ chronominer.sqlite)",
    ]
    assert lines["schema"][0] == SCHEMA
    assert lines["schema"][1].endswith("wage_records.json")
    assert lines["model"][:2] == ["openai / gpt-6-luna", "reasoning medium"]
    assert SCHEMA not in lines["response"]
    assert lines["run"] == ["sync", "resume", "concurrency 4"]
    assert _marked(review) == {
        "reasoning medium",
        "formats _output.json, csv (+ chronominer.sqlite)",
    }
    assert review.withheld is None
    assert review.notes[:2] == [
        "Plan: 1 to process, 0 complete",
        "  book.pdf: new (pdf)",
    ]
    screen = render_review(review)
    assert screen.splitlines()[0].endswith("(s) = from settings defaults")
    assert "reasoning medium (s)" in screen


@pytest.mark.usefixtures("keys")
def test_the_chunking_line_of_a_text_run(
    session: Session, text_file: Path, text_with_ranges: Path
) -> None:
    lines = _lines(build_review(_state(session, input=text_file), session))
    assert lines["chunking"] == ["auto", "3500 tokens per chunk"]
    assert lines["input"][-1] == "all chunks"
    assert "images" not in lines
    adjust = build_review(
        _state(
            session,
            input=text_with_ranges,
            chunking="adjust",
            context_window=4,
            pages="1",
        ),
        session,
    )
    lines = _lines(adjust)
    assert lines["chunking"] == [
        "adjust",
        "from _line_ranges.txt",
        "4 lines around each boundary",
    ]
    assert lines["input"][-1] == "line ranges 1"
    assert adjust.withheld is None


def test_run_is_withheld_while_a_key_is_missing(
    session: Session, scanned_pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    review = build_review(_state(session, input=scanned_pdf), session)
    assert review.withheld == "set OPENAI_API_KEY in the environment"
    assert review.notes[0] == "Plan: 1 to process, 0 complete"
    assert any(
        note.startswith("Missing API key: OPENAI_API_KEY is not set")
        for note in review.notes
    )
    assert "Run unavailable" in render_review(review)
    dry = build_review(_state(session, input=scanned_pdf, dry_run=True), session)
    assert dry.withheld is None
    assert any("OPENAI_API_KEY" in note for note in dry.notes)


@pytest.mark.usefixtures("keys")
def test_run_is_withheld_for_an_unknown_schema(
    session: Session, scanned_pdf: Path
) -> None:
    review = build_review(
        _state(session, input=scanned_pdf, schema="NoSuchSchema"), session
    )
    assert _lines(review)["schema"] == ["NoSuchSchema"]
    assert review.withheld == "the plan failed; edit a step"


def test_unresolvable_answers_withhold_run(session: Session, scanned_pdf: Path) -> None:
    review = build_review(
        _state(
            session,
            input=scanned_pdf,
            provider="openai",
            model="anthropic:claude-sonnet-5-5",
        ),
        session,
    )
    assert review.lines == ()
    assert review.withheld is not None
    assert "anthropic" in review.withheld


@pytest.mark.usefixtures("keys")
def test_condition_warnings_are_shown_among_the_notes(
    make_session: SessionMaker, scanned_pdf: Path
) -> None:
    session = make_session({"reasoning_effort": "medium"})
    state = _state(
        session,
        input=scanned_pdf,
        model="claude-sonnet-5-5",
        service_tier="priority",
    )
    review = build_review(state, session)
    [warning] = [note for note in review.notes if note.startswith("Warning: ")]
    assert warning.startswith("Warning: --service-tier applies")
    assert "the provider is anthropic" in warning
    assert review.withheld is None
    screen = render_review(review)
    assert "reasoning medium (s)" in screen
    assert warning in screen.splitlines()

    from_settings = make_session({"service_tier": "priority"})
    review = build_review(
        _state(from_settings, input=scanned_pdf, model="claude-sonnet-5-5"),
        from_settings,
    )
    assert any(
        note.startswith("Warning: the settings default service_tier")
        for note in review.notes
    )
    openai = build_review(
        _state(session, input=scanned_pdf, service_tier="priority"), session
    )
    assert not any(note.startswith("Warning: ") for note in openai.notes)


@pytest.mark.usefixtures("keys")
def test_a_force_action_is_offered_for_complete_items(
    session: Session, text_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = FakeLLM()
    for module_name, name in PROVIDER_CLASSES:
        replace(monkeypatch, module_name, name, controller.constructor(name))
    answers = {"input": text_file, "schema": SCHEMA}
    resolved = resolve(session.settings, answers)
    first = core.plan(resolved.spec, session.settings, sources=resolved.sources)
    assert asyncio.run(core.run(first)).failed == 0

    state = _state(session, **answers)
    review = build_review(state, session)
    assert review.notes[:2] == [
        "Plan: 0 to process, 1 complete",
        "  ledger.txt: complete (skipped)",
    ]
    [action] = review.actions
    assert action.key == FORCE_ACTION
    assert action.title == "Extract 1 complete item again (--force)"
    action.apply(state)
    assert state.value("force") is True
    forced = build_review(state, session)
    assert forced.actions == []
    assert _lines(forced)["run"][1] == "force"


@pytest.mark.parametrize(
    "defaults",
    [
        {},
        {"model": "claude-sonnet-5-5", "formats": ["csv"], "concurrency": 3},
        {"top_p": 0.5, "layout": "mirror", "recursive": True, "chunking": "adjust"},
    ],
)
def test_the_equivalent_command_parses_back_to_the_spec(
    make_session: SessionMaker,
    mixed_folder: Path,
    tmp_path: Path,
    defaults: Mapping[str, Any],
) -> None:
    session = make_session(defaults)
    state = _state(
        session,
        input=mixed_folder,
        type="pdf",
        output=tmp_path / "out",
        model="gpt-5.6-terra",
        pages="first:2",
        force=True,
    )
    words = equivalent_command(state, session)
    assert tuple(words[:2]) == PROGRAM
    assert words[-2:] == ["--settings", str(absolute(session.settings_path or ""))]
    spec = resolve(session.settings, state.answers).spec
    assert parse(words[2:], session.settings.defaults) == spec
    assert parse(words[2:-2]) == spec


def test_the_equivalent_command_names_the_service_tier_only_for_openai(
    make_session: SessionMaker, mixed_folder: Path, tmp_path: Path
) -> None:
    session = make_session({"service_tier": "priority"})
    answers: dict[str, Any] = {
        "input": mixed_folder,
        "type": "pdf",
        "output": tmp_path / "out",
        "schema": SCHEMA,
    }
    openai = equivalent_command(_state(session, **answers), session)
    assert openai[openai.index("--service-tier") + 1] == "priority"
    answers["model"] = "claude-sonnet-5-5"
    anthropic = equivalent_command(_state(session, **answers), session)
    assert not any(word.startswith("--service-tier") for word in anthropic)
    spec = resolve(session.settings, answers).spec
    assert parse(anthropic[2:], session.settings.defaults) == spec
