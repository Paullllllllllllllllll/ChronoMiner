"""The extract wizard steps: options, offered choices, Back, skipping, markers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from chronominer.common.testing import KEEP, ScriptedPrompter, ScriptError
from chronominer.common.wizard import (
    BACK,
    Choice,
    Decision,
    State,
    StepContext,
)
from chronominer.spec import OPTIONS
from chronominer.ui.extract import flow
from chronominer.ui.session import Session, build_wizard, new_state
from chronominer.ui.wizard import (
    STEP_NAMES,
    build_steps,
    file_names,
    offered_chunking,
    offered_types,
    run_types,
    unit_word,
)
from tests.characterization.inputs import make_text_file, write_ranges
from tests.ui.conftest import SessionMaker

SCHEMA = "WageRecords"


class RecordingPrompter(ScriptedPrompter):
    """A scripted prompter that keeps the values each select offered."""

    def __init__(self, answers: Sequence[Any]) -> None:
        super().__init__(answers)
        self.offered: dict[str, list[Any]] = {}

    def select(self, message: str, choices: Sequence[Choice[Any]], **kw: Any) -> Any:
        self.offered[message] = [choice.value for choice in choices]
        return super().select(message, choices, **kw)


def _step(session: Session, name: str) -> Any:
    return next(step for step in build_steps(session) if step.name == name)


def _ask(
    session: Session, state: State, name: str, answers: Sequence[Any]
) -> tuple[Any, RecordingPrompter]:
    prompter = RecordingPrompter(answers)
    with prompter:
        result = _step(session, name).ask(StepContext(state, prompter))
    return result, prompter


def _pdf_answers(pdf: Path, images: Sequence[Any], end: Sequence[Any]) -> list[Any]:
    """Answers of a sync run on one PDF, ending with the review answers."""
    return [
        KEEP,  # task
        str(pdf),
        "pdf",
        KEEP,  # pages: all
        "beside",
        [],  # formats
        KEEP,  # provider
        KEEP,  # model
        KEEP,  # reasoning
        KEEP,  # model parameters
        KEEP,  # response format
        SCHEMA,
        KEEP,  # context: automatic
        KEEP,  # fallback context: none
        KEEP,  # context image
        *images,
        KEEP,  # decision pre-pass
        KEEP,  # submission
        [],  # run options
        KEEP,  # concurrency
        *end,
    ]


def test_the_steps_set_every_option_but_json_and_settings(session: Session) -> None:
    steps = build_steps(session)
    assert [step.name for step in steps] == list(STEP_NAMES)
    assert set().union(*(step.sets for step in steps)) == set(OPTIONS.names) - {
        "json",
        "settings",
    }


def test_the_type_question_offers_auto_and_the_types_found(
    session: Session, mixed_folder: Path, scanned_pdf: Path, tmp_path: Path
) -> None:
    state = new_state(session, OPTIONS)
    state.set("input", mixed_folder)
    assert offered_types(state) == ["auto", "text", "images", "pdf"]
    state.set("recursive", True)
    assert offered_types(state) == ["auto", "text", "images", "pdf"]
    state.set("input", scanned_pdf)
    assert offered_types(state) == ["auto", "pdf"]
    empty = tmp_path / "empty"
    empty.mkdir()
    state.set("input", empty)
    assert offered_types(state) == []


def test_the_run_types_follow_the_type_and_the_files(
    session: Session, mixed_folder: Path
) -> None:
    state = new_state(session, OPTIONS)
    state.set("input", mixed_folder)
    assert run_types(state) == {"text", "image", "pdf"}
    state.set("type", "pdf")
    assert run_types(state) == {"pdf"}
    assert unit_word(state) == "pages"
    state.set("type", "text")
    assert run_types(state) == {"text"}
    assert unit_word(state) == "chunks"


def test_which_items_is_asked_when_the_folder_holds_several_of_the_type(
    session: Session, tmp_path: Path
) -> None:
    folder = tmp_path / "ledgers"
    make_text_file(folder / "a.txt")
    make_text_file(folder / "b.txt")
    state = new_state(session, OPTIONS)
    state.set("input", folder)
    _result, prompter = _ask(session, state, "type", ["text", "list", ["b.txt"]])
    assert prompter.offered["Which items?"] == ["all", "numbers", "search", "list"]
    assert file_names(state) == ["a.txt", "b.txt"]
    assert state.value("files") == ("b.txt",)
    assert run_types(state) == {"text"}
    _result, prompter = _ask(session, state, "type", ["text", KEEP, KEEP])
    assert " > Choose from the list" in prompter.lines
    assert state.value("files") == ("b.txt",)
    _ask(session, state, "type", ["text", "all"])
    assert state.value("files") == ()
    assert state.value("all") is True
    _ask(session, state, "type", ["auto", "all"])
    assert state.value("files") == ()
    assert state.value("all") is True


def test_items_are_chosen_by_number_or_name_search(
    session: Session, tmp_path: Path
) -> None:
    folder = tmp_path / "ledgers"
    for name in ("ledger_2.txt", "ledger_10.txt", "notes.txt"):
        make_text_file(folder / name)
    state = new_state(session, OPTIONS)
    state.set("input", folder)
    _result, prompter = _ask(session, state, "type", ["text", "numbers", "1-2"])
    assert "     1  ledger_2.txt" in prompter.lines
    assert state.value("select") == "1-2"
    assert state.value("all") is False
    assert file_names(state)[:2] == ["ledger_2.txt", "ledger_10.txt"]
    with pytest.raises(ScriptError, match="rejected"):
        _ask(session, state, "type", ["text", "numbers", "4"])
    _ask(session, state, "type", ["text", "search", "notes"])
    assert state.value("select") == "notes"
    assert run_types(state) == {"text"}
    with pytest.raises(ScriptError, match="rejected"):
        _ask(session, state, "type", ["text", "search", "2"])


def test_every_chunking_mode_is_offered_for_text(
    session: Session, text_file: Path, text_with_ranges: Path, tmp_path: Path
) -> None:
    state = new_state(session, OPTIONS)
    state.set("input", text_file)
    assert offered_chunking(state) == ["auto", "line-ranges", "adjust"]
    _result, prompter = _ask(session, state, "chunking", ["line-ranges", "2000"])
    assert prompter.offered["Text chunking:"] == ["auto", "line-ranges", "adjust"]
    assert state.value("chunking") == "line-ranges"
    assert state.value("chunk_tokens") == 2000

    state = new_state(session, OPTIONS)
    state.set("input", text_with_ranges)
    _result, prompter = _ask(session, state, "chunking", ["adjust", "4"])
    assert state.value("chunking") == "adjust"
    assert state.value("context_window") == 4
    assert unit_word(state) == "line ranges"
    _ask(session, state, "chunking", ["line-ranges"])
    assert state.answers["chunking"] == "line-ranges"
    assert "context_window" not in state.answers

    folder = text_with_ranges.parent
    make_text_file(folder / "other.txt")
    write_ranges(folder / "other.txt", [(1, 10)])
    fresh = new_state(session, OPTIONS)
    fresh.set("input", folder)
    assert offered_chunking(fresh) == ["auto", "line-ranges", "adjust"]


def test_a_settings_chunking_is_kept_without_line_ranges(
    make_session: SessionMaker, text_file: Path
) -> None:
    session = make_session({"chunking": "adjust"})
    state = new_state(session, OPTIONS)
    state.set("input", text_file)
    _ask(session, state, "chunking", [KEEP, KEEP, KEEP])
    assert state.value("chunking") == "adjust"
    assert "chunking" not in state.answers


def test_back_returns_to_the_previous_step_and_keeps_its_answers(
    session: Session, scanned_pdf: Path
) -> None:
    answers = _pdf_answers(scanned_pdf, [KEEP], ["quit"])
    where = answers.index("beside")
    prompter = ScriptedPrompter(
        [
            *answers[:where],
            BACK,  # on "Where should the outputs go?"
            BACK,  # on "Which pages?"
            KEEP,  # input type: pdf stays preselected
            KEEP,  # pages
            *answers[where:],
        ]
    )
    with prompter:
        outcome = build_wizard(flow(), session, prompter).run()
    assert outcome.action is Decision.QUIT
    assert outcome.answers["type"] == "pdf"
    assert outcome.answers["schema"] == SCHEMA
    assert prompter.transcript.count("? Input type:") == 2


def test_steps_of_other_inputs_are_skipped(
    session: Session, text_file: Path, scanned_pdf: Path, image_folder: Path
) -> None:
    wizard = build_wizard(flow(), session, ScriptedPrompter([]))
    state = wizard.state

    def applicable() -> list[str]:
        return [step.name for step in wizard.applicable()]

    common = ["task", "input", "type"]
    tail = ["output", "model", "response"]
    state.set("input", text_file)
    assert applicable() == [*common, "chunking", "pages", *tail, "decisions", "run"]
    state.set("input", scanned_pdf)
    assert applicable() == [*common, "pages", *tail, "images", "decisions", "run"]
    state.set("input", image_folder)
    assert applicable() == [*common, *tail, "images", "decisions", "run"]


def test_a_step_applying_again_after_an_edit_gets_its_answers_back(
    session: Session, scanned_pdf: Path, text_file: Path
) -> None:
    images = ["customize", "200", KEEP, KEEP, KEEP, KEEP]
    prompter = ScriptedPrompter(
        [
            *_pdf_answers(scanned_pdf, images, []),
            "edit",
            "input",
            str(text_file),
            KEEP,  # chunking mode
            KEEP,  # tokens per chunk
            "edit",
            "input",
            str(scanned_pdf),
            "quit",
        ]
    )
    with prompter:
        outcome = build_wizard(flow(), session, prompter).run()
    assert outcome.action is Decision.QUIT, prompter.transcript
    assert outcome.answers["input"] == scanned_pdf
    assert outcome.answers["dpi"] == 200
    assert "chunking" not in outcome.answers
    assert prompter.transcript.count("Edit which step?") == 2


def test_an_inapplicable_step_drops_its_answers(
    session: Session, scanned_pdf: Path, text_file: Path
) -> None:
    images = ["customize", "200", KEEP, KEEP, KEEP, KEEP]
    prompter = ScriptedPrompter(
        [
            *_pdf_answers(scanned_pdf, images, []),
            "edit",
            "input",
            str(text_file),
            KEEP,  # chunking mode
            KEEP,  # tokens per chunk
            "quit",
        ]
    )
    with prompter:
        outcome = build_wizard(flow(), session, prompter).run()
    assert outcome.action is Decision.QUIT
    assert outcome.answers["input"] == text_file
    assert "type" not in outcome.answers
    assert "dpi" not in outcome.answers
    reviews = [i for i, line in enumerate(prompter.lines) if line.startswith("Review")]
    last = prompter.lines[reviews[-1] :]
    assert any(line.startswith("  chunking ") for line in last)
    assert not any(line.startswith("  images ") for line in last)


def test_settings_defaults_are_preselected_and_marked(
    make_session: SessionMaker, scanned_pdf: Path
) -> None:
    session = make_session({"formats": ["csv"], "reasoning_effort": "medium"})
    answers = _pdf_answers(scanned_pdf, [KEEP], ["quit"])
    answers[answers.index("beside") + 1] = KEEP
    prompter = ScriptedPrompter(answers)
    with prompter:
        outcome = build_wizard(flow(), session, prompter).run()
    assert outcome.action is Decision.QUIT
    lines = prompter.lines
    assert "   [x] CSV (_output.csv) (s)" in lines
    assert " > medium (s)" in lines
    assert "formats" not in outcome.answers
    assert "(s) = from settings defaults" in prompter.transcript
    assert "reasoning medium (s)" in prompter.transcript


def test_the_pages_step_checks_the_range(session: Session, scanned_pdf: Path) -> None:
    state = new_state(session, OPTIONS)
    state.set("input", scanned_pdf)
    state.set("type", "pdf")
    _ask(session, state, "pages", ["range", "2-3"])
    assert state.value("pages") == "2-3"
    with pytest.raises(ScriptError, match="rejected"):
        _ask(session, state, "pages", ["range", "3-1"])
    _ask(session, state, "pages", ["all"])
    assert state.value("pages") is None


def test_the_layout_is_asked_only_for_folder_inputs(
    session: Session, scanned_pdf: Path, image_folder: Path, tmp_path: Path
) -> None:
    out = str(tmp_path / "out")
    state = new_state(session, OPTIONS)
    state.set("input", scanned_pdf)
    _result, prompter = _ask(session, state, "output", ["folder", out, []])
    assert "Folder layout:" not in prompter.offered
    state.set("input", image_folder)
    _result, prompter = _ask(session, state, "output", ["folder", out, "mirror", []])
    assert prompter.offered["Folder layout:"] == ["flat", "mirror"]
    assert state.value("layout") == "mirror"
    _ask(session, state, "output", ["beside", []])
    assert state.value("output") is None
