"""Every input type, chunking, response format and submission the wizard offers
plans without error."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from chronominer.common.testing import KEEP, ScriptedPrompter
from chronominer.common.wizard import BACK, DONE, Choice, State, StepContext
from chronominer.run import plan
from chronominer.spec import OPTIONS
from chronominer.ui.questions import resolution_of
from chronominer.ui.session import Session, new_state
from chronominer.ui.wizard import build_steps
from tests.ui.conftest import DUMMY_KEYS, ENDPOINT, SessionMaker

SCHEMA = "WageRecords"
TYPE_MESSAGE = "Input type:"
CHUNKING_MESSAGE = "Text chunking:"
FORMAT_MESSAGE = "Response format:"
SUBMISSION_MESSAGE = "Submission:"


class OfferPrompter(ScriptedPrompter):
    """A scripted prompter that records offered values and takes every file."""

    def __init__(self, answers: Sequence[Any]) -> None:
        super().__init__(answers)
        self.offered: dict[str, list[Any]] = {}

    def select(self, message: str, choices: Sequence[Choice[Any]], **kw: Any) -> Any:
        self.offered[message] = [choice.value for choice in choices]
        if message.startswith("Which "):
            return "all"
        return super().select(message, choices, **kw)


@pytest.fixture(autouse=True)
def keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set the dummy provider keys the plan checks."""
    for name, value in DUMMY_KEYS.items():
        monkeypatch.setenv(name, value)


def _ask(
    session: Session, state: State, name: str, answers: Sequence[Any]
) -> tuple[Any, OfferPrompter]:
    step = next(step for step in build_steps(session) if step.name == name)
    prompter = OfferPrompter(answers)
    with prompter:
        result = step.ask(StepContext(state, prompter))
    return result, prompter


def _offered(
    session: Session, state: State, name: str, message: str, default: list[Any]
) -> list[Any]:
    """Return what the first question of step *name* offers, Back answered."""
    result, prompter = _ask(session, state, name, [BACK])
    assert result is BACK
    return prompter.offered.get(message, default)


def _state(session: Session, location: Path) -> State:
    state = new_state(session, OPTIONS)
    state.set("schema", SCHEMA)
    state.set("input", location)
    return state


INPUTS: dict[str, Callable[[dict[str, Path]], Path]] = {
    "text": lambda fx: fx["text"],
    "text-with-ranges": lambda fx: fx["ranged"],
    "image-folder": lambda fx: fx["images"],
    "pdf": lambda fx: fx["pdf"],
    "mixed": lambda fx: fx["mixed"],
}
RANGES_OFFERED = {"text": True, "text-with-ranges": True, "mixed": True}


@pytest.fixture
def located(
    request: pytest.FixtureRequest,
    text_file: Path,
    text_with_ranges: Path,
    image_folder: Path,
    scanned_pdf: Path,
    mixed_folder: Path,
) -> tuple[str, Path]:
    name: str = request.param
    fixtures = {
        "text": text_file,
        "ranged": text_with_ranges,
        "images": image_folder,
        "pdf": scanned_pdf,
        "mixed": mixed_folder,
    }
    return name, INPUTS[name](fixtures)


@pytest.mark.parametrize("located", list(INPUTS), indirect=True)
def test_every_offered_choice_plans(
    located: tuple[str, Path], session: Session
) -> None:
    label, location = located
    types = _offered(session, _state(session, location), "type", TYPE_MESSAGE, [])
    assert types[0] == "auto"
    for kind in types:
        state = _state(session, location)
        result, _prompter = _ask(session, state, "type", [kind])
        assert result is DONE
        texts = "chunking" in [s.name for s in build_steps(session) if s.applies(state)]
        modes = ["auto"]
        if texts:
            modes = _offered(session, state, "chunking", CHUNKING_MESSAGE, ["auto"])
            if label in RANGES_OFFERED and kind in ("auto", "text"):
                assert ("line-ranges" in modes) is RANGES_OFFERED[label]
                assert ("adjust" in modes) is RANGES_OFFERED[label]
        formats = _offered(session, state, "response", FORMAT_MESSAGE, [None])
        batches = _offered(session, state, "run", SUBMISSION_MESSAGE, [False])
        assert batches == [False, True]
        for mode, fmt, batch in itertools.product(modes, formats, batches):
            trial = _state(session, location)
            _ask(session, trial, "type", [kind])
            if texts:
                trial.set("chunking", mode)
            trial.set("response_format", fmt)
            trial.set("batch", batch)
            spec = resolution_of(trial, session).spec
            plan(spec, session.settings)


@pytest.mark.parametrize(
    "answers",
    [
        {"provider": "openrouter", "model": "openai/gpt-5.6-luna"},
        {"provider": "custom", "endpoint": ENDPOINT, "model": "local-model"},
    ],
)
def test_batch_is_not_offered_without_a_batch_api(
    make_session: SessionMaker, scanned_pdf: Path, answers: dict[str, Any]
) -> None:
    session = make_session({"batch": True})
    state = _state(session, scanned_pdf)
    for name, value in answers.items():
        state.set(name, value)
    result, prompter = _ask(session, state, "run", [[], KEEP])
    assert result is DONE
    assert SUBMISSION_MESSAGE not in prompter.offered
    assert state.value("batch") is False
    plan(resolution_of(state, session).spec, session.settings)


def test_batch_is_offered_for_each_batch_provider(
    session: Session, scanned_pdf: Path
) -> None:
    for model in ("gpt-5.6-luna", "claude-sonnet-5-5", "gemini-3.1-pro"):
        state = _state(session, scanned_pdf)
        state.set("model", model)
        offered = _offered(session, state, "run", SUBMISSION_MESSAGE, [])
        assert offered == [False, True], model
