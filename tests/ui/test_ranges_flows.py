"""The ranges generate and adjust tasks of the wizard: steps, review, command, run.

Run on Windows with:

    uv --directory <repo> run pytest tests/ui/test_ranges_flows.py -q

Each session is scripted through :func:`build_wizard`. Generate calls no
model; adjust answers its boundary calls through the characterization
``FakeLLM``. No request leaves the machine.
"""

from __future__ import annotations

import argparse
import io
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from chronominer import ext
from chronominer.common.testing import KEEP, ScriptedPrompter
from chronominer.common.usage import Usage
from chronominer.common.wizard import (
    BACK,
    SETTINGS_LEGEND,
    Decision,
    Nav,
    Outcome,
    State,
    StepContext,
    render_review,
)
from chronominer.output.store import DATABASE_NAME
from chronominer.ranges.lines import ranges_path, read_ranges
from chronominer.run import ItemResult, RangesPlan, RunResult
from chronominer.spec import (
    RANGES_ADJUST_OPTIONS,
    RANGES_GENERATE_OPTIONS,
    CommandSpec,
    parse_ranges_adjust,
    parse_ranges_generate,
)
from chronominer.ui import ranges as ranges_ui
from chronominer.ui.session import (
    RunRequest,
    Session,
    TaskFlow,
    absolute,
    build_wizard,
    new_state,
)
from tests.characterization.fakes import PROVIDER_CLASSES, FakeLLM
from tests.characterization.inputs import make_text_file, write_ranges
from tests.characterization.seams import replace
from tests.ui.conftest import SessionMaker

SCHEMA = "PriceListEntries"
MID_RECORD = [(1, 20), (21, 38), (39, 56), (57, 71)]
"""Starts 21, 39 and 57 lie inside records; the next headings are 25, 43, 61."""
MODEL_KEEPS = [KEEP] * 4
"""Provider, model, reasoning effort and the recommended tuning."""

FLOWS: Mapping[str, Callable[[], TaskFlow]] = {
    "generate": ranges_ui.generate_flow,
    "adjust": ranges_ui.adjust_flow,
}


@pytest.fixture
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> FakeLLM:
    """Replace the provider classes with ``FakeLLM`` stand-ins."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    controller = FakeLLM()
    for module_name, name in PROVIDER_CLASSES:
        replace(monkeypatch, module_name, name, controller.constructor(name))
    return controller


def session_of(
    flow: TaskFlow, session: Session, prompter: ScriptedPrompter
) -> tuple[Outcome, State]:
    wizard = build_wizard(flow, session, prompter, width=100)
    with prompter:
        outcome = wizard.run()
    return outcome, wizard.state


def scripted(
    flow: TaskFlow, session: Session, answers: Sequence[Any]
) -> tuple[Outcome, State, ScriptedPrompter]:
    prompter = ScriptedPrompter(answers)
    outcome, state = session_of(flow, session, prompter)
    return outcome, state, prompter


def run(flow: TaskFlow, session: Session, answers: Any) -> tuple[int, str]:
    console = Console(file=io.StringIO(), width=200)
    request = RunRequest(session, answers, console, argparse.Namespace(command="ui"))
    code = flow.run(request)
    output = console.file.getvalue()  # type: ignore[attr-defined]
    return code, output


def settings_words(session: Session) -> list[str]:
    return ["--settings", str(absolute(session.settings_path or ""))]


def generate_script(path: Path, *, start: str = "run") -> list[Any]:
    return [KEEP, str(path), "60", KEEP, KEEP, start]


def adjust_script(path: Path, *, start: str = "run") -> list[Any]:
    return [
        KEEP,
        str(path),
        KEEP,
        SCHEMA,
        KEEP,
        "none",
        *MODEL_KEEPS,
        KEEP,
        KEEP,
        KEEP,
        start,
    ]


@pytest.mark.parametrize(
    ("name", "table"),
    [("generate", RANGES_GENERATE_OPTIONS), ("adjust", RANGES_ADJUST_OPTIONS)],
)
def test_the_steps_set_every_option_but_json_and_settings(
    session: Session, name: str, table: Any
) -> None:
    flow = FLOWS[name]()
    sets = {option for step in flow.steps(session) for option in step.sets}
    assert flow.table is table
    assert sets == set(table.names) - {"json", "settings"}


def test_a_scripted_generate_session_writes_the_ranges(
    session: Session, text_file: Path
) -> None:
    flow = ranges_ui.generate_flow()
    answers = [KEEP, str(text_file), "60", KEEP, BACK, KEEP, KEEP, KEEP, "run"]

    outcome, state, prompter = scripted(flow, session, answers)

    assert outcome.action is Decision.RUN
    assert outcome.command == (
        "chronominer",
        "ranges",
        "generate",
        "--input",
        str(text_file),
        "--chunk-tokens",
        "60",
        *settings_words(session),
    )
    assert "  found 1 text file" in prompter.lines
    assert "Plan (ranges generate): 1 file(s)" in prompter.transcript
    spec = flow.resolve(state, session).spec
    assert parse_ranges_generate(outcome.command[3:], session.settings.defaults) == (
        spec
    )

    code, output = run(flow, session, outcome.answers)

    assert code == 0
    ranges = read_ranges(ranges_path(text_file))
    assert len(ranges) > 1
    assert (text_file.parent / DATABASE_NAME).is_file()
    assert "Run summary" in output


def test_a_scripted_adjust_session_adjusts_the_ranges(
    session: Session, text_with_ranges: Path, fake_llm: FakeLLM
) -> None:
    flow = ranges_ui.adjust_flow()
    write_ranges(text_with_ranges, MID_RECORD)

    outcome, state, prompter = scripted(flow, session, adjust_script(text_with_ranges))

    assert outcome.action is Decision.RUN
    assert outcome.command == (
        "chronominer",
        "ranges",
        "adjust",
        "--input",
        str(text_with_ranges),
        "--schema",
        SCHEMA,
        "--context",
        "none",
        *settings_words(session),
    )
    assert "  found 1 text file with line ranges" in prompter.lines
    assert "Plan (ranges adjust): 1 file(s)" in prompter.transcript
    spec = flow.resolve(state, session).spec
    assert parse_ranges_adjust(outcome.command[3:], session.settings.defaults) == spec

    code, output = run(flow, session, outcome.answers)

    assert code == 0, output
    assert fake_llm.calls
    assert (text_with_ranges.parent / DATABASE_NAME).is_file()
    starts = [start for start, _end in read_ranges(ranges_path(text_with_ranges))]
    assert starts == [1, 25, 43, 61]
    assert "Run summary" in output


class BackOnce(ScriptedPrompter):
    """Answer by message, defaulting to KEEP; go Back once at every select.

    The task step and the quit confirmation are answered without Back.
    """

    def __init__(self, answers: Mapping[str, Any], *, back: bool) -> None:
        super().__init__([])
        self.by_message = dict(answers)
        self.back = back
        self.asked: list[str] = []
        self.backed: set[str] = set()

    def _take(self, message: str) -> Any:
        return self.by_message.get(message, KEEP)

    def select(self, message: str, choices: Sequence[Any], **kwargs: Any) -> Any:
        self.asked.append(message)
        skip = message in ("Task:", "Quit without running?")
        if self.back and not skip and message not in self.backed:
            self.backed.add(message)
            self.lines.append(f"? {message}")
            return Nav.BACK
        return super().select(message, choices, **kwargs)


def by_message(name: str, folder: Path) -> dict[str, Any]:
    if name == "generate":
        return {
            "Text file or folder of text files:": str(folder),
            "Tokens per chunk:": "60",
            "Start:": "quit",
        }
    return {
        "Text file or folder whose line ranges to adjust:": str(folder),
        "Schema:": SCHEMA,
        "Stored decisions:": True,
        "Start:": "quit",
    }


@pytest.mark.parametrize("name", list(FLOWS))
def test_back_works_on_every_select(
    tmp_path: Path, session: Session, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    folder = tmp_path / "ledgers"
    text = make_text_file(folder / "ledger.txt")
    ranges_path(text).write_text("(1, 29)\n(30, 71)\n", encoding="utf-8")
    (folder / DATABASE_NAME).write_bytes(b"")
    answers = by_message(name, folder)
    flow = FLOWS[name]()

    straight = BackOnce(answers, back=False)
    plain, plain_state = session_of(flow, session, straight)
    backed = BackOnce(answers, back=True)
    outcome, state = session_of(flow, session, backed)

    selects = set(straight.asked) - {"Task:"}
    assert "Search the subfolders too?" in selects
    if name == "adjust":
        assert {"Schema:", "Provider:", "Response format:", "Stored decisions:"} <= (
            selects
        )
    assert backed.backed == selects
    assert outcome.command == plain.command
    assert state.answers == plain_state.answers


def ask_input(flow: TaskFlow, session: Session, answers: Sequence[Any]) -> Any:
    state = new_state(session, flow.table)
    prompter = ScriptedPrompter(answers)
    (step,) = [step for step in flow.steps(session) if step.name == "input"]
    with prompter:
        result = step.ask(StepContext(state, prompter))
    return result, prompter


def test_adjust_refuses_a_folder_without_line_ranges(
    session: Session, text_file: Path
) -> None:
    result, prompter = ask_input(
        ranges_ui.adjust_flow(),
        session,
        [str(text_file.parent), KEEP, BACK, BACK, BACK],
    )
    assert result is BACK
    assert "  found 0 text files with line ranges" in prompter.lines
    assert (
        "  no text file here has a _line_ranges.txt; run "
        "'chronominer ranges generate' first" in prompter.lines
    )


def test_generate_refuses_a_folder_without_text_files(
    tmp_path: Path, session: Session
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    result, prompter = ask_input(
        ranges_ui.generate_flow(), session, [str(empty), KEEP, BACK, BACK, BACK]
    )
    assert result is BACK
    assert (
        "  no text file here; choose another input or search the subfolders"
        in prompter.lines
    )


def test_the_generate_review_marks_settings_defaults_and_previews_the_plan(
    make_session: SessionMaker, text_file: Path
) -> None:
    session = make_session({"chunk_tokens": 500})
    flow = ranges_ui.generate_flow()
    answers = [KEEP, str(text_file), KEEP, KEEP, KEEP, "quit"]

    _outcome, state, prompter = scripted(flow, session, answers)

    review = flow.review(state, session)
    text = render_review(review)
    assert SETTINGS_LEGEND in text
    assert "500 tokens per chunk (s)" in text
    assert review.withheld is None
    assert any("Plan (ranges generate): 1 file(s)" in note for note in review.notes)
    assert "Equivalent command:" in prompter.transcript


def test_the_adjust_review_shows_the_condition_warnings(
    make_session: SessionMaker, text_with_ranges: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    session = make_session({"service_tier": "priority"})
    flow = ranges_ui.adjust_flow()
    state = new_state(session, flow.table)
    state.set("input", text_with_ranges)
    state.set("schema", SCHEMA)
    state.set("provider", "anthropic")
    state.set("model", "claude-sonnet-5-5")

    review = flow.review(state, session)

    text = render_review(review)
    assert review.withheld is None
    assert "Warning: the settings default service_tier (--service-tier)" in text
    assert "Adjust schema: PriceListEntries" in text


def test_run_is_withheld_when_the_adjust_cannot_be_planned(
    session: Session, text_with_ranges: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    flow = ranges_ui.adjust_flow()

    _outcome, state, prompter = scripted(
        flow, session, adjust_script(text_with_ranges, start="quit")
    )

    review = flow.review(state, session)
    assert review.withheld is not None and "OPENAI_API_KEY" in review.withheld
    assert "   Run" not in prompter.lines
    assert "Run unavailable:" in prompter.transcript


@pytest.mark.parametrize("name", list(FLOWS))
def test_a_plan_error_exits_2_without_startup(
    tmp_path: Path, session: Session, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    calls: list[CommandSpec] = []
    monkeypatch.setattr(ext, "startup", lambda spec, **_: calls.append(spec))
    answers = {"input": tmp_path / "missing.txt", "schema": SCHEMA}
    if name == "generate":
        del answers["schema"]
    code, output = run(FLOWS[name](), session, answers)
    assert code == 2
    assert "Error: Input path does not exist" in output
    assert calls == []


def test_a_dry_run_starts_the_extension_and_writes_nothing(
    session: Session, text_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[CommandSpec] = []
    monkeypatch.setattr(ext, "startup", lambda spec, **_: calls.append(spec))
    before = sorted(text_file.parent.iterdir())

    code, output = run(
        ranges_ui.generate_flow(), session, {"input": text_file, "dry_run": True}
    )

    assert code == 0
    assert len(calls) == 1
    assert "[DRY RUN] ranges generate: 1 text file(s) would be processed." in output
    assert sorted(text_file.parent.iterdir()) == before


def test_a_failed_adjust_exits_1(
    session: Session, text_with_ranges: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    async def failing(ranges_plan: RangesPlan, events: Any) -> RunResult:
        items = tuple(ItemResult(path.name, False) for path in ranges_plan.files)
        return RunResult(items=items, usage={"adjust": Usage(10, 2, 12)})

    monkeypatch.setattr(ranges_ui, "run_ranges", failing)
    code, output = run(
        ranges_ui.adjust_flow(),
        session,
        {"input": text_with_ranges, "schema": SCHEMA},
    )
    assert code == 1
    assert "Run summary" in output
