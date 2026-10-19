"""The render and drop tasks of the wizard: steps, review, command, run.

Run on Windows with:

    uv --directory <repo> run pytest tests/ui/test_store_flows.py -q

Scripted sessions drive each flow through the wizard, with Back, to its
review; the outcome's answers then run the task as ``ui`` does. The render
run works on a store seeded with the helpers of ``tests/output``, the
versionless refusal on a database without schema version.
"""

from __future__ import annotations

import argparse
import io
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from chronominer.common.options import OptionTable
from chronominer.common.testing import KEEP, ScriptedPrompter, ScriptError
from chronominer.common.wizard import (
    BACK,
    BACK_TITLE,
    Decision,
    Outcome,
    State,
)
from chronominer.output.store import DATABASE_NAME
from chronominer.spec import (
    DROP_OPTIONS,
    RENDER_OPTIONS,
    parse_drop,
    parse_render,
)
from chronominer.ui.session import RunRequest, Session, TaskFlow, build_wizard
from chronominer.ui.store import drop_flow, render_flow
from tests.output.test_store import NAME, build_store, chunk


def _versionless(root: Path) -> Path:
    """Write a database without schema version in *root*; return *root*."""
    root.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(root / DATABASE_NAME)
    try:
        connection.execute("CREATE TABLE documents (name TEXT PRIMARY KEY)")
        connection.commit()
    finally:
        connection.close()
    return root


def _seeded_store(root: Path) -> Path:
    """Seed one document with two done chunks; return its _output.json path."""
    build_store(root, [chunk(0), chunk(1)])
    return root / f"{NAME}_output.json"


def _items(path: Path) -> list[Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        json.loads(record["response"]["output_text"])["entries"][0]["item"]
        for record in data["records"]
    ]


def _listing(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _session_run(
    flow: TaskFlow, session: Session, answers: list[Any]
) -> tuple[Outcome, ScriptedPrompter]:
    prompter = ScriptedPrompter(answers)
    with prompter:
        outcome = build_wizard(flow, session, prompter, width=120).run()
    return outcome, prompter


def _run(flow: TaskFlow, session: Session, outcome: Outcome) -> tuple[int, str]:
    file = io.StringIO()
    console = Console(file=file, force_terminal=False, width=120, color_system=None)
    request = RunRequest(session, outcome.answers, console, argparse.Namespace())
    return flow.run(request), file.getvalue()


def _start_choices(prompter: ScriptedPrompter) -> list[str]:
    """Return the entries of the last Start menu, without Back."""
    lines = prompter.lines
    start = max(index for index, line in enumerate(lines) if line == "? Start:")
    entries: list[str] = []
    for line in lines[start + 1 :]:
        entry = line[3:]
        if entry == BACK_TITLE:
            return entries
        entries.append(entry)
    raise AssertionError("the Start menu has no Back entry")


def _sets(flow: TaskFlow, session: Session) -> set[str]:
    return {name for step in flow.steps(session) for name in step.sets}


@pytest.mark.parametrize(
    ("flow", "table"),
    [
        (render_flow(), RENDER_OPTIONS),
        (drop_flow(), DROP_OPTIONS),
    ],
    ids=["render", "drop"],
)
def test_every_option_but_json_and_settings_is_set_by_a_step(
    flow: TaskFlow, table: OptionTable, session: Session
) -> None:
    assert _sets(flow, session) == set(table.names) - {"json", "settings"}


def test_a_render_session_with_back_writes_the_chosen_views(
    tmp_path: Path, session: Session
) -> None:
    root = tmp_path / "out"
    output = _seeded_store(root)
    flow = render_flow()

    outcome, prompter = _session_run(
        flow,
        session,
        [
            KEEP,
            str(tmp_path),
            BACK,
            f'"{root}"',
            "choose",
            BACK,
            "choose",
            ["csv"],
            KEEP,
            KEEP,
            "run",
        ],
    )

    assert outcome.action is Decision.RUN
    assert outcome.answers == {"input": root, "formats": ("csv",)}
    assert outcome.command is not None
    assert outcome.command[:2] == ("chronominer", "render")
    assert "--settings" not in outcome.command
    spec = parse_render(list(outcome.command[2:]))
    answered = State(RENDER_OPTIONS, {}, outcome.answers)
    assert spec == flow.resolve(answered, session).spec
    assert (spec.input, spec.formats) == (root, ("csv",))
    assert f"{root}" in prompter.transcript
    assert "1 database, 1 document" in prompter.transcript
    assert "Run" in _start_choices(prompter)

    code, shown = _run(flow, session, outcome)

    assert code == 0
    assert _items(output) == ["item 0", "item 1"]
    assert output.with_suffix(".csv").is_file()
    assert "wrote 2 file(s)" in shown


def test_a_render_dry_run_writes_nothing(tmp_path: Path, session: Session) -> None:
    root = tmp_path / "out"
    _seeded_store(root)
    before = _listing(root)
    flow = render_flow()

    outcome, _prompter = _session_run(
        flow, session, [KEEP, str(root), KEEP, True, True, "run"]
    )
    code, shown = _run(flow, session, outcome)

    assert code == 0
    assert outcome.answers == {"input": root, "force": True, "dry_run": True}
    assert "[DRY RUN]" in shown and "would write 1 file(s)" in shown
    assert _listing(root) == before


def test_a_versionless_database_withholds_run_as_not_a_store(
    tmp_path: Path, session: Session
) -> None:
    out = _versionless(tmp_path / "out")
    before = _listing(out)

    outcome, prompter = _session_run(
        render_flow(), session, [KEEP, str(out), KEEP, KEEP, KEEP, "quit"]
    )

    assert outcome.action is Decision.QUIT
    assert "Run" not in _start_choices(prompter)
    transcript = prompter.transcript
    assert "Run unavailable:" in transcript
    assert "is not a chronominer store" in transcript
    assert "migrate" not in transcript
    assert _listing(out) == before


def test_render_withholds_run_for_a_missing_input_or_another_file(
    tmp_path: Path, session: Session
) -> None:
    other = tmp_path / "notes.sqlite"
    other.write_bytes(b"")
    for given, reason in (
        (tmp_path / "missing", "does not exist"),
        (other, f"is not a {DATABASE_NAME} file"),
    ):
        _outcome, prompter = _session_run(
            render_flow(), session, [KEEP, str(given), KEEP, KEEP, KEEP, "quit"]
        )
        assert "Run" not in _start_choices(prompter)
        assert reason in prompter.transcript


def test_a_drop_session_drops_the_searched_document(
    tmp_path: Path, session: Session
) -> None:
    root = tmp_path / "out"
    for name in ("ledger", "prices"):
        build_store(root, [chunk(0)], name=name)
    flow = drop_flow()

    outcome, prompter = _session_run(
        flow, session, [KEEP, str(root), "list", BACK, "search", "pri", KEEP, "run"]
    )

    assert outcome.action is Decision.RUN
    assert outcome.answers == {"input": root, "select": "pri"}
    assert outcome.command is not None
    assert parse_drop(list(outcome.command[2:])) == parse_drop(
        ["--input", str(root), "--select", "pri"]
    )
    assert "2 documents stored" in prompter.transcript
    assert "Would drop 1 document" in prompter.transcript

    code, shown = _run(flow, session, outcome)

    assert code == 0
    assert "dropped 1 document(s)" in shown
    connection = sqlite3.connect(root / DATABASE_NAME)
    try:
        names = [row[0] for row in connection.execute("SELECT name FROM documents")]
    finally:
        connection.close()
    assert names == ["ledger"]


def test_drop_by_number_runs_on_the_reviewed_names(
    tmp_path: Path, session: Session
) -> None:
    root = tmp_path / "out"
    for name in ("ledger", "prices"):
        build_store(root, [chunk(0)], name=name)

    outcome, _prompter = _session_run(
        drop_flow(), session, [KEEP, str(root), "numbers", "2", KEEP, "copy"]
    )

    assert outcome.answers == {"input": root, "files": ("prices",)}
    assert outcome.command is not None
    assert outcome.command[-2:] == ("--files", "prices")


def test_drop_asks_for_a_store(tmp_path: Path, session: Session) -> None:
    prompter = ScriptedPrompter([KEEP, str(tmp_path)])
    wizard = build_wizard(drop_flow(), session, prompter)
    with pytest.raises(ScriptError, match=f"No {DATABASE_NAME} in"):
        wizard.run()
