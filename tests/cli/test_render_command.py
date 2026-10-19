"""The render subcommand: plans, exit codes and the JSON line.

Run on Windows with:

    uv --directory <repo> run pytest tests/cli/test_render_command.py -q

CLI oracle: ``render --dry-run`` leaves the directory listing and the
database bytes unchanged and prints its plan on stderr; ``--json`` writes
exactly one line on every exit (0, 2 and 130); ``render`` re-renders a stale
output and leaves a hand-edited file alone without ``--force``; a
versionless database makes ``render`` exit 2 as not a chronominer store.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

import chronominer.cli.store as store_cli
from chronominer import __main__ as entry
from chronominer.common.store import UnitResult
from chronominer.output.render import render_item
from chronominer.output.store import DATABASE_NAME, KIND_CHUNK, open_store
from tests.cli.conftest import json_lines
from tests.output.test_store import MODEL, NAME, PROVIDER, answer, chunk
from tests.output.test_store import seed_document as seed

RENDER_KEYS = {
    "dry_run",
    "command",
    "databases",
    "documents",
    "written",
    "unchanged",
    "kept",
    "stale",
    "skipped",
    "outputs",
}


def _tree(root: Path) -> Path:
    """Store document NAME with two chunks, render it; return its output."""
    output = root / f"{NAME}_output.json"

    async def main() -> None:
        async with open_store(root) as store:
            await store.begin_run("extract")
            lease = await seed(store, [chunk(0), chunk(1)])
            await render_item(store, NAME, (), output, lease)

    asyncio.run(main())
    return output


def _correct_first_chunk(root: Path) -> None:
    async def main() -> None:
        async with open_store(root) as store:
            await store.begin_run("extract")
            row = store.document(NAME)
            assert row is not None
            lease = await store.acquire(row.doc_id)
            assert lease is not None
            await store.write_unit(
                lease,
                UnitResult(
                    kind=KIND_CHUNK,
                    idx=0,
                    status="done",
                    model=MODEL,
                    provider=PROVIDER,
                    content=answer({"item": "corrected"}),
                ),
            )
            await store.release(lease)

    asyncio.run(main())
    connection = sqlite3.connect(root / DATABASE_NAME)
    try:
        connection.execute("UPDATE outputs SET stale = 1")
        connection.commit()
    finally:
        connection.close()


def _first_item(output: Path) -> str:
    data = json.loads(output.read_text(encoding="utf-8"))
    entries = json.loads(data["records"][0]["response"]["output_text"])["entries"]
    return str(entries[0]["item"])


def _listing(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, Any, str]:
    code = entry.main(list(argv))
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert len(lines) == 1, captured.out
    (payload,) = json_lines(captured.out)
    return code, payload, captured.err


def test_render_dry_run_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = _tree(tmp_path)
    output.unlink()
    before = _listing(tmp_path)

    code, payload, err = _run(
        capsys, "render", "--input", str(tmp_path), "--dry-run", "--json"
    )

    assert code == 0
    assert set(payload) == RENDER_KEYS
    assert (payload["dry_run"], payload["written"]) == (True, 1)
    assert "[DRY RUN]" in err and "would write" in err
    assert _listing(tmp_path) == before


def test_render_rerenders_a_stale_output_and_keeps_an_edited_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = _tree(tmp_path)
    _correct_first_chunk(tmp_path)

    code, payload, _err = _run(capsys, "render", "--input", str(tmp_path), "--json")

    assert code == 0
    assert payload["written"] == 1 and payload["outputs"] == [str(output)]
    assert _first_item(output) == "corrected"

    output.write_text('{"edited": "by hand"}\n', encoding="utf-8")
    code, payload, err = _run(capsys, "render", "--input", str(tmp_path), "--json")

    assert code == 0
    assert (payload["written"], payload["kept"]) == (0, 1)
    assert "--force" in err
    assert output.read_text(encoding="utf-8") == '{"edited": "by hand"}\n'

    code, payload, _err = _run(
        capsys, "render", "--input", str(tmp_path), "--force", "--json"
    )
    assert (code, payload["written"]) == (0, 1)
    assert _first_item(output) == "corrected"


def test_render_adds_the_csv_view(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = _tree(tmp_path)

    code, payload, _err = _run(
        capsys, "render", "--input", str(tmp_path), "--formats", "csv", "--json"
    )

    assert code == 0
    assert payload["outputs"] == [str(output.with_suffix(".csv"))]
    assert payload["unchanged"] == 1


def test_render_refuses_a_versionless_database_as_not_a_store(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    connection = sqlite3.connect(tmp_path / DATABASE_NAME)
    try:
        with connection:
            connection.execute("CREATE TABLE documents (name TEXT PRIMARY KEY)")
            connection.execute("CREATE TABLE pages (name TEXT, idx INTEGER)")
    finally:
        connection.close()
    before = _listing(tmp_path)

    code, payload, err = _run(capsys, "render", "--input", str(tmp_path), "--json")

    assert code == 2
    assert payload["written"] == 0
    assert "is not a chronominer store" in err
    assert _listing(tmp_path) == before


def test_render_of_a_file_that_is_no_database_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    other = tmp_path / "notes.txt"
    other.write_text("x\n", encoding="utf-8")

    code, payload, err = _run(capsys, "render", "--input", str(other), "--json")

    assert code == 2
    assert payload["command"] == "render"
    assert DATABASE_NAME in err


@pytest.mark.parametrize(
    "argv",
    [
        ["render", "--formats", "pdf", "--json"],
        ["render", "--bogus", "--json"],
    ],
    ids=["bad format", "unknown flag"],
)
def test_usage_errors_exit_2_with_one_line(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    code, payload, err = _run(capsys, *argv)
    assert code == 2
    assert payload["command"] == "render"
    assert "Invalid arguments" in err


def test_a_missing_input_path_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, payload, _err = _run(
        capsys, "render", "--input", str(tmp_path / "missing"), "--json"
    )
    assert code == 2
    assert payload["command"] == "render"


def test_an_interrupt_exits_130_with_one_line(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def interrupted(*args: Any, **kwargs: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(store_cli, "find_databases", interrupted)

    code, payload, _err = _run(capsys, "render", "--input", str(tmp_path), "--json")

    assert code == 130
    assert payload["command"] == "render"


def test_without_json_stdout_stays_empty(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _tree(tmp_path)

    assert entry.main(["render", "--input", str(tmp_path), "--dry-run"]) == 0

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "RENDER" in captured.err
