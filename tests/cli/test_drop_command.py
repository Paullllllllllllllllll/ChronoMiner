"""The drop subcommand: exit codes, the JSON line, the dry run and refusals.

Run on Windows with:

    uv --directory <repo> run pytest tests/cli/test_drop_command.py -q

CLI oracle: ``drop --dry-run`` leaves the directory listing and the database
bytes unchanged; ``drop`` deletes the selected document's rows, including
``entry_rows`` and ``document_meta``, and its rendered file, keeps a file
edited by hand and names it, and leaves the other document intact; a
document leased by a live run is refused with exit 1 while the others are
dropped; usage errors exit 2. ``--json`` writes exactly one line on every
exit.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from pathlib import Path
from typing import Any

import pytest

from chronominer import __main__ as entry
from chronominer.output.render import render_item
from chronominer.output.store import DATABASE_NAME, open_store
from tests.cli.conftest import json_lines
from tests.output.test_store import chunk
from tests.output.test_store import seed_document as seed

DROP_KEYS = {"dry_run", "command", "dropped", "refused", "deleted_files", "kept_files"}
NAMES = ("ledger", "prices")


def _tree(root: Path) -> dict[str, Path]:
    """Store and render the documents of NAMES; return their outputs."""
    outputs = {name: root / f"{name}_output.json" for name in NAMES}

    async def main() -> None:
        async with open_store(root) as store:
            await store.begin_run("extract")
            for name, output in outputs.items():
                lease = await seed(store, [chunk(0), chunk(1)], name=name)
                await render_item(store, name, (), output, lease)
            await store.end_run(0)

    asyncio.run(main())
    connection = sqlite3.connect(root / DATABASE_NAME)
    try:
        with connection:
            run_id = connection.execute("SELECT run_id FROM runs").fetchone()[0]
            for doc_id, name in connection.execute(
                "SELECT doc_id, name FROM documents"
            ):
                connection.execute(
                    "INSERT INTO document_meta (doc_id, metadata) VALUES (?, ?)",
                    (doc_id, "{}"),
                )
                connection.execute(
                    "INSERT INTO entry_rows (doc_id, idx, entry_index, run_id, tool, "
                    "model, provider, entry) VALUES (?, 0, 0, ?, 'chronominer', "
                    "'m', 'p', ?)",
                    (doc_id, run_id, f'{{"item": "{name}"}}'),
                )
    finally:
        connection.close()
    return outputs


def _rows(root: Path, name: str) -> dict[str, int]:
    """Count the rows keyed to document *name* in each table."""
    connection = sqlite3.connect(root / DATABASE_NAME)
    try:
        found = connection.execute(
            "SELECT doc_id FROM documents WHERE name = ?", (name,)
        ).fetchone()
        doc_id = -1 if found is None else found[0]
        return {
            table: connection.execute(
                f"SELECT count(*) FROM {table} WHERE doc_id = ?", (doc_id,)
            ).fetchone()[0]
            for table in (
                "documents",
                "units",
                "outputs",
                "entry_rows",
                "document_meta",
            )
        }
    finally:
        connection.close()


def _listing(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, Any, str]:
    code = entry.main(list(argv))
    captured = capsys.readouterr()
    assert len(captured.out.splitlines()) == 1, captured.out
    (payload,) = json_lines(captured.out)
    return code, payload, captured.err


def test_a_dry_run_changes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    outputs = _tree(tmp_path)
    before = _listing(tmp_path)

    code, payload, err = _run(
        capsys, "drop", "--input", str(tmp_path), "--select", "1", "--dry-run", "--json"
    )

    assert code == 0
    assert set(payload) == DROP_KEYS
    assert payload["dry_run"] is True and payload["command"] == "drop"
    assert payload["dropped"] == ["ledger"]
    assert payload["deleted_files"] == [str(outputs["ledger"])]
    assert "[DRY RUN]" in err and "would drop ledger" in err
    assert _listing(tmp_path) == before


def test_drop_deletes_one_document_and_keeps_an_edited_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    outputs = _tree(tmp_path)
    kept = outputs["prices"]
    kept.write_text('{"edited": "by hand"}\n', encoding="utf-8")
    database = tmp_path / DATABASE_NAME

    code, payload, err = _run(
        capsys, "drop", "--input", str(database), "--files", "prices.txt", "--json"
    )

    assert code == 0
    assert (payload["dry_run"], payload["dropped"]) == (False, ["prices"])
    assert payload["refused"] == [] and payload["deleted_files"] == []
    assert payload["kept_files"] == [
        {"path": str(kept), "reason": "it was changed outside the tool"}
    ]
    assert f"kept {kept}" in err and "as unstored" in err
    assert kept.read_text(encoding="utf-8") == '{"edited": "by hand"}\n'
    assert set(_rows(tmp_path, "prices").values()) == {0}
    assert _rows(tmp_path, "ledger") == {
        "documents": 1,
        "units": 2,
        "outputs": 1,
        "entry_rows": 1,
        "document_meta": 1,
    }
    assert outputs["ledger"].is_file()


def test_a_leased_document_is_refused_with_exit_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    outputs = _tree(tmp_path)
    connection = sqlite3.connect(tmp_path / DATABASE_NAME)
    try:
        with connection:
            connection.execute(
                "UPDATE documents SET lease_run_id = 'other', lease_until = ? "
                "WHERE name = 'ledger'",
                (time.time() + 3600,),
            )
    finally:
        connection.close()

    code, payload, err = _run(
        capsys, "drop", "--input", str(tmp_path), "--all", "--json"
    )

    assert code == 1
    assert payload["dropped"] == ["prices"]
    assert [item["name"] for item in payload["refused"]] == ["ledger"]
    assert "run other leases it" in payload["refused"][0]["reason"]
    assert "refused ledger" in err
    assert outputs["ledger"].is_file() and not outputs["prices"].exists()
    assert _rows(tmp_path, "ledger")["units"] == 2


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--select", "1", "--all"], "not allowed with argument"),
        ([], "exactly one of --select, --files and --all"),
        (["--select", "9"], "No stored document matches --select"),
        (["--files", "missing.txt"], "--files names no stored document"),
    ],
    ids=["two selections", "no selection", "no match", "unknown file"],
)
def test_selection_errors_exit_2_and_change_nothing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    message: str,
) -> None:
    _tree(tmp_path)
    before = _listing(tmp_path)

    code, payload, err = _run(capsys, "drop", "--input", str(tmp_path), *argv, "--json")

    assert code == 2
    assert payload["command"] == "drop" and payload["dropped"] == []
    assert message in err
    assert _listing(tmp_path) == before


@pytest.mark.parametrize("target", ["missing", "empty", "notes.txt"])
def test_an_input_without_a_store_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], target: str
) -> None:
    (tmp_path / "empty").mkdir()
    (tmp_path / "notes.txt").write_text("x\n", encoding="utf-8")

    code, payload, _err = _run(
        capsys, "drop", "--input", str(tmp_path / target), "--all", "--json"
    )

    assert code == 2
    assert payload["command"] == "drop"


def test_a_versionless_database_exits_2_as_not_a_store(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    connection = sqlite3.connect(tmp_path / DATABASE_NAME)
    connection.execute("CREATE TABLE documents (name TEXT PRIMARY KEY)")
    connection.commit()
    connection.close()
    before = _listing(tmp_path)

    code, payload, err = _run(
        capsys, "drop", "--input", str(tmp_path), "--all", "--json"
    )

    assert code == 2
    assert payload["command"] == "drop"
    assert "is not a chronominer store" in err and "migrate" not in err
    assert _listing(tmp_path) == before


def test_without_json_stdout_stays_empty(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _tree(tmp_path)

    assert entry.main(["drop", "--input", str(tmp_path), "--all", "--dry-run"]) == 0

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "DROP" in captured.err
