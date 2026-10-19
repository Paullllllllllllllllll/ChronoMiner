"""Ctrl+Break ends a run like Ctrl+C: the lease is released, the run exits 130.

The test starts ``tests/cli/interrupt_child.py`` in its own process group,
waits until the child has stored the first chunk of a three-page PDF and
sends it Ctrl+Break. The oracle: exit code 130 with the JSON line, no lease
left on the document, the run's end recorded, and a rerun right away that
extracts only the two missing chunks. Windows only.
"""

from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

from chronominer import __main__ as entry
from chronominer.common.sqlite import read
from chronominer.output.store import database_path
from tests.cli.conftest import MakePdf, json_lines
from tests.cli.interrupt_child import ANSWER
from tests.llm import helpers

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="Ctrl+Break is a Windows console event"
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CHILD = "tests/cli/interrupt_child.py"
TIMEOUT = 120.0


def _rows(db: Path, sql: str) -> list[tuple[Any, ...]]:
    def query(connection: sqlite3.Connection) -> list[tuple[Any, ...]]:
        return [tuple(row) for row in connection.execute(sql).fetchall()]

    return read(db, query)


def test_ctrl_break_releases_the_lease_and_a_rerun_resumes(
    tmp_path: Path,
    make_pdf: MakePdf,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    pdf = make_pdf(pages=3)
    out = tmp_path / "out"
    log = tmp_path / "child.err"
    with log.open("w", encoding="utf-8") as errors:
        process = subprocess.Popen(
            [sys.executable, CHILD, str(pdf), str(out), str(tmp_path)],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=errors,
            text=True,
            encoding="utf-8",
            env=dict(os.environ),
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,  # type: ignore[attr-defined]
        )
        try:
            assert process.stdout is not None
            first: list[str] = []
            reader = threading.Thread(
                target=lambda: first.append(process.stdout.readline().strip()),  # type: ignore[union-attr]
                daemon=True,
            )
            reader.start()
            reader.join(TIMEOUT)
            if first != ["READY"]:
                process.kill()
                process.communicate(timeout=TIMEOUT)
                err = log.read_text(encoding="utf-8")
                raise AssertionError(f"child did not get ready: {first!r}\n{err}")
            os.kill(process.pid, signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
            stdout, _ = process.communicate(timeout=TIMEOUT)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=TIMEOUT)
    stderr = log.read_text(encoding="utf-8")

    assert process.returncode == 130, stderr
    assert "Interrupted by the user." in stderr
    (summary,) = json_lines(stdout)
    assert summary["items_complete"] == 0
    db = database_path(out)
    assert _rows(db, "SELECT lease_run_id, lease_until FROM documents") == [
        (None, None)
    ]
    ((child_run, exit_code),) = _rows(db, "SELECT run_id, exit_code FROM runs")
    assert exit_code == 130

    script = helpers.install(monkeypatch, [ANSWER])
    argv = [
        "extract",
        "--schema",
        "WageRecords",
        "--input",
        str(pdf),
        "--output",
        str(out),
        "--concurrency",
        "1",
        "--json",
    ]
    assert entry.main(argv) == 0

    payload = json.loads(capsys.readouterr().out)
    assert (payload["items_complete"], payload["items_blocked"]) == (1, 0)
    assert len(script.requests) == 2
    units = _rows(db, "SELECT idx, status, run_id FROM units ORDER BY idx")
    assert [status for _idx, status, _run in units] == ["done"] * 3
    assert units[0][2] == child_run
    assert child_run not in {run for _idx, _status, run in units[1:]}
