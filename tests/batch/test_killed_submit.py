"""A submission killed right after the provider accepted its create.

Run on Windows with:

    uv --directory <repo> run pytest tests/batch/test_killed_submit.py -q

The test starts ``tests/batch/submit_child.py`` with its own interpreter and
the repository root as working directory. The child submits through the fake
backend, whose batches it writes to a JSON file, prints READY once the fake
provider has accepted the create and is killed with ``Popen.kill()`` before
the store records the batch id. The oracle: the store holds the submission
as ``submitting`` without a batch id, and the next ``batch check`` marks it
unresolved, finds the OpenAI batch by the submission id in its metadata,
adopts it and finalizes it.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

from chronominer.batch.check import check_submissions
from chronominer.batch.finalize import FINALIZED_OUTCOME
from chronominer.common.sqlite import read
from chronominer.output.store import database_path
from chronominer.settings import Settings
from tests.batch.conftest import World, load_batches

REPO_ROOT = Path(__file__).resolve().parents[2]
CHILD = "tests/batch/submit_child.py"
LEASE_TTL = 1.0
TIMEOUT = 120.0
OUTPUT = "doc_output.json"


def _state(db: Path) -> tuple[str, str | None, str, float]:
    """Return the submission's status, batch id and id, and the lease's end."""

    def query(connection: sqlite3.Connection) -> tuple[str, str | None, str, float]:
        submission_id, status, batch_id = connection.execute(
            "SELECT submission_id, status, batch_id FROM submissions"
        ).fetchone()
        (until,) = connection.execute("SELECT lease_until FROM documents").fetchone()
        return status, batch_id, submission_id, float(until or 0.0)

    return read(db, query)


def test_a_submission_killed_after_the_accept_is_adopted(
    world: World, tmp_path: Path
) -> None:
    state = tmp_path / "world.json"
    process = subprocess.Popen(
        [sys.executable, CHILD, str(tmp_path), str(state), str(LEASE_TTL)],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=dict(os.environ),
    )
    try:
        assert process.stdout is not None
        line = process.stdout.readline().strip()
        if line != "READY":
            process.kill()
            _out, err = process.communicate(timeout=TIMEOUT)
            raise AssertionError(f"child did not get ready: {line!r}\n{err}")
        process.kill()
    finally:
        process.communicate(timeout=TIMEOUT)
    assert process.returncode != 0

    db = database_path(tmp_path)
    status, batch_id, submission_id, until = _state(db)
    assert (status, batch_id) == ("submitting", None)
    world.batches.update(load_batches(state))
    (accepted,) = world.batches.values()
    assert accepted.submission_id == submission_id
    while time.time() <= until:
        time.sleep(0.05)

    result = asyncio.run(check_submissions(tmp_path, Settings()))

    (check,) = result.submissions
    assert check.outcome == FINALIZED_OUTCOME
    assert "adopted batch" in (check.detail or "")
    assert _state(db)[:2] == ("finalized", accepted.batch_id)
    assert world.count("openai", "submit") == 0
    assert (tmp_path / OUTPUT).is_file()
