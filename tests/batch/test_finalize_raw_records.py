"""A malformed result line is stored without a unit and changes no unit.

Windows command::

    uv --directory <repo> run pytest tests/batch/test_finalize_raw_records.py -q

Oracle: finalizing a submission whose output file holds one malformed line
stores that line as a ``responses`` row with the submission id, ``raw``
equal to the line and ``doc_id``, ``kind``, ``idx`` and ``custom_id`` NULL;
the units come out exactly as from the same file without the line, the unit
without a record failing with the missing-result placeholder.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from chronominer.batch.check import check_submissions
from chronominer.batch.finalize import FINALIZED_OUTCOME, MISSING_TEXT
from chronominer.common.sqlite import read
from chronominer.output.store import database_path
from chronominer.settings import Settings
from tests.batch.test_finalize import submission_row, submit_store_pages

MALFORMED = '{"custom_id": "x-00002", "response": {"status_code": 2'
OUTPUT = "doc_output.json"


async def _finalize(
    root: Path, openai: Any, malformed: str | None
) -> tuple[str, list[tuple[Any, ...]], list[tuple[Any, ...]]]:
    """Finalize three page chunks whose second has no record; return the rows."""
    submission_id = await submit_store_pages(root, pages=3)
    batch_id = submission_row(root, submission_id)["batch_id"]
    openai.complete(batch_id)
    batch = openai._retrieve(batch_id)
    lines = openai.data[batch.output_file_id].decode("utf-8").splitlines()
    kept = [lines[0], lines[2]]
    if malformed is not None:
        kept.insert(1, malformed)
    openai.data[batch.output_file_id] = ("\n".join(kept) + "\n").encode("utf-8")

    result = await check_submissions(root, Settings())

    (check,) = result.submissions
    assert (check.outcome, check.errors) == (FINALIZED_OUTCOME, 1)
    units = read(
        database_path(root),
        lambda connection: connection.execute(
            "SELECT idx, status, content, error, response_id IS NOT NULL "
            "FROM units ORDER BY idx"
        ).fetchall(),
    )
    responses = read(
        database_path(root),
        lambda connection: connection.execute(
            "SELECT submission_id, custom_id, doc_id, kind, idx, raw, raw_kind, "
            "error FROM responses WHERE submission_id = ? ORDER BY response_id",
            (submission_id,),
        ).fetchall(),
    )
    return submission_id, units, responses


async def test_a_malformed_line_is_stored_without_a_unit(
    sdk: dict[str, Any], tmp_path: Path
) -> None:
    openai = sdk["openai"]
    submission_id, units, responses = await _finalize(
        tmp_path / "malformed", openai, MALFORMED
    )
    _, clean_units, clean_responses = await _finalize(tmp_path / "clean", openai, None)

    assert units == clean_units
    assert units[1][:4] == (1, "failed", None, MISSING_TEXT)
    assert len(responses) == len(clean_responses) + 1
    (stored,) = [row for row in responses if row[1] is None]
    assert stored[:7] == (submission_id, None, None, None, None, MALFORMED, "line")
    assert "line 2" in (stored[7] or "")
    assert [row[4] for row in responses if row[1] is not None] == [0, 2]
    assert _records(tmp_path / "malformed") == _records(tmp_path / "clean")


def _records(root: Path) -> Any:
    document = json.loads((root / OUTPUT).read_text(encoding="utf-8"))
    return document["records"], document["_chronominer_metadata"]["failed_chunks"]
