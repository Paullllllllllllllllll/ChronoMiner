"""Database size of a 200-chunk fake extraction run.

Run on Windows with (``<repo>`` is the repository root):

    uv --directory <repo> run pytest tests/core/test_store_size.py -q -s

A scripted chat model answers each chunk with a schema response of ten
entries, about 2,500 characters, a dense page of a wage book. Every attempt
is stored verbatim in ``responses``, every chunk as a unit and every entry as
a row of the ``entries`` view. The test prints the database size and the
bytes per chunk (compression is worth adding only if this measurement calls
for it) and bounds the size. Measured on 12.10.2026: 3,248,128 bytes,
16,241 bytes per chunk (raw responses 3,319 bytes per chunk).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

import chronominer.run as core
from chronominer.common.sqlite import read, read_rows
from chronominer.output.store import DATABASE_NAME
from tests.conftest import make_settings, make_spec
from tests.llm import helpers

CHUNKS = 200
ENTRY: dict[str, Any] = {
    "person": "Johann Georg Müller",
    "occupation": "Maurergeselle",
    "wage": 1.25,
    "period": "für die Woche vom 3. bis 8. Mai",
    "currency": "Gulden",
    "date": "1791-05-08",
    "employer": "Bauamt der Stadt Zürich",
    "source_page": "fol. 12r",
    "notes": None,
}
ANSWER: dict[str, Any] = {
    "contains_no_content_of_requested_type": False,
    "entries": [ENTRY] * 10,
}
MAX_BYTES_PER_CHUNK = 32_768
"""Upper bound per chunk: the unit, its entries, the stored response, indexes."""


def _text(path: Path, chunks: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{n}. Johann Georg Müller, Maurergeselle\n" for n in range(chunks)),
        encoding="utf-8",
    )
    path.with_name(f"{path.stem}_line_ranges.txt").write_text(
        "".join(f"({n}, {n})\n" for n in range(1, chunks + 1)), encoding="utf-8"
    )
    return path


def test_the_database_of_a_200_chunk_run_stays_small(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    script = helpers.install(monkeypatch, [ANSWER])
    out = tmp_path / "out"
    spec = make_spec(
        _text(tmp_path / "in" / "wages.txt", CHUNKS),
        output=out,
        chunking="line-ranges",
    )

    result = asyncio.run(
        core.run(core.plan(spec, make_settings(rate_limits=((100_000, 1),))))
    )

    db = out / DATABASE_NAME
    size = db.stat().st_size
    responses = read_rows(db, "responses")
    raw = sum(len(str(row["raw"]).encode("utf-8")) for row in responses)
    print(  # noqa: T201 - the measurement is the intended output
        f"\nstore size of a {CHUNKS}-chunk run: {size:,} bytes, "
        f"{size / CHUNKS:,.0f} bytes per chunk "
        f"(raw responses {raw / CHUNKS:,.0f} bytes per chunk)"
    )
    (entries,) = read(
        db,
        lambda connection: connection.execute(
            "SELECT count(*) FROM entries"
        ).fetchone(),
    )
    assert result.complete == 1
    assert len(script.requests) == CHUNKS
    assert len(responses) == CHUNKS
    chunks = [row for row in read_rows(db, "units") if row["kind"] == "chunk"]
    assert len(chunks) == CHUNKS
    assert entries == CHUNKS * len(ANSWER["entries"])
    assert size <= CHUNKS * MAX_BYTES_PER_CHUNK
