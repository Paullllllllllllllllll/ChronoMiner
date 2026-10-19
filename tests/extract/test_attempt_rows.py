"""Every provider attempt of a chunk is a response row with the body verbatim.

Run on Windows with:

    uv --directory <repo> run pytest tests/extract/test_attempt_rows.py -q

The unit pipeline runs the real ``ChunkExtractor`` (through ``ModelCaller``);
an ``httpx.MockTransport`` answers behind its httpx client
(``tests.llm.network``), so no request leaves the machine. When request *n*
arrives, the transport reads the store through a ``mode=ro`` connection and
expects the rows of the *n - 1* attempts before it.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from chronominer.common.capture import AttemptRecord
from chronominer.common.sqlite import read_rows
from chronominer.extract.chunks import extract_payload
from chronominer.extract.extraction import ChunkExtractor
from chronominer.extract.images import PageImage
from chronominer.extract.units import UnitsFailedError, run_unit_stream
from chronominer.output.attempts import AttemptRows
from chronominer.settings import RetrySettings
from chronominer.spec import resolve_schema
from tests.extract.conftest import (
    TOOL,
    make_store,
    open_handle,
    responses,
    stream,
    units,
)
from tests.llm import helpers
from tests.llm.conftest import Sleeps
from tests.llm.network import MockNetwork
from tests.llm.test_attempt_capture import (
    OPENAI_MALFORMED,
    OPENAI_REFUSAL,
    SERVER_ERROR,
    _openai_body,
)

ANSWER = json.dumps(
    {
        "contains_no_content_of_requested_type": False,
        "entries": [
            {
                "person": "Hans Müller",
                "occupation": "mason",
                "wage": 12.5,
                "currency": "shillings",
                "period": "week",
                "date": "1701",
                "employer": None,
                "source_page": None,
                "notes": None,
            }
        ],
    },
    ensure_ascii=False,
)
OPENAI_ENTRIES = _openai_body(
    json.dumps(
        {"type": "output_text", "text": ANSWER, "annotations": []},
        ensure_ascii=False,
    ).encode("utf-8")
)

PAGE = PageImage(
    index=0,
    image_name="page_0001",
    source_file="doc.pdf",
    page_index=0,
    base64="QUJD" * 600,
    mime_type="image/jpeg",
)


class StoredRows:
    """The response rows of the store, read through a ``mode=ro`` connection."""

    def __init__(self, db: Path) -> None:
        self.db = db

    @property
    def records(self) -> list[dict[str, Any]]:
        return responses(self.db) if self.db.exists() else []


@pytest.fixture
def db(tmp_path: Path) -> Path:
    return tmp_path / "chronominer.sqlite"


@pytest.fixture
def network(monkeypatch: pytest.MonkeyPatch, db: Path) -> MockNetwork:
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.setenv(name, f"test-{name}")
    view: Any = StoredRows(db)
    return MockNetwork(view).install(monkeypatch)


def _bodies(rows: list[dict[str, Any]]) -> list[bytes]:
    return [str(row["raw"]).encode("utf-8") for row in rows]


def _run(db: Path, extractor: Any, payload: Any) -> UnitsFailedError | None:
    async def main() -> UnitsFailedError | None:
        async with make_store(db) as store:
            handle = await open_handle(store, pages=1, model="gpt-5.6-luna")
            try:
                await run_unit_stream(
                    handle,
                    stream([payload]),
                    extractor,
                    extract_payload,
                    concurrency=1,
                )
            except UnitsFailedError as exc:
                return exc
            finally:
                await extractor.close()
        return None

    return asyncio.run(main())


def _page(db: Path, sleeps: Sleeps, validation_attempts: int = 1) -> Any:
    retry = dataclasses.replace(
        RetrySettings(), validation_attempts=validation_attempts
    )
    env = helpers.env(retry, clock=sleeps.clock)
    schema = resolve_schema("WageRecords")
    extractor = ChunkExtractor(
        helpers.model_spec(response_format="json"),
        env,
        schema=schema.schema,
        schema_name=schema.name,
    )
    return _run(db, extractor, PAGE)


def _check_rows(db: Path, network: MockNetwork, expected: list[bytes]) -> Any:
    """Check the bodies and stamps of the rows; return the unit row."""
    rows = responses(db)
    (unit,) = units(db)
    assert _bodies(rows) == expected
    assert [row["raw_kind"] for row in rows] == ["http"] * len(expected)
    assert [row["attempt"] for row in rows] == list(range(1, len(expected) + 1))
    for row in rows:
        assert (row["doc_id"], row["kind"], row["idx"]) == (
            unit["doc_id"],
            unit["kind"],
            0,
        )
        assert (row["tool"], row["run_id"]) == (TOOL, unit["run_id"])
    assert network.violations == []
    assert len(network.requests) == len(expected)
    assert network.scoped == [True] * len(expected)
    return unit, rows


class TestChunkExtractor:
    """The oracle cases through ``ModelCaller`` on the Responses API."""

    def test_a_success_is_one_row_and_the_units_response(
        self, db: Path, network: MockNetwork, sleeps: Sleeps
    ) -> None:
        network.answers = [(200, OPENAI_ENTRIES)]

        assert _page(db, sleeps) is None

        unit, rows = _check_rows(db, network, [OPENAI_ENTRIES])
        assert (rows[0]["provider"], rows[0]["model"]) == ("openai", "gpt-6-luna")
        assert (rows[0]["http_status"], rows[0]["error"]) == (200, None)
        assert (rows[0]["role"], rows[0]["kind"]) == ("extract", "chunk")
        assert "QUJD" not in rows[0]["request"]
        assert unit["status"] == "done"
        assert unit["content"] == ANSWER
        assert unit["outcome"] == "ok"
        (entry,) = read_rows(db, "entries", order_by=["entry_index"])
        assert (entry["person"], entry["wage"]) == ("Hans Müller", 12.5)
        assert (unit["response_id"], unit["attempts"]) == (rows[0]["response_id"], 1)

    def test_a_malformed_output_is_two_rows_and_the_second_is_selected(
        self, db: Path, network: MockNetwork, sleeps: Sleeps
    ) -> None:
        network.answers = [(200, OPENAI_MALFORMED), (200, OPENAI_ENTRIES)]

        assert _page(db, sleeps, validation_attempts=2) is None

        unit, rows = _check_rows(db, network, [OPENAI_MALFORMED, OPENAI_ENTRIES])
        assert [row["error_type"] for row in rows] == ["TypeError", None]
        assert (unit["response_id"], unit["attempts"]) == (rows[1]["response_id"], 2)
        assert unit["status"] == "done"

    def test_a_refusal_is_one_row_and_a_failed_unit(
        self, db: Path, network: MockNetwork, sleeps: Sleeps
    ) -> None:
        network.answers = [(200, OPENAI_REFUSAL)]

        failure = _page(db, sleeps)

        assert failure is not None and failure.failed_chunks == 1
        unit, rows = _check_rows(db, network, [OPENAI_REFUSAL])
        assert unit["status"] == "failed"
        assert unit["outcome"] == "failed"
        assert unit["content"] is None
        assert unit["error"]
        assert unit["response_id"] == rows[0]["response_id"]

    def test_a_500_then_a_200_is_two_rows_committed_before_the_retry(
        self, db: Path, network: MockNetwork, sleeps: Sleeps
    ) -> None:
        network.answers = [(500, SERVER_ERROR), (200, OPENAI_ENTRIES)]

        assert _page(db, sleeps) is None

        unit, rows = _check_rows(db, network, [SERVER_ERROR, OPENAI_ENTRIES])
        assert [row["http_status"] for row in rows] == [500, 200]
        assert rows[0]["error_type"] == "InternalServerError"
        assert unit["response_id"] == rows[1]["response_id"]
        assert unit["content"] == ANSWER


def test_the_recorder_selects_the_last_attempt_without_an_error(
    tmp_path: Path,
) -> None:
    """Without a successful attempt the last attempt is selected."""

    def record(attempt: int, error: str | None) -> AttemptRecord:
        return AttemptRecord("openai", "m", "extract", attempt, error=error)

    async def main() -> list[int | None]:
        async with make_store(tmp_path / "chronominer.sqlite") as store:
            await store.begin_run("extract")
            rows = AttemptRows(store)
            picked = [rows.selected()]
            await rows.record(record(1, "boom"))
            picked.append(rows.selected().response_id)  # type: ignore[union-attr]
            await rows.record(record(2, None))
            await rows.record(record(3, "late"))
            picked.append(rows.selected().response_id)  # type: ignore[union-attr]
            return picked

    assert asyncio.run(main()) == [None, 1, 2]
