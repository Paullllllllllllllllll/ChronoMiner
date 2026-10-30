"""Fixtures of the classify tests: a fake decision API and seeded stores."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest

from chronominer import __main__ as entry
from chronominer.common.store import PlannedUnit, UnitResult, file_sha256
from chronominer.llm import decision_model
from chronominer.output.entries import entries_writer
from chronominer.output.store import database_path, metadata_writer, open_store
from chronominer.spec import resolve_schema
from tests.cli.conftest import json_lines

SCHEMA = resolve_schema("price_list_entries")
MODEL = "gpt-5.6-luna"
PROVIDER = "openai"
LINES = "".join(f"Entry {n}: bread, 2 lb, 4 d\n" for n in range(1, 41))
"""Forty lines of a price list."""


@pytest.fixture(autouse=True)
def _key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")


@dataclass
class DecisionServer:
    """The OpenRouter decision API: answers every question, records requests.

    A predicate gets *probability*, a choice its first value, a score its
    lowest level. With *fail* every request gets a 400.
    """

    probability: float = 0.8
    fail: bool = False
    requests: list[dict[str, Any]] = field(default_factory=list)
    on_request: Callable[[], None] | None = None
    """Runs while a request is in flight, before it is answered."""

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        if self.on_request is not None:
            self.on_request()
        if self.fail:
            return httpx.Response(400, json={"error": {"message": "bad request"}})
        answers = {name: self._answer(q) for name, q in body["questions"].items()}
        usage = {"input_tokens": 10, "output_tokens": 2}
        return httpx.Response(
            200, json={"answers": answers, "model": body["model"], "usage": usage}
        )

    def _answer(self, question: dict[str, Any]) -> dict[str, Any]:
        if question["type"] == "noul":
            return {"type": "noul", "noul": self.probability}
        if question["type"] == "choice":
            first = next(iter(question["criteria"]))
            return {"type": "choice", "choice": first, "probabilities": {first: 0.9}}
        return {"type": "score", "score": 0, "probabilities": {"0": 1.0}}

    def parts(self, number: int = -1) -> list[dict[str, Any]]:
        """Return the state parts of request *number* (the last by default)."""
        state: list[dict[str, Any]] = self.requests[number]["state"]
        return state


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> DecisionServer:
    """Route every decision request of the test to a :class:`DecisionServer`."""
    fake = DecisionServer()
    real = httpx.AsyncClient

    def client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        return real(transport=httpx.MockTransport(fake.handler))

    monkeypatch.setattr(decision_model.httpx, "AsyncClient", client)
    return fake


Classify = Callable[..., tuple[int, dict[str, Any]]]


@pytest.fixture
def classify(capsys: pytest.CaptureFixture[str]) -> Classify:
    """Return a runner of ``chronominer classify ... --json``: exit code, summary."""

    def run(*argv: str | Path) -> tuple[int, dict[str, Any]]:
        code = entry.main(["classify", *(str(arg) for arg in argv), "--json"])
        lines = json_lines(capsys.readouterr().out)
        assert len(lines) == 1
        return code, lines[0]

    return run


def write_text(path: Path, text: str = LINES) -> Path:
    """Write a text input."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def answer(*entries: dict[str, Any]) -> str:
    """Return a model answer holding *entries*."""
    return json.dumps(
        {"contains_no_content_of_requested_type": False, "entries": list(entries)}
    )


@dataclass(frozen=True)
class Chunk:
    """One chunk to seed: its line range, status and entries."""

    start: int
    end: int
    status: str = "done"
    entries: tuple[dict[str, Any], ...] = ()


def seed(
    root: Path,
    text: Path,
    chunks: Sequence[Chunk],
    *,
    name: str | None = None,
    input_sha256: str | None = None,
) -> int:
    """Store document *name* of *text* in *root* as extract would; return its id.

    Done chunks get their entries; the run metadata names the schema.
    """
    document = name or text.stem
    sha = input_sha256 or file_sha256(text)

    async def main() -> int:
        async with open_store(root) as store:
            await store.begin_run("extract")
            doc_id = await store.ensure_document(
                document, input_path=text, input_sha256=sha, kind="text"
            )
            lease = await store.acquire(doc_id)
            assert lease is not None
            planned = [
                PlannedUnit(idx, f"{c.start}-{c.end}", f"{c.start}-{c.end}")
                for idx, c in enumerate(chunks)
            ]
            await store.plan_units(
                lease, "chunk", planned, model=MODEL, provider=PROVIDER
            )
            for idx, c in enumerate(chunks):
                if c.status != "pending":
                    await write_chunk(store, lease, idx, c)
            await store.write(
                lease,
                metadata_writer(store, doc_id, {"schema_name": SCHEMA.name}),
            )
            await store.release(lease)
            await store.end_run(0)
            return doc_id

    return asyncio.run(main())


async def write_chunk(store: Any, lease: Any, idx: int, c: Chunk) -> None:
    """Write chunk *idx* with its entries, as an extraction or repair does."""
    done = c.status == "done"
    result = UnitResult(
        kind="chunk",
        idx=idx,
        status=c.status,
        model=MODEL,
        provider=PROVIDER,
        content=answer(*c.entries) if done else None,
        source_index=f"{c.start}-{c.end}",
        error=None if done else "Error code: 400",
    )
    extra = entries_writer(
        store,
        lease.doc_id,
        idx,
        list(c.entries),
        model=MODEL,
        provider=PROVIDER,
        schema=SCHEMA.schema,
    )
    await store.write_unit(lease, result, extra=extra if done else None)


def rows(root: Path, sql: str, *params: Any) -> list[dict[str, Any]]:
    """Return the rows of *sql* on *root*'s database as dictionaries."""
    connection = sqlite3.connect(database_path(root))
    connection.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in connection.execute(sql, params)]
    finally:
        connection.close()


def view(root: Path) -> list[dict[str, Any]]:
    """Return the ``classifications`` view of *root*'s store."""
    return rows(root, "SELECT * FROM classifications")


def classifier_file(folder: Path, body: dict[str, Any]) -> Path:
    """Write a classifier file into *folder*; return its path."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{body['name']}.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    return path
