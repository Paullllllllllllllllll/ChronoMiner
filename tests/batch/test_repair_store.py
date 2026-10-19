"""Repair on the store: synchronous and batch repairs of failed chunk units.

Run on Windows with:

    uv --directory <repo> run pytest tests/batch/test_repair_store.py -q

Each test builds an output folder whose ``chronominer.sqlite`` holds one
document of chunk units and its rendered ``doc_output.json``, as an
extraction run leaves them, then repairs the failed units. The fake
extractor records one attempt per call through the store's recorder, as the
model caller does; the batch tests run the fake batch backend of
``tests.batch.conftest``. No request leaves the machine.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import fitz
import pytest

from chronominer.batch import repair as repair_module
from chronominer.common.batch import BatchHandle, BatchResultItem
from chronominer.common.capture import AttemptRecord, record_attempt
from chronominer.common.sqlite import read
from chronominer.common.store import PlannedUnit, ResponseRow, UnitKey, UnitResult
from chronominer.extract.images import resolve_image_settings
from chronominer.extract.pdf import PDF_PAGE_NAME
from chronominer.output.entries import entries_writer, parse_entries
from chronominer.output.render import render_item
from chronominer.output.store import (
    database_path,
    image_settings_record,
    metadata_writer,
    open_store,
    output_path,
)
from chronominer.settings import Settings
from chronominer.spec import RepairSpec, resolve_repair, resolve_schema
from tests.batch.conftest import DEFAULT_MODELS, FakeBackend, World, answer
from tests.conftest import DEFAULT_TEST_SCHEMA

PROVIDER = "openai"
MODEL = DEFAULT_MODELS[PROVIDER]
SCHEMA = resolve_schema(DEFAULT_TEST_SCHEMA)
LINES = 3


@dataclass(frozen=True)
class Stored:
    """An extracted document: its output root, source, rendered file and id."""

    root: Path
    source: Path
    output: Path
    doc_id: int

    @property
    def db(self) -> Path:
        return database_path(self.root)


def page_name(idx: int) -> str:
    return PDF_PAGE_NAME.format(number=idx + 1)


def chunk_label(idx: int) -> str:
    return f"{idx * LINES + 1}-{(idx + 1) * LINES}"


def make_pdf(path: Path, pages: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with fitz.open() as doc:
        for number in range(pages):
            doc.new_page(width=200, height=100).insert_text((20, 50), f"P{number}")
        doc.save(path)
    return path


def make_text(path: Path, chunks: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"Paid to worker {idx} on line {n}"
        for idx in range(chunks)
        for n in range(LINES)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def image_record(provider: str, model: str) -> dict[str, Any]:
    config, model_type, _section = resolve_image_settings(provider, model, None)
    return image_settings_record(config, model_type)


def _run_spec(context: Path | str | None, context_image: Path | None) -> dict[str, Any]:
    def entry(value: Path | str | None) -> dict[str, Any]:
        return {"value": None if value is None else str(value), "source": "flag"}

    return {
        "prompt.context": entry(context),
        "prompt.context_image": entry(context_image)
        if context_image
        else {
            "value": "none",
            "source": "default",
        },
    }


async def store_document(
    tmp_path: Path,
    outcomes: Sequence[str],
    *,
    provider: str = PROVIDER,
    model: str | None = None,
    record: Mapping[str, Any] | None | bool = True,
    formats: Sequence[str] = ("json",),
    folder: str = "out",
    kind: str = "pdf",
    context: Path | str | None = None,
    context_image: Path | None = None,
) -> Stored:
    """Store document ``doc`` with one chunk unit per entry of *outcomes*.

    An outcome is ``ok`` or ``failed``. Each unit gets one earlier response
    row; the document's files are rendered. *kind* ``pdf`` makes a PDF of
    one page per chunk, ``text`` a text of three lines per chunk. *record*
    True stores the image settings of *model*, a mapping stores that record
    and None stores none. *context* and *context_image* are the extraction
    run's options.
    """
    model = model or DEFAULT_MODELS[provider]
    if kind == "pdf":
        source = make_pdf(tmp_path / "in" / "doc.pdf", len(outcomes))
    else:
        source = make_text(tmp_path / "in" / "doc.txt", len(outcomes))
    root = tmp_path / folder
    root.mkdir(parents=True, exist_ok=True)
    output = output_path(root, "doc")
    async with open_store(root) as store:
        await store.begin_run("extract", spec=_run_spec(context, context_image))
        doc_id = await store.ensure_document("doc", input_path=source, kind=kind)
        lease = await store.acquire(doc_id)
        assert lease is not None
        sha = await store.store_artifact(
            "schema", SCHEMA.path.read_text(encoding="utf-8")
        )
        await store.write(
            lease,
            metadata_writer(
                store, doc_id, {"schema_name": SCHEMA.name}, schema_sha256=sha
            ),
        )
        if kind == "pdf" and record is True:
            await store.set_image_settings(lease, image_record(provider, model))
        elif isinstance(record, Mapping):
            await store.set_image_settings(lease, record)
        labels = [
            page_name(idx) if kind == "pdf" else chunk_label(idx)
            for idx in range(len(outcomes))
        ]
        await store.plan_units(
            lease,
            "chunk",
            [
                PlannedUnit(idx, labels[idx], idx if kind == "pdf" else labels[idx])
                for idx in range(len(outcomes))
            ],
            model=model,
            provider=provider,
        )
        for idx, outcome in enumerate(outcomes):
            content = answer(f"Chunk {idx}") if outcome == "ok" else None
            response_id = await store.append_response(
                ResponseRow(
                    provider=provider,
                    model=model,
                    unit=UnitKey(doc_id, "chunk", idx),
                    role="extract",
                    attempt=1,
                    raw=json.dumps({"first": idx}),
                    raw_kind="http",
                )
            )
            await store.write_unit(
                lease,
                UnitResult(
                    kind="chunk",
                    idx=idx,
                    status="failed" if outcome == "failed" else "done",
                    model=model,
                    provider=provider,
                    content=content,
                    outcome=outcome,
                    label=labels[idx],
                    source_index=idx if kind == "pdf" else labels[idx],
                    error="boom" if outcome == "failed" else None,
                    provenance={"custom_id": f"doc-chunk-{idx + 1}"},
                    response_id=response_id,
                    attempts=1,
                ),
                extra=entries_writer(
                    store,
                    doc_id,
                    idx,
                    parse_entries(content),
                    model=model,
                    provider=provider,
                    schema=SCHEMA.schema,
                ),
            )
        await render_item(store, doc_id, formats, output, lease)
    return Stored(root, source, output, doc_id)


def spec_for(path: Path, *flags: str, **values: Any) -> RepairSpec:
    given: dict[str, Any] = {"input": path, **values}
    given.setdefault("provider", PROVIDER)
    if "model" not in given:
        given["model"] = DEFAULT_MODELS[given["provider"]]
    for flag in flags:
        given[flag] = True
    return resolve_repair(given).spec


def rows(db: Path, sql: str, params: Sequence[Any] = ()) -> list[tuple[Any, ...]]:
    def query(connection: sqlite3.Connection) -> list[tuple[Any, ...]]:
        return [tuple(row) for row in connection.execute(sql, params).fetchall()]

    return read(db, query)


def unit_rows(db: Path) -> list[tuple[Any, ...]]:
    return rows(
        db,
        "SELECT idx, status, outcome, content, response_id FROM units ORDER BY idx",
    )


def people(db: Path) -> list[tuple[int, str]]:
    return rows(db, "SELECT chunk_index, person FROM entries ORDER BY chunk_index")


def tree(root: Path) -> list[str]:
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))


def assert_no_side_files(root: Path) -> None:
    names = tree(root)
    assert not [name for name in names if name.endswith(".jsonl")]
    assert not [name for name in names if "debug" in name or "manifest" in name]


class FakeExtractor:
    """Answers each chunk by its label from *texts*; None fails the chunk.

    Every call records one attempt through the installed recorder, as the
    model caller does, and is kept in *calls* with its request options.
    """

    texts: dict[str, str | None] = {}
    calls: list[tuple[str, dict[str, Any]]] = []
    created: list[dict[str, Any]] = []
    context_images: list[Path | None] = []

    def __init__(self, model: Any, env: Any, **kwargs: Any) -> None:
        self.model = model
        FakeExtractor.created.append(kwargs)

    def update_context_image(self, path: Path | None) -> None:
        FakeExtractor.context_images.append(path)

    async def _answer(self, name: str, options: dict[str, Any]) -> dict[str, Any]:
        FakeExtractor.calls.append((name, options))
        text = self.texts.get(name)
        await record_attempt(
            AttemptRecord(
                provider=self.model.provider,
                model=self.model.model,
                role="extract",
                attempt=1,
                label=name,
                raw=json.dumps({"repair": name, "text": text}),
                raw_kind="http",
                error=None if text is not None else "boom",
            )
        )
        if text is None:
            return {"output_text": "", "error": "boom"}
        return {"output_text": answer(text)}

    async def extract_text(self, text: str, *, label: str | None = None) -> Any:
        return await self._answer(label or "", {"text": text})

    async def extract_image_from_base64(
        self, data: str, mime: str, *, label: str | None = None, **options: Any
    ) -> Any:
        return await self._answer(label or "", options)

    async def close(self) -> None:
        return None


@pytest.fixture
def extractor(monkeypatch: pytest.MonkeyPatch) -> type[FakeExtractor]:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(repair_module, "ChunkExtractor", FakeExtractor)
    FakeExtractor.texts = {}
    FakeExtractor.calls = []
    FakeExtractor.created = []
    FakeExtractor.context_images = []
    return FakeExtractor


async def test_a_sync_repair_rewrites_the_failed_units_and_their_file(
    tmp_path: Path, extractor: type[FakeExtractor]
) -> None:
    stored = await store_document(tmp_path, ["ok", "failed", "ok", "failed", "ok"])
    before = rows(stored.db, "SELECT response_id, raw FROM responses")
    extractor.texts = {page_name(1): "Fixed one", page_name(3): "Fixed three"}

    result = await repair_module.repair_extraction(spec_for(stored.output), Settings())

    assert (result.repaired, result.failed, result.selected) == (2, 0, 2)
    assert result.outputs == (stored.output,)
    responses = rows(
        stored.db, "SELECT response_id, raw, idx FROM responses ORDER BY response_id"
    )
    assert len(responses) == 7
    assert [row[:2] for row in responses[:5]] == before
    assert [row[2] for row in responses[5:]] == [1, 3]
    units = unit_rows(stored.db)
    assert [unit[:4] for unit in units] == [
        (0, "done", "ok", answer("Chunk 0")),
        (1, "done", "ok", answer("Fixed one")),
        (2, "done", "ok", answer("Chunk 2")),
        (3, "done", "ok", answer("Fixed three")),
        (4, "done", "ok", answer("Chunk 4")),
    ]
    assert {units[1][4], units[3][4]} == {row[0] for row in responses[5:]}
    assert people(stored.db) == [
        (1, "Chunk 0"),
        (2, "Fixed one"),
        (3, "Chunk 2"),
        (4, "Fixed three"),
        (5, "Chunk 4"),
    ]
    document = json.loads(stored.output.read_text(encoding="utf-8"))
    assert [record["chunk_index"] for record in document["records"]] == [1, 2, 3, 4, 5]
    assert "failed_chunks" not in document["_chronominer_metadata"]
    (run,) = rows(stored.db, "SELECT exit_code FROM runs WHERE command = 'repair'")
    assert run == (0,)
    assert_no_side_files(tmp_path)


async def test_a_unit_that_fails_again_stays_failed(
    tmp_path: Path, extractor: type[FakeExtractor]
) -> None:
    stored = await store_document(tmp_path, ["ok", "failed", "failed"])
    extractor.texts = {page_name(1): "Fixed one", page_name(2): None}

    result = await repair_module.repair_extraction(spec_for(stored.output), Settings())

    assert (result.repaired, result.failed, result.selected) == (1, 1, 2)
    status = [unit[1] for unit in unit_rows(stored.db)]
    assert status == ["done", "done", "failed"]
    document = json.loads(stored.output.read_text(encoding="utf-8"))
    assert document["_chronominer_metadata"]["failed_chunks"] == [3]
    (run,) = rows(stored.db, "SELECT exit_code FROM runs WHERE command = 'repair'")
    assert run == (1,)


async def test_the_recorded_detail_and_prompt_inputs_reach_the_extractor(
    tmp_path: Path, extractor: type[FakeExtractor]
) -> None:
    record = {**image_record(PROVIDER, MODEL), "resolved_detail": "low"}
    context = tmp_path / "context.txt"
    context.write_text("Wage book of a Basel mason.", encoding="utf-8")
    stored = await store_document(tmp_path, ["failed"], record=record, context=context)
    extractor.texts = {page_name(0): "Fixed"}

    await repair_module.repair_extraction(spec_for(stored.output), Settings())

    assert extractor.calls == [(page_name(0), {"image_detail": "low"})]
    (created,) = extractor.created
    assert created["role"] == "extract"
    assert created["schema_name"] == SCHEMA.name
    assert created["schema"] == SCHEMA.schema
    assert created["context"] == "Wage book of a Basel mason."
    assert extractor.context_images == [None]


async def test_a_text_chunk_is_cut_again_from_its_line_range(
    tmp_path: Path, extractor: type[FakeExtractor]
) -> None:
    stored = await store_document(tmp_path, ["ok", "failed", "ok"], kind="text")
    extractor.texts = {chunk_label(1): "Fixed text"}

    result = await repair_module.repair_extraction(spec_for(stored.output), Settings())

    assert (result.repaired, result.failed) == (1, 0)
    ((label, options),) = extractor.calls
    assert label == "4-6"
    assert options["text"] == "\n".join(
        f"Paid to worker 1 on line {n}" for n in range(LINES)
    )
    (provenance,) = rows(stored.db, "SELECT provenance FROM units WHERE idx = 1")
    assert json.loads(provenance[0])["custom_id"] == "doc-chunk-2"


async def test_the_input_path_finds_the_document_too(
    tmp_path: Path, extractor: type[FakeExtractor]
) -> None:
    stored = await store_document(tmp_path, ["failed", "ok"], folder="in")
    plan = repair_module.plan_repair(spec_for(stored.source), Settings())
    assert plan.document is not None
    assert plan.document.doc_id == stored.doc_id
    assert plan.document.output == stored.output
    assert plan.document.failures == [0]


def store_raw_records(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give every fake batch result a downloaded record, as the backends do."""
    real = FakeBackend._fetch_results

    def fetch(
        self: FakeBackend, handle: BatchHandle
    ) -> tuple[list[BatchResultItem], str | None]:
        items, error = real(self, handle)
        return [
            replace(
                item,
                raw=json.dumps({"custom_id": item.custom_id, "text": item.text}),
                raw_kind="line",
            )
            for item in items
        ], error

    monkeypatch.setattr(FakeBackend, "_fetch_results", fetch)


@pytest.mark.parametrize("provider", ["openai", "anthropic", "google"])
async def test_a_batch_repair_submits_finalizes_and_renders_the_units(
    world: World,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
) -> None:
    world.file_uploads = True
    store_raw_records(monkeypatch)
    stored = await store_document(
        tmp_path, ["ok", "failed", "ok", "failed", "ok"], provider=provider
    )

    result = await repair_module.repair_extraction(
        spec_for(stored.output, "batch", provider=provider), Settings()
    )

    assert (result.repaired, result.failed, result.selected) == (2, 0, 2)
    (submission,) = rows(
        stored.db, "SELECT submission_id, status, batch_id, provider FROM submissions"
    )
    submission_id, status, batch_id, stored_provider = submission
    assert (status, stored_provider) == ("finalized", provider)
    requests = rows(
        stored.db,
        "SELECT custom_id, idx FROM submission_requests WHERE submission_id = ? "
        "ORDER BY idx",
        (submission_id,),
    )
    assert [idx for _custom_id, idx in requests] == [1, 3]
    responses = rows(
        stored.db,
        "SELECT custom_id, raw_kind, idx FROM responses WHERE submission_id = ? "
        "ORDER BY idx",
        (submission_id,),
    )
    assert [(kind, idx) for _cid, kind, idx in responses] == [("line", 1), ("line", 3)]
    assert len(rows(stored.db, "SELECT 1 FROM responses")) == 7
    units = unit_rows(stored.db)
    custom_ids = dict((idx, custom_id) for custom_id, idx in requests)
    assert units[1][1:4] == ("done", "ok", answer(f"text of {custom_ids[1]}"))
    assert units[3][1:4] == ("done", "ok", answer(f"text of {custom_ids[3]}"))
    provenance = json.loads(
        rows(stored.db, "SELECT provenance FROM units WHERE idx = 1")[0][0]
    )
    assert provenance["custom_id"] == "doc-chunk-2"
    assert "render_dpi" in provenance["image_provenance"]
    document = json.loads(stored.output.read_text(encoding="utf-8"))
    assert [record["custom_id"] for record in document["records"]] == [
        f"doc-chunk-{n}" for n in range(1, 6)
    ]
    assert world.cleaned == [batch_id]
    assert_no_side_files(tmp_path)


class ApiError(Exception):
    """A provider error carrying an HTTP status."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"Error code: {status_code} - refused")
        self.status_code = status_code


@pytest.mark.parametrize(
    ("error", "status"),
    [(ApiError(400), "failed"), (RuntimeError("connection lost"), "unresolved")],
)
async def test_a_batch_not_accepted_leaves_the_units_as_they_were(
    world: World, tmp_path: Path, error: Exception, status: str
) -> None:
    world.errors[("openai", "submit")] = [error]
    stored = await store_document(tmp_path, ["ok", "failed"])
    before = unit_rows(stored.db)
    text = stored.output.read_bytes()

    result = await repair_module.repair_extraction(
        spec_for(stored.output, "batch"), Settings()
    )

    assert (result.repaired, result.failed) == (0, 1)
    assert unit_rows(stored.db) == before
    assert stored.output.read_bytes() == text
    assert rows(stored.db, "SELECT status FROM submissions") == [(status,)]
