"""The extractor's contract on the store, for text, image and PDF documents.

Run on Windows with (``<repo>`` is the repository root):

    uv --directory <repo> run pytest tests/extract/test_llm_store.py -q

A scripted chat model answers every chunk (``tests.llm.helpers``), so no
request leaves the machine. Each test leases the item's document as the run
core does, runs the extractor and reads the rows back through ``mode=ro``
connections; ``_output.json`` is rendered from them with the renderer.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any

import fitz
import pytest
from PIL import Image

import chronominer.extract.chunks as chunks_module
import chronominer.extract.llm as llm_module
from chronominer.common.selection import parse_page_range
from chronominer.common.sqlite import read, read_rows
from chronominer.common.testing import FakeRequest
from chronominer.extract import Item
from chronominer.extract.llm import (
    BatchSubmissionFailedError,
    ChunkingChangedError,
    Extractor,
    ImageSettingsChangedError,
)
from chronominer.extract.units import DocumentHandle, UnitsFailedError
from chronominer.output.output_json import render_output_json
from chronominer.output.store import DATABASE_NAME, document_metadata, open_store
from chronominer.ranges.adjust import AdjustReport
from chronominer.ranges.generate import auto_ranges
from chronominer.ranges.lines import ranges_path, read_lines
from chronominer.settings import Concurrency
from chronominer.spec import DEFAULT_MODEL as MODEL
from tests.conftest import make_job, make_spec
from tests.llm import helpers
from tests.llm.helpers import StatusError
from tests.recording_events import RecordingEvents


def _entries(person: str) -> str:
    entry = {
        "person": person,
        "occupation": None,
        "wage": 1.5,
        "currency": None,
        "period": None,
        "date": None,
        "employer": None,
        "source_page": None,
        "notes": None,
    }
    return json.dumps(
        {"contains_no_content_of_requested_type": False, "entries": [entry]}
    )


def _answer(request: FakeRequest) -> str:
    content = request.messages[-1].content
    if isinstance(content, str):
        return _entries(content.strip().splitlines()[0])
    return _entries("page")


def _ledger(path: Path, records: int = 12) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    blocks = [
        f"Record {i:03d}.\nPerson: worker {i}\nWage: {i} shillings\n"
        for i in range(1, records + 1)
    ]
    path.write_text("\n".join(blocks), encoding="utf-8")
    return path


def _pdf(path: Path, pages: int = 3) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with fitz.open() as document:
        for index in range(pages):
            page = document.new_page(width=200, height=300)
            page.insert_text((50, 100), f"Page {index + 1}")
        document.save(str(path))
    return path


def _image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (40, 60), "white").save(path)
    return path


@pytest.fixture(autouse=True)
def openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set the key variable of the default provider."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


def _run(
    source: Path,
    out: Path,
    kind: str,
    *,
    pages: str | None = None,
    events: RecordingEvents | None = None,
    submit: Any = None,
    **flags: Any,
) -> BaseException | None:
    """Extract *source* into the store of *out*; return the item's error."""
    flags.setdefault("concurrency", 1)
    flags.setdefault("chunk_tokens", 60)
    spec = make_spec(source, output=out, **flags)
    job = make_job(
        spec,
        output_dir=out,
        input_root=source.parent,
        settings=helpers.settings(
            rate_limits=((100_000, 1),),
            concurrency=Concurrency(delay_between_tasks=0.0),
        ),
        events=events,
        batch=submit is not None,
        submit=submit,
    )
    out.mkdir(parents=True, exist_ok=True)

    async def main() -> BaseException | None:
        async with open_store(out) as store:
            await store.begin_run("extract")
            doc_id = await store.ensure_document(
                source.stem, input_path=source, kind=kind
            )
            lease = await store.acquire(doc_id)
            assert lease is not None
            handle = DocumentHandle(
                store,
                doc_id,
                source.stem,
                lease,
                spec.model.model,
                spec.model.provider,
            )
            extractor = Extractor(job)
            await extractor.open()
            item = Item(
                source,
                kind,
                pages=parse_page_range(pages) if pages else None,
                document=handle,
            )
            try:
                await extractor.extract(item)
            except Exception as exc:  # noqa: BLE001 - the tests inspect it
                return exc
            finally:
                await extractor.close()
                await store.release(lease)
                await store.end_run(0)
        return None

    return asyncio.run(main())


def _units(out: Path) -> list[dict[str, Any]]:
    return read_rows(out / DATABASE_NAME, "units", order_by=["kind", "idx"])


def _chunks(out: Path) -> list[dict[str, Any]]:
    return [row for row in _units(out) if row["kind"] == "chunk"]


def _output(out: Path) -> dict[str, Any]:
    text = read(out / DATABASE_NAME, lambda c: render_output_json(c, 1))
    data: dict[str, Any] = json.loads(text)
    return data


def _metadata(out: Path) -> dict[str, Any]:
    stored, _sha = read(out / DATABASE_NAME, lambda c: document_metadata(c, 1))
    return stored


def _text_ranges(source: Path, tokens: int = 60) -> list[tuple[int, int]]:
    return auto_ranges(read_lines(source), tokens, "gpt-5.6-luna")


class TestTextDocuments:
    def test_a_clean_run_commits_every_chunk_with_its_entries(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        script = helpers.install(monkeypatch, [_answer])
        source = _ledger(tmp_path / "in" / "ledger.txt")
        out = tmp_path / "out"
        events = RecordingEvents()
        ranges = _text_ranges(source)
        assert len(ranges) > 2

        assert _run(source, out, "text", events=events) is None

        assert len(script.requests) == len(ranges)
        rows = _chunks(out)
        assert [row["status"] for row in rows] == ["done"] * len(ranges)
        assert [row["source_index"] for row in rows] == [f"{s}-{e}" for s, e in ranges]
        entries = read_rows(out / DATABASE_NAME, "entries", order_by=["chunk_index"])
        assert [entry["person"] for entry in entries] == [
            read_lines(source)[start - 1].strip() for start, _end in ranges
        ]
        assert [event.chunk for event in events.of("chunk_done")] == list(
            range(len(ranges))
        )
        document = _output(out)
        meta = document["_chronominer_metadata"]
        assert (meta["schema_name"], meta["model_name"]) == ("WageRecords", MODEL)
        assert meta["chunking_method"] == "auto"
        assert (meta["chunking_text_version"], meta["document_total_chunks"]) == (
            2,
            len(ranges),
        )
        first = document["records"][0]
        assert first["custom_id"] == "ledger-chunk-1"
        assert first["chunk_range"] == list(ranges[0])
        assert json.loads(first["response"]["output_text"])["entries"]
        assert first["response"]["response_data"]["model"] == MODEL

    def test_a_failing_chunk_fails_the_item_after_the_others_committed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        helpers.install(monkeypatch, [_answer, StatusError(400), _answer])
        source = _ledger(tmp_path / "in" / "ledger.txt")
        out = tmp_path / "out"
        total = len(_text_ranges(source))

        error = _run(source, out, "text")

        assert isinstance(error, UnitsFailedError)
        assert f"1 of {total}" in str(error)
        statuses = [row["status"] for row in _chunks(out)]
        assert statuses == ["done", "failed"] + ["done"] * (total - 2)
        meta = _output(out)["_chronominer_metadata"]
        assert (meta["partial"], meta["failed_chunks"]) == (True, [2])

    def test_a_pages_run_leaves_the_document_partial_and_a_rerun_completes_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        script = helpers.install(monkeypatch, [_answer])
        source = _ledger(tmp_path / "in" / "ledger.txt")
        out = tmp_path / "out"
        total = len(_text_ranges(source))

        assert _run(source, out, "text", pages="first:2") is None
        assert [row["status"] for row in _chunks(out)] == ["done"] * 2 + ["pending"] * (
            total - 2
        )
        assert _metadata(out)["chunk_slice"] == {"first_n": 2}
        first = len(script.requests)

        assert _run(source, out, "text") is None

        assert (first, len(script.requests)) == (2, total)
        assert [row["status"] for row in _chunks(out)] == ["done"] * total
        assert "chunk_slice" not in _metadata(out)

    @pytest.mark.parametrize(
        "changed",
        [{"model": "gpt-5-mini"}, {"chunk_tokens": 40}],
        ids=["model", "chunk_tokens"],
    )
    def test_an_auto_chunked_item_resumes_on_its_stored_ranges(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: dict[str, Any]
    ) -> None:
        helpers.install(monkeypatch, [_answer])
        source = tmp_path / "in" / "ledger.txt"
        source.parent.mkdir(parents=True)
        blocks = [
            f"Record {i:03d}.\nPerson: Arbeiter Müller Nr. {i}\n"
            f"Lohn: {i} Gulden für Wäsche und Kost\n"
            for i in range(1, 13)
        ]
        source.write_text("\n".join(blocks), encoding="utf-8")
        out = tmp_path / "out"
        lines = read_lines(source)
        ranges = auto_ranges(lines, 60, MODEL)
        fresh = auto_ranges(
            lines, changed.get("chunk_tokens", 60), changed.get("model", MODEL)
        )
        assert fresh != ranges

        assert _run(source, out, "text", pages="first:2") is None
        assert _run(source, out, "text", **changed) is None

        rows = _chunks(out)
        assert [row["source_index"] for row in rows] == [f"{s}-{e}" for s, e in ranges]
        assert [row["status"] for row in rows] == ["done"] * len(ranges)

    def test_line_ranges_are_generated_when_missing_and_then_read(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        script = helpers.install(monkeypatch, [_answer])
        source = _ledger(tmp_path / "in" / "ledger.txt")
        out = tmp_path / "out"

        assert _run(source, out, "text", chunking="line-ranges") is None

        ranges_file = ranges_path(source)
        assert ranges_file.is_file()
        generated = ranges_file.read_text(encoding="utf-8").splitlines()
        assert len(script.requests) == len(generated)
        assert _metadata(out)["chunking_method"] == "line_ranges"
        units = _units(out)
        assert {row["kind"] for row in units} == {"chunk", "range"}

        ranges_file.write_text("(1, 4)\n(5, 400)\n", encoding="utf-8")
        assert isinstance(
            _run(source, out, "text", chunking="line-ranges"), ChunkingChangedError
        )
        assert _run(source, out, "text", chunking="line-ranges", force=True) is None
        assert [row["source_index"] for row in _chunks(out)] == [
            "1-4",
            f"5-{len(read_lines(source))}",
        ]

    def test_force_recuts_the_chunks_of_a_document_with_a_finalized_batch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        helpers.install(monkeypatch, [_answer])
        source = _ledger(tmp_path / "in" / "ledger.txt")
        out = tmp_path / "out"
        fresh = _text_ranges(source, tokens=40)
        assert fresh != _text_ranges(source)
        assert _run(source, out, "text") is None
        connection = sqlite3.connect(out / DATABASE_NAME)
        with connection:
            (run_id,) = connection.execute("SELECT run_id FROM runs").fetchone()
            connection.execute(
                "INSERT INTO submissions (submission_id, tool, provider, model, "
                "run_id, intent_at, batch_id, status) VALUES "
                "('s1', 'chronominer', 'openai', ?, ?, '2026-01-01', 'b1', "
                "'finalized')",
                (MODEL, run_id),
            )
            connection.execute(
                "INSERT INTO submission_requests "
                "(submission_id, custom_id, doc_id, kind, idx) VALUES "
                "('s1', 'ledger-chunk-1', 1, 'chunk', 0)"
            )
        connection.close()

        assert _run(source, out, "text", chunk_tokens=40, force=True) is None

        rows = _chunks(out)
        assert [row["source_index"] for row in rows] == [f"{s}-{e}" for s, e in fresh]
        assert [row["status"] for row in rows] == ["done"] * len(fresh)
        assert read_rows(out / DATABASE_NAME, "submission_requests") == []

    @pytest.mark.parametrize("force", [False, True])
    def test_the_adjust_mode_works_under_the_runs_lease(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, force: bool
    ) -> None:
        helpers.install(monkeypatch, [_answer])
        source = _ledger(tmp_path / "in" / "ledger.txt")
        ranges_path(source).write_text("(1, 7)\n(8, 30)\n", encoding="utf-8")
        out = tmp_path / "out"
        calls: list[dict[str, Any]] = []

        async def fake_adjust(store: Any, text_file: Path, **kwargs: Any) -> Any:
            doc_id = await store.ensure_document(kwargs["document"])
            lease = await store.acquire(doc_id)
            await store.release(lease)
            calls.append({"file": text_file, "lease": lease, **kwargs})
            return AdjustReport(
                text_file, ranges_path(text_file), "adjusted", ((1, 9), (10, 30))
            )

        monkeypatch.setattr(chunks_module, "adjust_ranges", fake_adjust)

        assert _run(source, out, "text", chunking="adjust", force=force) is None

        (call,) = calls
        assert (call["file"], call["document"]) == (source, "ledger")
        assert call["options"].boundary_type == "WageRecords"
        assert call["options"].context_window == 6
        assert call["options"].force is force
        assert [row["source_index"] for row in _chunks(out)] == ["1-9", "10-30"]
        assert [row["status"] for row in _chunks(out)] == ["done", "done"]
        assert _metadata(out)["chunking_method"] == "line_ranges.txt"


class TestPageDocuments:
    def test_every_pdf_page_is_a_chunk(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        script = helpers.install(monkeypatch, [_answer])
        out = tmp_path / "out"

        assert _run(_pdf(tmp_path / "in" / "book.pdf"), out, "pdf") is None

        assert len(script.requests) == 3
        rows = _chunks(out)
        assert [(row["label"], row["source_index"]) for row in rows] == [
            ("page_0001", 0),
            ("page_0002", 1),
            ("page_0003", 2),
        ]
        document = _output(out)
        meta = document["_chronominer_metadata"]
        assert meta["chunking_method"] == "pages"
        assert meta["image_provenance"]["source_file"] == "book.pdf"
        assert "chunking_text_version" not in meta
        record = document["records"][1]
        assert record["custom_id"] == "book-page-2"
        assert record["chunk_range"] is None
        assert record["image_provenance"]["image_sha256"]

    def test_an_image_file_is_one_chunk(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        script = helpers.install(monkeypatch, [_answer])
        out = tmp_path / "out"

        assert _run(_image(tmp_path / "in" / "page.png"), out, "image") is None

        assert len(script.requests) == 1
        (row,) = _chunks(out)
        assert (row["label"], row["source_index"], row["status"]) == (
            "page.png",
            "page.png",
            "done",
        )
        assert _output(out)["records"][0]["custom_id"] == "page-page-1"

    def test_changed_image_settings_refuse_the_document_unless_forced(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        script = helpers.install(monkeypatch, [_answer])
        source = _pdf(tmp_path / "in" / "book.pdf")
        out = tmp_path / "out"
        assert _run(source, out, "pdf", pages="1") is None

        error = _run(source, out, "pdf", jpeg_quality=50)

        assert isinstance(error, ImageSettingsChangedError)
        assert "jpeg_quality" in str(error)
        assert len(script.requests) == 1
        assert _run(source, out, "pdf", jpeg_quality=50, force=True) is None
        assert len(script.requests) == 4

    def test_a_model_change_alone_resumes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        script = helpers.install(monkeypatch, [_answer])
        source = _pdf(tmp_path / "in" / "book.pdf")
        out = tmp_path / "out"
        assert _run(source, out, "pdf", pages="1") is None

        assert _run(source, out, "pdf", model="gpt-5.4") is None

        assert len(script.requests) == 3
        assert [row["model"] for row in _chunks(out)] == [MODEL] + ["gpt-5.4"] * 2

    def test_an_error_while_streaming_fails_the_item(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        script = helpers.install(monkeypatch, [_answer])

        def failing_stream(*_args: Any) -> Any:
            async def gen() -> Any:
                raise RuntimeError("incompatible page source")
                yield  # pragma: no cover

            return gen()

        monkeypatch.setattr(llm_module, "stream_pages", failing_stream)
        out = tmp_path / "out"

        error = _run(_pdf(tmp_path / "in" / "book.pdf"), out, "pdf")

        assert isinstance(error, RuntimeError)
        assert "incompatible page source" in str(error)
        assert script.requests == []
        assert [row["status"] for row in _chunks(out)] == ["pending"] * 3


class _Outcome:
    def __init__(self, status: str, detail: str | None = None) -> None:
        self.status = status
        self.detail = detail


class TestBatch:
    @pytest.mark.parametrize(
        ("status", "raised"),
        [
            ("submitted", None),
            ("refused", BatchSubmissionFailedError),
            ("unresolved", BatchSubmissionFailedError),
        ],
    )
    def test_the_chunks_go_to_the_submitter(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        status: str,
        raised: type[BaseException] | None,
    ) -> None:
        script = helpers.install(monkeypatch, [_answer])
        source = _ledger(tmp_path / "in" / "ledger.txt")
        submitted: list[Any] = []

        async def submit(payloads: Any, **kwargs: Any) -> _Outcome:
            submitted.append((payloads, kwargs))
            return _Outcome(status, "no capacity")

        error = _run(source, tmp_path / "out", "text", submit=submit)

        assert (type(error) if error else None) is raised
        assert script.requests == []
        ((payloads, kwargs),) = submitted
        assert [payload.custom_id for payload in payloads][:2] == [
            "ledger-chunk-1",
            "ledger-chunk-2",
        ]
        assert payloads[0].provenance()["chunk_range"] == [
            payloads[0].start,
            payloads[0].end,
        ]
        assert kwargs["schema_path"].name == "wage_records.json"
        if status == "unresolved":
            assert "batch check" in str(error)
