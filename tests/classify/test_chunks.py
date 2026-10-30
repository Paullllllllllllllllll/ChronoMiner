"""``classify chunks``: units, store writes, reuse, refusals and the dry run."""

from __future__ import annotations

import asyncio
import contextlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import pytest
from PIL import Image as PILImage

from chronominer.classify import plan_classify, run_classify, units
from chronominer.classify import run as classify_run
from chronominer.classify.plan import stored_images
from chronominer.common.store import PlannedUnit, UnitResult, file_sha256
from chronominer.extract.images import resolve_image_settings
from chronominer.output.store import database_path, image_settings_record, open_store
from chronominer.ranges.lines import ranges_path
from chronominer.settings import Settings
from chronominer.spec import ImageSpec, resolve_classify_items
from tests.classify.conftest import (
    MODEL,
    Chunk,
    Classify,
    DecisionServer,
    answer,
    classifier_file,
    rows,
    seed,
    view,
    write_text,
)

CHUNKS = (Chunk(1, 10), Chunk(11, 25), Chunk(26, 40))


def _extraction_tables(root: Path) -> tuple[list[dict], list[dict]]:
    documents = rows(root, "SELECT * FROM documents ORDER BY doc_id")
    units = rows(root, "SELECT * FROM units ORDER BY doc_id, kind, idx")
    return documents, units


def test_a_text_the_store_lacks_is_cut_in_memory_and_writes_no_document(
    tmp_path: Path, server: DecisionServer, classify: Classify
) -> None:
    text = write_text(tmp_path / "in" / "prices.txt")
    out = tmp_path / "out"

    code, summary = classify(
        "chunks",
        "--classifier",
        "language",
        "--input",
        text,
        "--output",
        out,
        "--chunk-tokens",
        "200",
    )

    assert code == 0, summary
    found = view(out)
    assert summary["command"] == "classify" and summary["unit"] == "chunk"
    assert summary["classifier"] == "language"
    assert summary["items_classified"] == 1
    assert summary["units_asked"] == len(found) > 1
    assert summary["outputs"] == [str(database_path(out).resolve())]
    assert summary["usage"]["classify"]["input_tokens"] == 10 * len(server.requests)
    assert {row["unit"] for row in found} == {"chunk"}
    assert {row["doc_id"] for row in found} == {None}
    assert [row["ordinal"] for row in found] == list(range(1, len(found) + 1))
    assert found[0]["locator"].startswith("lines 1-")
    assert {row["document"] for row in found} == {"prices"}
    assert _extraction_tables(out) == ([], [])
    first = server.parts(0)
    assert json.loads(first[0]["text"]) == {"document": "prices"}


def test_a_stored_document_keeps_its_ranges_and_is_left_untouched(
    tmp_path: Path, server: DecisionServer, classify: Classify
) -> None:
    text = write_text(tmp_path / "prices.txt")
    seed(tmp_path, text, CHUNKS)
    before = _extraction_tables(tmp_path)

    code, summary = classify(
        "chunks",
        "--classifier",
        "language",
        "--input",
        text,
        "--chunking",
        "line-ranges",
    )

    assert code == 0, summary
    assert [row["locator"] for row in view(tmp_path)] == [
        "lines 1-10",
        "lines 11-25",
        "lines 26-40",
    ]
    assert _extraction_tables(tmp_path) == before


def test_line_ranges_without_the_file_exit_2_naming_the_ranges_commands(
    tmp_path: Path, server: DecisionServer, classify: Classify
) -> None:
    text = write_text(tmp_path / "prices.txt")

    code, summary = classify(
        "chunks",
        "--classifier",
        "language",
        "--input",
        text,
        "--chunking",
        "line-ranges",
    )

    assert code == 2
    assert summary["items_total"] == 0
    assert not database_path(tmp_path).exists()
    assert server.requests == []


def test_line_ranges_read_the_file_beside_the_input(
    tmp_path: Path, server: DecisionServer, classify: Classify
) -> None:
    text = write_text(tmp_path / "prices.txt")
    ranges_path(text).write_text("(1, 20)\n(21, 40)\n", encoding="utf-8")

    code, _ = classify(
        "chunks",
        "--classifier",
        "language",
        "--input",
        text,
        "--chunking",
        "line-ranges",
    )

    assert code == 0
    assert [row["locator"] for row in view(tmp_path)] == ["lines 1-20", "lines 21-40"]


def test_adjust_is_refused(
    tmp_path: Path, server: DecisionServer, classify: Classify
) -> None:
    text = write_text(tmp_path / "prices.txt")

    code, _ = classify(
        "chunks", "--classifier", "language", "--input", text, "--chunking", "adjust"
    )

    assert code == 2
    assert server.requests == []


def test_a_classifier_of_another_unit_exits_2(
    tmp_path: Path, server: DecisionServer, classify: Classify
) -> None:
    text = write_text(tmp_path / "prices.txt")

    code, _ = classify("chunks", "--classifier", "document_type", "--input", text)

    assert code == 2


def test_a_dry_run_counts_without_writing_then_sees_the_stored_answers(
    tmp_path: Path, server: DecisionServer, classify: Classify
) -> None:
    text = write_text(tmp_path / "prices.txt")
    base = ("chunks", "--classifier", "language", "--input", text)

    code, dry = classify(*base, "--dry-run")
    assert code == 0
    assert dry["dry_run"] is True
    assert dry["units_asked"] > 0 and dry["units_reused"] == 0
    assert dry["targets"][0]["state"] == "classified"
    assert not database_path(tmp_path).exists()
    assert server.requests == []

    classify(*base)
    asked = len(server.requests)
    code, again = classify(*base, "--dry-run")
    assert (again["units_asked"], again["units_reused"]) == (0, dry["units_asked"])
    assert len(server.requests) == asked


def test_answers_are_reused_and_republished_across_models(
    tmp_path: Path, server: DecisionServer, classify: Classify
) -> None:
    text = write_text(tmp_path / "prices.txt")
    base = ("chunks", "--classifier", "language", "--input", text)
    other = ("--decision-model", "vendor/other-model")

    _, first = classify(*base)
    calls = len(server.requests)
    _, second = classify(*base)
    assert (second["units_asked"], second["units_reused"]) == (0, first["units_asked"])
    assert len(server.requests) == calls

    _, model_b = classify(*base, *other)
    assert model_b["units_asked"] == first["units_asked"]
    assert {row["model"] for row in view(tmp_path)} == {"vendor/other-model"}

    _, back = classify(*base)
    assert (back["units_asked"], back["units_reused"]) == (0, first["units_asked"])
    shown = view(tmp_path)
    assert {row["model"] for row in shown} == {"typesafe/jev-1.14"}
    assert {row["reused"] for row in shown} == {1}
    run_ids = {row["run_id"] for row in shown}
    latest = rows(tmp_path, "SELECT run_id FROM runs ORDER BY rowid DESC LIMIT 1")
    assert run_ids == {latest[0]["run_id"]}


def test_force_asks_again(
    tmp_path: Path, server: DecisionServer, classify: Classify
) -> None:
    text = write_text(tmp_path / "prices.txt")
    base = ("chunks", "--classifier", "language", "--input", text)
    _, first = classify(*base)

    _, forced = classify(*base, "--force")

    assert (forced["units_asked"], forced["units_reused"]) == (first["units_asked"], 0)
    assert {row["reused"] for row in view(tmp_path)} == {0}


def test_a_changed_input_is_refused_and_writes_nothing(
    tmp_path: Path, server: DecisionServer, classify: Classify
) -> None:
    text = write_text(tmp_path / "prices.txt")
    seed(tmp_path, text, CHUNKS, input_sha256="0" * 64)

    code, summary = classify("chunks", "--classifier", "language", "--input", text)

    assert code == 1
    assert summary["items_refused"] == 1
    assert server.requests == []
    assert rows(tmp_path, "SELECT * FROM classification_rows") == []


def test_a_drop_during_an_in_flight_request_fails_the_item_unwritten(
    tmp_path: Path, server: DecisionServer, classify: Classify
) -> None:
    text = write_text(tmp_path / "prices.txt")
    doc_id = seed(tmp_path, text, CHUNKS)

    def drop() -> None:
        with contextlib.closing(sqlite3.connect(database_path(tmp_path))) as db:
            for table in ("entry_rows", "document_meta", "units", "documents"):
                db.execute(f"DELETE FROM {table} WHERE doc_id = ?", (doc_id,))
            db.commit()

    server.on_request = drop

    code, summary = classify("chunks", "--classifier", "language", "--input", text)

    assert code == 1
    assert (summary["items_failed"], summary["items_refused"]) == (1, 0)
    assert len(server.requests) == 1
    assert rows(tmp_path, "SELECT * FROM classification_rows") == []


def test_a_failed_request_fails_the_item(
    tmp_path: Path, server: DecisionServer, classify: Classify
) -> None:
    server.fail = True
    text = write_text(tmp_path / "prices.txt")

    code, summary = classify("chunks", "--classifier", "language", "--input", text)

    assert code == 1
    assert summary["items_failed"] == 1
    assert summary["units_asked"] == 0


def test_pdf_pages_go_as_decision_images_named_by_page(
    tmp_path: Path, server: DecisionServer, classify: Classify
) -> None:
    pdf = tmp_path / "book.pdf"
    document = fitz.open()
    for number in range(3):
        page = document.new_page(width=200, height=300)
        page.insert_text((50, 100), f"Page {number + 1}")
    document.save(pdf)
    document.close()

    code, summary = classify(
        "chunks", "--classifier", "chunk_kind", "--input", pdf, "--pages", "2-3"
    )

    assert code == 0, summary
    found = view(tmp_path)
    assert [(row["locator"], row["ordinal"]) for row in found] == [
        ("page 2", 2),
        ("page 3", 3),
    ]
    assert {row["question"] for row in found} == {"kind"}
    images = [part for part in server.parts() if part["type"] == "image_url"]
    assert len(images) == 2
    assert images[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def _pdf(path: Path, pages: int) -> Path:
    document = fitz.open()
    for number in range(pages):
        document.new_page(width=200, height=300).insert_text((50, 100), str(number))
    document.save(path)
    document.close()
    return path


def _stored_pages(
    root: Path,
    pdf: Path,
    images: ImageSpec,
    model: str = MODEL,
    side: int | None = None,
) -> dict:
    """Store page 1 of *pdf* as extracted with *images*; return the record."""
    settings, family, _section = resolve_image_settings("openai", model, images, side)
    record = image_settings_record(settings, family)

    async def main() -> None:
        async with open_store(root) as store:
            await store.begin_run("extract")
            doc_id = await store.ensure_document(
                pdf.stem, input_path=pdf, input_sha256=file_sha256(pdf), kind="pdf"
            )
            lease = await store.acquire(doc_id)
            assert lease is not None
            planned = [PlannedUnit(0, "page_0001", 0)]
            await store.plan_units(
                lease, "chunk", planned, model=MODEL, provider="openai"
            )
            await store.set_image_settings(lease, record)
            await store.write_unit(
                lease,
                UnitResult("chunk", 0, "done", MODEL, "openai", content=answer()),
            )

    asyncio.run(main())
    return record


def test_stored_pages_keep_their_settings_and_refuse_conflicting_options(
    tmp_path: Path, server: DecisionServer, classify: Classify
) -> None:
    pdf = _pdf(tmp_path / "book.pdf", 2)
    record = _stored_pages(tmp_path, pdf, ImageSpec(dpi=150, detail="high"))
    base = ("chunks", "--classifier", "chunk_kind", "--input", pdf)

    assert (stored_images(record).dpi, stored_images(record).detail) == (150, "high")
    code, summary = classify(*base, "--dpi", "200")
    assert (code, summary["items_refused"]) == (1, 1)
    assert server.requests == []

    code, summary = classify(*base, "--dpi", "150")
    assert (code, summary["units_asked"]) == (0, 2)


def test_stored_pages_keep_their_recorded_long_side_over_the_settings_file(
    tmp_path: Path,
    server: DecisionServer,
    classify: Classify,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = _pdf(tmp_path / "book.pdf", 1)
    images = ImageSpec(detail="original")
    record = _stored_pages(tmp_path, pdf, images, "gpt-6-luna", side=1000)
    assert record["original_max_side_px"] == 1000
    settings = tmp_path / "settings.yaml"
    settings.write_text("images:\n  original_max_side_px: 2000\n", encoding="utf-8")
    used: list[tuple[dict, str]] = []
    real = classify_run.page_units

    def page_units(*args: Any, **kwargs: Any) -> Any:
        used.append((kwargs["img_cfg"], kwargs["model_type"]))
        return real(*args, **kwargs)

    monkeypatch.setattr(classify_run, "page_units", page_units)

    code, _ = classify(
        "chunks", "--classifier", "chunk_kind", "--input", pdf, "--settings", settings
    )

    assert code == 0
    [(img_cfg, family)] = used
    assert image_settings_record(img_cfg, family) == record


def test_a_page_source_that_gives_up_fails_its_item_and_the_run_goes_on(
    tmp_path: Path,
    server: DecisionServer,
    classify: Classify,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder = tmp_path / "scans"
    folder.mkdir()
    broken = _pdf(folder / "a.pdf", 2)
    _pdf(folder / "b.pdf", 2)
    real = units.stream_pages

    async def stream_pages(source: Any, indices: Any) -> Any:
        if getattr(source, "source_path", None) == broken:
            raise RuntimeError("2 of 2 pages failed to render")
        async for page in real(source, indices):
            yield page

    monkeypatch.setattr(units, "stream_pages", stream_pages)

    code, summary = classify(
        "chunks", "--classifier", "chunk_kind", "--input", folder, "--all"
    )

    assert code == 1
    assert (summary["items_failed"], summary["items_classified"]) == (1, 1)
    assert {row["document"] for row in view(folder)} == {"b"}


@pytest.mark.parametrize("kind", ["text", "pdf"])
def test_an_input_changed_after_planning_is_refused_unasked(
    tmp_path: Path, server: DecisionServer, kind: str
) -> None:
    path = (
        write_text(tmp_path / "prices.txt")
        if kind == "text"
        else _pdf(tmp_path / "book.pdf", 1)
    )
    flags = {"input": path, "classifier": "chunk_kind"}
    plan = plan_classify(resolve_classify_items("chunk", flags).spec, Settings())
    with path.open("ab") as handle:
        handle.write(b"\nchanged\n")

    result = asyncio.run(run_classify(plan))

    assert [target.state for target in result.targets] == ["refused"]
    assert server.requests == []


def test_a_page_count_that_fills_whole_requests_is_classified(
    tmp_path: Path, server: DecisionServer, classify: Classify
) -> None:
    pdf = _pdf(tmp_path / "book.pdf", 2)
    one = classifier_file(
        tmp_path / "classifiers",
        {
            "name": "one_at_a_time",
            "version": 1,
            "unit": "chunk",
            "questions": [{"name": "kind", "type": "page_types"}],
        },
    )

    code, summary = classify("chunks", "--classifier", one, "--input", pdf)

    assert code == 0, summary
    assert (summary["items_classified"], summary["units_asked"]) == (1, 2)
    assert len(server.requests) == 2


def test_an_image_is_one_chunk_named_by_its_file(
    tmp_path: Path, server: DecisionServer, classify: Classify
) -> None:
    image = tmp_path / "scan_01.png"
    PILImage.new("RGBA", (900, 1200), (200, 190, 170, 255)).save(image)

    code, summary = classify("chunks", "--classifier", "chunk_kind", "--input", image)

    assert code == 0, summary
    [row] = view(tmp_path)
    assert (row["locator"], row["ordinal"], row["document"]) == (
        "scan_01.png",
        1,
        "scan_01",
    )
