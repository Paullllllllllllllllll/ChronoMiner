"""``classify documents``: one request per item from the classifier's selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import pytest

from chronominer.classify import units
from chronominer.ranges.generate import count_tokens
from tests.classify.conftest import (
    Classify,
    DecisionServer,
    classifier_file,
    rows,
    view,
    write_text,
)


def _document_classifier(folder: Path, **selection: int) -> Path:
    return classifier_file(
        folder,
        {
            "name": "kind_of_source",
            "version": 1,
            "unit": "document",
            "selection": selection,
            "questions": [
                {"name": "list", "type": "predicate", "instructions": "A list."}
            ],
        },
    )


def test_a_text_sends_its_leading_chunks_within_the_token_limit(
    tmp_path: Path, server: DecisionServer, classify: Classify
) -> None:
    text = write_text(tmp_path / "in" / "prices.txt")
    classifier = _document_classifier(tmp_path / "classifiers", max_tokens=150)

    code, summary = classify(
        "documents",
        "--classifier",
        classifier,
        "--input",
        text,
        "--output",
        tmp_path / "out",
        "--chunk-tokens",
        "100",
    )

    assert code == 0, summary
    assert len(server.requests) == 1
    texts = server.parts()[1:]
    assert 1 <= len(texts) < 4
    assert texts[0]["text"].startswith("Entry 1:")
    [row] = view(tmp_path / "out")
    assert (row["unit"], row["locator"], row["ordinal"]) == ("document", "document", 0)
    assert row["doc_id"] is None
    assert rows(tmp_path / "out", "SELECT * FROM documents") == []


def test_a_pdf_sends_its_first_pages_in_one_request(
    tmp_path: Path, server: DecisionServer, classify: Classify
) -> None:
    pdf = tmp_path / "book.pdf"
    document = fitz.open()
    for number in range(3):
        document.new_page(width=200, height=300).insert_text((50, 100), str(number))
    document.save(pdf)
    document.close()
    classifier = _document_classifier(tmp_path / "classifiers", first_pages=2)

    code, _ = classify("documents", "--classifier", classifier, "--input", pdf)

    assert code == 0
    [request] = server.requests
    images = [part for part in request["state"] if part["type"] == "image_url"]
    assert len(images) == 2
    assert [row["locator"] for row in view(tmp_path)] == ["document"]


def test_the_shipped_document_type_classifier_answers_one_choice(
    tmp_path: Path, server: DecisionServer, classify: Classify
) -> None:
    text = write_text(tmp_path / "prices.txt")

    code, summary = classify(
        "documents", "--classifier", "document_type", "--input", text
    )

    assert code == 0
    assert summary["units_asked"] == 1
    [row] = view(tmp_path)
    assert (row["question"], row["kind"], row["value"]) == (
        "type",
        "choice",
        "printed_book",
    )


def test_a_first_chunk_over_the_limit_is_cut_to_the_limit(
    tmp_path: Path, server: DecisionServer, classify: Classify
) -> None:
    text = write_text(tmp_path / "prices.txt")
    classifier = _document_classifier(tmp_path / "classifiers", max_tokens=20)

    code, _ = classify("documents", "--classifier", classifier, "--input", text)

    assert code == 0
    [sample] = server.parts()[1:]
    assert sample["text"].startswith("Entry 1:")
    assert 0 < count_tokens(sample["text"], "gpt-6-luna") <= 20


def test_a_page_that_cannot_be_prepared_fails_the_document_unasked(
    tmp_path: Path,
    server: DecisionServer,
    classify: Classify,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = tmp_path / "book.pdf"
    document = fitz.open()
    for number in range(3):
        document.new_page(width=200, height=300).insert_text((50, 100), str(number))
    document.save(pdf)
    document.close()
    real = units.decision_jpeg
    calls: list[int] = []

    def flaky(data: bytes, max_px: int) -> Any:
        calls.append(1)
        if len(calls) == 2:
            raise OSError("broken page")
        return real(data, max_px)

    monkeypatch.setattr(units, "decision_jpeg", flaky)
    classifier = _document_classifier(tmp_path / "classifiers", first_pages=3)

    code, summary = classify("documents", "--classifier", classifier, "--input", pdf)

    assert code == 1
    assert (summary["items_failed"], summary["units_asked"]) == (1, 0)
    assert server.requests == []
    assert view(tmp_path) == []
