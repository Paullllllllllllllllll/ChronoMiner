from __future__ import annotations

from pathlib import Path

import pytest

from modules.core.data_processing import CSVConverter
from modules.core.text_processing import DocumentConverter


def test_document_converter_convert_to_txt_writes_message_when_no_entries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    converter = DocumentConverter(schema_name="StructuredSummaries")
    monkeypatch.setattr(converter, "get_entries", lambda _path: [])

    json_file = tmp_path / "input.json"
    json_file.write_text("{}", encoding="utf-8")

    out = tmp_path / "out.txt"
    converter.convert_to_txt(json_file, out)

    assert out.exists()
    assert "No valid entries found" in out.read_text(encoding="utf-8")


def test_csv_converter_convert_to_csv_returns_when_no_entries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    converter = CSVConverter(schema_name="StructuredSummaries")
    monkeypatch.setattr(converter, "get_entries", lambda _path: [])

    json_file = tmp_path / "input.json"
    json_file.write_text("{}", encoding="utf-8")

    out = tmp_path / "out.csv"
    converter.convert_to_csv(json_file, out)

    assert not out.exists()


def test_document_converter_convert_ignores_unsupported_suffix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    converter = DocumentConverter(schema_name="StructuredSummaries")

    monkeypatch.setattr(converter, "convert_to_docx", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("docx")))
    monkeypatch.setattr(converter, "convert_to_txt", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("txt")))

    json_file = tmp_path / "input.json"
    json_file.write_text("{}", encoding="utf-8")

    out = tmp_path / "out.pdf"
    converter.convert(json_file, out)
