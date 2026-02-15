from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from main.generate_line_ranges import generate_line_ranges_for_file, write_line_ranges_file


@pytest.mark.unit
def test_generate_and_write_line_ranges(tmp_path: Path, monkeypatch):
    text_file = tmp_path / "doc.txt"
    text_file.write_text("a\n" * 20, encoding="utf-8")

    from modules.core.text_utils import TextProcessor

    monkeypatch.setattr(TextProcessor, "detect_encoding", staticmethod(lambda p: "utf-8"))

    ranges = generate_line_ranges_for_file(text_file=text_file, default_tokens_per_chunk=10, model_name="gpt-4o")
    assert ranges

    out = write_line_ranges_file(text_file, ranges)
    assert out.exists()
    assert out.name.endswith("_line_ranges.txt")


@pytest.mark.unit
def test_select_single_file_excludes_context_files(tmp_path: Path):
    """_select_single_file must not return context files as candidates."""
    from main.generate_line_ranges import GenerateLineRangesScript

    raw = tmp_path / "input"
    raw.mkdir()
    (raw / "document.txt").write_text("content", encoding="utf-8")
    (raw / "document_extract_context.txt").write_text("ctx", encoding="utf-8")
    (raw / "document_adjust_context.txt").write_text("ctx", encoding="utf-8")
    (raw / "document_transcr_context.txt").write_text("ctx", encoding="utf-8")
    (raw / "document_line_ranges.txt").write_text("(1, 5)", encoding="utf-8")

    script = GenerateLineRangesScript.__new__(GenerateLineRangesScript)
    mock_ui = MagicMock()
    mock_ui.get_input.return_value = "document.txt"
    script.ui = mock_ui

    result = script._select_single_file(raw)
    assert result is not None
    assert len(result) == 1
    assert result[0].name == "document.txt"
