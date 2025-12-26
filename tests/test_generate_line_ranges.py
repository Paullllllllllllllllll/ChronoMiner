from __future__ import annotations

from pathlib import Path

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
