from __future__ import annotations

from pathlib import Path

import pytest

from modules.core.chunking_service import ChunkingService


class DummyTextProcessor:
    def estimate_tokens(self, text: str, model_name: str = "") -> int:
        return len(text)


@pytest.mark.unit
def test_chunking_service_auto_ranges(tmp_path: Path):
    svc = ChunkingService(model_name="x", default_tokens_per_chunk=5, text_processor=DummyTextProcessor())

    lines = ["aa", "bbb", "c", "dd"]
    chunks, ranges = svc.chunk_text(lines, strategy="auto", original_start_line=10)

    assert len(chunks) == len(ranges)
    assert ranges[0][0] == 10


@pytest.mark.unit
def test_chunking_service_line_ranges_file(tmp_path: Path):
    svc = ChunkingService(model_name="x", default_tokens_per_chunk=999, text_processor=DummyTextProcessor())

    lines = ["a", "b", "c", "d"]
    lr = tmp_path / "f_line_ranges.txt"
    lr.write_text("(1, 2)\n(3, 4)\n", encoding="utf-8")

    chunks, ranges = svc.chunk_text(lines, strategy="line_ranges", line_ranges_file=lr)
    assert ranges == [(1, 2), (3, 4)]
    assert chunks == ["ab", "cd"]
