from __future__ import annotations

from pathlib import Path

import pytest

from modules.infra.chunking import ChunkingService


class DummyTextProcessor:
    """Test double for TextProcessor that counts characters as tokens."""

    def estimate_tokens(self, text: str) -> int:
        return len(text)


class TestChunkingServiceBasic:
    """Basic tests for ChunkingService."""

    @pytest.mark.unit
    def test_auto_ranges(self, tmp_path: Path):
        """Test automatic chunking with start line offset."""
        svc = ChunkingService(
            model_name="x",
            default_tokens_per_chunk=5,
            text_processor=DummyTextProcessor(),
        )

        lines = ["aa", "bbb", "c", "dd"]
        chunks, ranges = svc.chunk_text(lines, strategy="auto", original_start_line=10)

        assert len(chunks) == len(ranges)
        assert ranges[0][0] == 10

    @pytest.mark.unit
    def test_line_ranges_file(self, tmp_path: Path):
        """Test chunking from line ranges file."""
        svc = ChunkingService(
            model_name="x",
            default_tokens_per_chunk=999,
            text_processor=DummyTextProcessor(),
        )

        lines = ["a", "b", "c", "d"]
        lr = tmp_path / "f_line_ranges.txt"
        lr.write_text("(1, 2)\n(3, 4)\n", encoding="utf-8")

        chunks, ranges = svc.chunk_text(
            lines, strategy="line_ranges", line_ranges_file=lr
        )
        assert ranges == [(1, 2), (3, 4)]
        # Lines are joined with "\n" (production feeds rstripped lines).
        assert chunks == ["a\nb", "c\nd"]

    @pytest.mark.unit
    def test_line_ranges_txt_strategy(self, tmp_path: Path):
        """Test line_ranges.txt strategy name variant."""
        svc = ChunkingService(
            model_name="test",
            default_tokens_per_chunk=100,
            text_processor=DummyTextProcessor(),
        )

        lines = ["line1\n", "line2\n", "line3\n", "line4\n"]
        lr = tmp_path / "ranges.txt"
        lr.write_text("1, 2\n3, 4\n", encoding="utf-8")

        chunks, ranges = svc.chunk_text(
            lines, strategy="line_ranges.txt", line_ranges_file=lr
        )

        assert len(chunks) == 2
        assert ranges == [(1, 2), (3, 4)]


class TestChunkingServiceStartLineOffset:
    """Chunk content must stay aligned when original_start_line > 1.

    Regression: _chunk_with_adjustment used to split the local ``lines`` list
    with document-space ranges (already offset by original_start_line), so the
    chunks shifted by original_start_line - 1 lines. The auto path never had
    this bug; auto-adjust must now mirror it.
    """

    @pytest.mark.unit
    def test_auto_adjust_offset_matches_local_content(self, monkeypatch):
        """auto-adjust with offset keeps defaults and slices local lines."""
        # Keep every default boundary (press Enter for each prompt).
        monkeypatch.setattr("builtins.input", lambda *a, **k: "")

        svc = ChunkingService(
            model_name="x",
            default_tokens_per_chunk=5,
            text_processor=DummyTextProcessor(),
        )
        lines = ["aa", "bbb", "c", "dd"]

        chunks, ranges = svc.chunk_text(
            lines, strategy="auto-adjust", original_start_line=10
        )

        # Returned ranges are document-space (offset by original_start_line).
        assert ranges[0][0] == 10
        # But the chunk text is sliced from the LOCAL lines, so no content is
        # dropped or shifted: every original line survives across the chunks.
        assert "\n".join(chunks).split("\n") == lines

    @pytest.mark.unit
    def test_auto_adjust_offset_equals_auto_path(self, monkeypatch):
        """With defaults kept, auto-adjust yields the same chunks as auto."""
        monkeypatch.setattr("builtins.input", lambda *a, **k: "")

        svc = ChunkingService(
            model_name="x",
            default_tokens_per_chunk=5,
            text_processor=DummyTextProcessor(),
        )
        lines = ["aa", "bbb", "c", "dd"]

        auto_chunks, auto_ranges = svc.chunk_text(
            lines, strategy="auto", original_start_line=10
        )
        adj_chunks, adj_ranges = svc.chunk_text(
            lines, strategy="auto-adjust", original_start_line=10
        )

        assert adj_chunks == auto_chunks
        assert adj_ranges == auto_ranges

    @pytest.mark.unit
    def test_auto_offset_content_aligned(self):
        """The auto path slices local lines correctly under an offset."""
        svc = ChunkingService(
            model_name="x",
            default_tokens_per_chunk=5,
            text_processor=DummyTextProcessor(),
        )
        lines = ["aa", "bbb", "c", "dd"]

        chunks, ranges = svc.chunk_text(lines, strategy="auto", original_start_line=10)

        assert ranges[0][0] == 10
        assert "\n".join(chunks).split("\n") == lines


class TestChunkingServiceFallback:
    """Test fallback behavior in ChunkingService."""

    @pytest.mark.unit
    def test_fallback_when_no_line_ranges_file(self, tmp_path: Path):
        """Test fallback to auto when line ranges file doesn't exist."""
        svc = ChunkingService(
            model_name="test",
            default_tokens_per_chunk=100,
            text_processor=DummyTextProcessor(),
        )

        lines = ["a", "b", "c"]
        nonexistent = tmp_path / "nonexistent.txt"

        chunks, ranges = svc.chunk_text(
            lines, strategy="line_ranges", line_ranges_file=nonexistent
        )

        # Should fall back to auto chunking
        assert len(chunks) > 0

    @pytest.mark.unit
    def test_fallback_when_no_file_path(self, tmp_path: Path):
        """Test fallback to auto when line_ranges_file is None."""
        svc = ChunkingService(
            model_name="test",
            default_tokens_per_chunk=100,
            text_processor=DummyTextProcessor(),
        )

        lines = ["line1\n", "line2\n"]

        chunks, ranges = svc.chunk_text(
            lines, strategy="line_ranges", line_ranges_file=None
        )

        assert len(chunks) > 0

    @pytest.mark.unit
    def test_unknown_strategy_fallback(self, tmp_path: Path):
        """Test unknown strategy defaults to auto."""
        svc = ChunkingService(
            model_name="test",
            default_tokens_per_chunk=100,
            text_processor=DummyTextProcessor(),
        )

        lines = ["a", "b", "c"]

        chunks, ranges = svc.chunk_text(lines, strategy="unknown_strategy")

        assert len(chunks) > 0


class TestChunkingServiceFromConfig:
    """Test ChunkingService.from_config factory method."""

    @pytest.mark.unit
    def test_from_config_defaults(self):
        """Test from_config with minimal config."""
        config = {}
        svc = ChunkingService.from_config(config)

        assert svc.model_name == "o3-mini"
        assert svc.default_tokens_per_chunk == 7500

    @pytest.mark.unit
    def test_from_config_custom_values(self):
        """Test from_config with custom values."""
        config = {
            "model_name": "gpt-4o",
            "default_tokens_per_chunk": 5000,
        }
        svc = ChunkingService.from_config(config)

        assert svc.model_name == "gpt-4o"
        assert svc.default_tokens_per_chunk == 5000


class TestChunkingServiceEdgeCases:
    """Test edge cases for ChunkingService."""

    @pytest.mark.unit
    def test_empty_lines(self):
        """Test chunking with empty lines list."""
        svc = ChunkingService(
            model_name="test",
            default_tokens_per_chunk=100,
            text_processor=DummyTextProcessor(),
        )

        chunks, ranges = svc.chunk_text([], strategy="auto")

        # Empty input produces a single empty chunk (actual behavior)
        assert len(chunks) == 1
        assert chunks[0] == ""

    @pytest.mark.unit
    def test_single_line(self):
        """Test chunking with single line."""
        svc = ChunkingService(
            model_name="test",
            default_tokens_per_chunk=100,
            text_processor=DummyTextProcessor(),
        )

        lines = ["single line"]
        chunks, ranges = svc.chunk_text(lines, strategy="auto")

        assert len(chunks) == 1
        assert len(ranges) == 1

    @pytest.mark.unit
    def test_very_small_chunk_size(self):
        """Test with very small chunk size creating many chunks."""
        svc = ChunkingService(
            model_name="test",
            default_tokens_per_chunk=2,
            text_processor=DummyTextProcessor(),
        )

        lines = ["aaa", "bbb", "ccc", "ddd"]
        chunks, ranges = svc.chunk_text(lines, strategy="auto")

        # Should create multiple chunks due to small size
        assert len(chunks) >= 2

    @pytest.mark.unit
    def test_large_chunk_size(self):
        """Test with large chunk size creating single chunk."""
        svc = ChunkingService(
            model_name="test",
            default_tokens_per_chunk=10000,
            text_processor=DummyTextProcessor(),
        )

        lines = ["a", "b", "c", "d"]
        chunks, ranges = svc.chunk_text(lines, strategy="auto")

        # Should create single chunk since content fits
        assert len(chunks) == 1

    @pytest.mark.unit
    def test_initialization_without_text_processor(self):
        """Test that ChunkingService creates default TextProcessor."""
        svc = ChunkingService(
            model_name="test",
            default_tokens_per_chunk=100,
            text_processor=None,  # Should create default
        )

        assert svc.text_processor is not None
