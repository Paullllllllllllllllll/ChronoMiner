# tests/test_chunk_slice.py

"""Tests for ChunkSlice dataclass and apply_chunk_slice() helper."""

import logging
import pytest
from typing import List, Tuple

from modules.core.chunking_service import ChunkSlice, apply_chunk_slice


# ---------------------------------------------------------------------------
# ChunkSlice validation
# ---------------------------------------------------------------------------

class TestChunkSliceValidation:
    """Tests for ChunkSlice construction and validation."""

    def test_default_none(self):
        cs = ChunkSlice()
        assert cs.first_n is None
        assert cs.last_n is None

    def test_first_n_only(self):
        cs = ChunkSlice(first_n=5)
        assert cs.first_n == 5
        assert cs.last_n is None

    def test_last_n_only(self):
        cs = ChunkSlice(last_n=3)
        assert cs.first_n is None
        assert cs.last_n == 3

    def test_mutual_exclusivity(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            ChunkSlice(first_n=2, last_n=3)

    def test_first_n_zero_raises(self):
        with pytest.raises(ValueError, match="first_n must be >= 1"):
            ChunkSlice(first_n=0)

    def test_first_n_negative_raises(self):
        with pytest.raises(ValueError, match="first_n must be >= 1"):
            ChunkSlice(first_n=-1)

    def test_last_n_zero_raises(self):
        with pytest.raises(ValueError, match="last_n must be >= 1"):
            ChunkSlice(last_n=0)

    def test_last_n_negative_raises(self):
        with pytest.raises(ValueError, match="last_n must be >= 1"):
            ChunkSlice(last_n=-5)

    def test_first_n_one_valid(self):
        cs = ChunkSlice(first_n=1)
        assert cs.first_n == 1

    def test_last_n_one_valid(self):
        cs = ChunkSlice(last_n=1)
        assert cs.last_n == 1


# ---------------------------------------------------------------------------
# apply_chunk_slice
# ---------------------------------------------------------------------------

def _make_chunks(n: int) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Create n dummy chunks and ranges for testing."""
    chunks = [f"chunk_{i}" for i in range(1, n + 1)]
    ranges = [(i * 10, i * 10 + 9) for i in range(1, n + 1)]
    return chunks, ranges


class TestApplyChunkSlice:
    """Tests for the apply_chunk_slice helper function."""

    def test_none_slice_returns_all(self):
        chunks, ranges = _make_chunks(5)
        result_chunks, result_ranges = apply_chunk_slice(chunks, ranges, None)
        assert result_chunks == chunks
        assert result_ranges == ranges

    def test_first_n_basic(self):
        chunks, ranges = _make_chunks(10)
        cs = ChunkSlice(first_n=3)
        result_chunks, result_ranges = apply_chunk_slice(chunks, ranges, cs)
        assert result_chunks == chunks[:3]
        assert result_ranges == ranges[:3]

    def test_last_n_basic(self):
        chunks, ranges = _make_chunks(10)
        cs = ChunkSlice(last_n=4)
        result_chunks, result_ranges = apply_chunk_slice(chunks, ranges, cs)
        assert result_chunks == chunks[-4:]
        assert result_ranges == ranges[-4:]

    def test_first_n_exceeds_total(self, caplog):
        chunks, ranges = _make_chunks(3)
        cs = ChunkSlice(first_n=10)
        with caplog.at_level(logging.WARNING):
            result_chunks, result_ranges = apply_chunk_slice(chunks, ranges, cs)
        assert result_chunks == chunks
        assert result_ranges == ranges
        assert "only 3 available" in caplog.text

    def test_last_n_exceeds_total(self, caplog):
        chunks, ranges = _make_chunks(2)
        cs = ChunkSlice(last_n=5)
        with caplog.at_level(logging.WARNING):
            result_chunks, result_ranges = apply_chunk_slice(chunks, ranges, cs)
        assert result_chunks == chunks
        assert result_ranges == ranges
        assert "only 2 available" in caplog.text

    def test_first_n_equals_total(self, caplog):
        chunks, ranges = _make_chunks(5)
        cs = ChunkSlice(first_n=5)
        with caplog.at_level(logging.WARNING):
            result_chunks, result_ranges = apply_chunk_slice(chunks, ranges, cs)
        assert result_chunks == chunks
        assert result_ranges == ranges

    def test_last_n_equals_total(self, caplog):
        chunks, ranges = _make_chunks(5)
        cs = ChunkSlice(last_n=5)
        with caplog.at_level(logging.WARNING):
            result_chunks, result_ranges = apply_chunk_slice(chunks, ranges, cs)
        assert result_chunks == chunks
        assert result_ranges == ranges

    def test_first_n_one(self):
        chunks, ranges = _make_chunks(5)
        cs = ChunkSlice(first_n=1)
        result_chunks, result_ranges = apply_chunk_slice(chunks, ranges, cs)
        assert result_chunks == [chunks[0]]
        assert result_ranges == [ranges[0]]

    def test_last_n_one(self):
        chunks, ranges = _make_chunks(5)
        cs = ChunkSlice(last_n=1)
        result_chunks, result_ranges = apply_chunk_slice(chunks, ranges, cs)
        assert result_chunks == [chunks[-1]]
        assert result_ranges == [ranges[-1]]

    def test_ranges_aligned_with_chunks(self):
        """Verify ranges are sliced in exact sync with chunks."""
        chunks = ["a", "b", "c", "d", "e"]
        ranges = [(1, 10), (11, 20), (21, 30), (31, 40), (41, 50)]
        cs = ChunkSlice(last_n=2)
        result_chunks, result_ranges = apply_chunk_slice(chunks, ranges, cs)
        assert result_chunks == ["d", "e"]
        assert result_ranges == [(31, 40), (41, 50)]

    def test_empty_chunks_none_slice(self):
        result_chunks, result_ranges = apply_chunk_slice([], [], None)
        assert result_chunks == []
        assert result_ranges == []

    def test_default_chunk_slice_returns_all(self):
        """A ChunkSlice with both fields None behaves like None."""
        chunks, ranges = _make_chunks(5)
        cs = ChunkSlice()
        result_chunks, result_ranges = apply_chunk_slice(chunks, ranges, cs)
        assert result_chunks == chunks
        assert result_ranges == ranges
