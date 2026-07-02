"""Regression tests for the line-range readjuster defect fixes.

Covers marker-matching hardening (bounded search, nearest-match, minimum
marker length, inversion rejection), geometric window generation, deletion-
aware gap enforcement, single-range overlap sanitization, absolute-index
chunk slicing, coverage-gated finalization, prompt payload structure
(sentinel, boundary type, cache-friendly retry guidance), and newline-aware
token counting.
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import tiktoken

from modules.infra.chunking import TextProcessor, TokenBasedChunking
from modules.infra.jsonl import read_jsonl_header, read_jsonl_records
from modules.line_ranges.readjuster import (
    BOUNDARY_SENTINEL,
    BoundaryDecision,
    LineRangeReadjuster,
    RangeResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_readjuster(
    context_window: int = 3,
    matching_config: dict[str, Any] | None = None,
    retry_config: dict[str, Any] | None = None,
) -> LineRangeReadjuster:
    """Create a readjuster without needing real API keys or prompt files."""
    with (
        patch(
            "modules.line_ranges.readjuster.load_prompt_template",
            return_value="fake prompt",
        ),
        patch(
            "modules.line_ranges.readjuster.detect_capabilities",
            return_value=MagicMock(supports_prompt_caching=False),
        ),
    ):
        return LineRangeReadjuster(
            {"extraction_model": {"name": "gpt-4o"}},
            context_window=context_window,
            matching_config=matching_config,
            retry_config=retry_config,
        )


def _make_raw_lines(n: int = 200) -> list[str]:
    return [f"filler line {i}\n" for i in range(1, n + 1)]


def _decision(marker: str, certainty: int = 90) -> BoundaryDecision:
    return BoundaryDecision(
        contains_no_semantic_boundary=False,
        needs_more_context=False,
        certainty=certainty,
        semantic_marker=marker,
    )


class _async_noop_context:
    """Async context manager stand-in for open_extractor."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> MagicMock:
        return MagicMock()

    async def __aexit__(self, *exc: Any) -> None:
        return None


def _fake_range_result(
    index: int,
    original: tuple[int, int],
    adjusted: tuple[int, int],
    delete: bool = False,
) -> RangeResult:
    return RangeResult(
        range_index=index,
        original_range=original,
        adjusted_range=adjusted,
        should_delete=delete,
        decision=BoundaryDecision(
            contains_no_semantic_boundary=delete,
            needs_more_context=False,
            certainty=90,
            semantic_marker=None if delete else "marker",
        ),
        attempts=[],
        total_llm_calls=1,
    )


# ---------------------------------------------------------------------------
# Marker matching: bounded search, nearest-match, minimum length
# ---------------------------------------------------------------------------


class TestMatchBoundaryText:
    def test_no_whole_document_fallback(self) -> None:
        """A marker far outside the search window must NOT match: the old
        whole-document fallback produced boundary jumps of hundreds of
        lines."""
        readjuster = _make_readjuster(context_window=3)
        raw_lines = _make_raw_lines(500)
        raw_lines[449] = "UNIQUE-BOUNDARY-MARKER here\n"  # line 450

        matched = readjuster._match_boundary_text(
            marker="UNIQUE-BOUNDARY-MARKER",
            raw_lines=raw_lines,
            search_start=94,
            search_end=112,
            nearest_to=100,
        )
        assert matched is None

    def test_bounded_expansion_matches_near_window(self) -> None:
        """A marker just outside the window (within one context radius) is
        still found."""
        readjuster = _make_readjuster(context_window=3)
        raw_lines = _make_raw_lines(500)
        raw_lines[113] = "UNIQUE-BOUNDARY-MARKER here\n"  # line 114

        matched = readjuster._match_boundary_text(
            marker="UNIQUE-BOUNDARY-MARKER",
            raw_lines=raw_lines,
            search_start=94,
            search_end=112,  # expanded window reaches 115
            nearest_to=100,
        )
        assert matched == 114

    def test_ambiguous_match_resolves_to_nearest(self) -> None:
        """Multiple matches no longer fail: the candidate nearest the
        original start wins."""
        readjuster = _make_readjuster(context_window=3)
        raw_lines = _make_raw_lines(200)
        raw_lines[89] = "Recipe: Consomme\n"  # line 90 (|90-100| = 10)
        raw_lines[107] = "Recipe: Consomme\n"  # line 108 (|108-100| = 8)

        matched = readjuster._match_boundary_text(
            marker="Recipe: Consomme",
            raw_lines=raw_lines,
            search_start=85,
            search_end=115,
            nearest_to=100,
        )
        assert matched == 108

    def test_ambiguous_tie_breaks_to_earlier_line(self) -> None:
        readjuster = _make_readjuster(context_window=3)
        raw_lines = _make_raw_lines(200)
        raw_lines[94] = "Recipe: Consomme\n"  # line 95
        raw_lines[104] = "Recipe: Consomme\n"  # line 105

        matched = readjuster._match_boundary_text(
            marker="Recipe: Consomme",
            raw_lines=raw_lines,
            search_start=85,
            search_end=115,
            nearest_to=100,
        )
        assert matched == 95

    def test_short_marker_rejected(self) -> None:
        """Markers below min_substring_length are rejected so the mismatch
        retry can request a longer one (the knob used to be dead)."""
        readjuster = _make_readjuster()  # default min_substring_length = 8
        raw_lines = ["aaa\n", "short\n", "ccc\n"]

        matched = readjuster._match_boundary_text(
            marker="short",
            raw_lines=raw_lines,
            search_start=1,
            search_end=3,
            nearest_to=1,
        )
        assert matched is None

    def test_marker_at_min_length_accepted(self) -> None:
        readjuster = _make_readjuster(
            matching_config={"min_substring_length": 5},
        )
        raw_lines = ["aaa\n", "short\n", "ccc\n"]

        matched = readjuster._match_boundary_text(
            marker="short",
            raw_lines=raw_lines,
            search_start=1,
            search_end=3,
            nearest_to=1,
        )
        assert matched == 2


class TestValidateAndApplyDecision:
    def test_match_beyond_range_end_rejected(self) -> None:
        """A marker resolving past the range end would invert the range;
        it must be rejected instead of producing (start > end)."""
        readjuster = _make_readjuster(context_window=3)
        raw_lines = _make_raw_lines(200)
        raw_lines[109] = "UNIQUE-BOUNDARY-MARKER here\n"  # line 110

        result = readjuster._validate_and_apply_decision(
            decision=_decision("UNIQUE-BOUNDARY-MARKER"),
            raw_lines=raw_lines,
            context_window=(94, 112),
            fallback_range=(100, 105),  # end 105 < matched line 110
        )
        assert result is None

    def test_match_within_range_applied(self) -> None:
        readjuster = _make_readjuster(context_window=3)
        raw_lines = _make_raw_lines(200)
        raw_lines[101] = "UNIQUE-BOUNDARY-MARKER here\n"  # line 102

        result = readjuster._validate_and_apply_decision(
            decision=_decision("UNIQUE-BOUNDARY-MARKER"),
            raw_lines=raw_lines,
            context_window=(94, 112),
            fallback_range=(100, 160),
        )
        assert result == (102, 160)


# ---------------------------------------------------------------------------
# Window generation: geometric growth sized by the retry budgets
# ---------------------------------------------------------------------------


class TestGenerateWindows:
    def test_geometric_growth_and_budget_supply(self) -> None:
        """Radii double per window; the supply covers base + expansion budget
        + low-certainty budget (both counters advance windows)."""
        readjuster = _make_readjuster(
            context_window=3,
            retry_config={
                "max_context_expansion_attempts": 3,
                "max_low_certainty_retries": 3,
            },
        )
        windows = list(readjuster._generate_windows((1000, 1200), 100_000))

        assert len(windows) == 7  # 1 + 3 + 3
        radii = [(1000 - s, e - 1000) for s, e in windows]
        assert radii == [(r, r) for r in (3, 6, 12, 24, 48, 96, 192)]

    def test_stops_early_at_full_document(self) -> None:
        readjuster = _make_readjuster(
            context_window=3,
            retry_config={
                "max_context_expansion_attempts": 10,
                "max_low_certainty_retries": 10,
            },
        )
        windows = list(readjuster._generate_windows((10, 15), 20))

        assert windows == [(7, 13), (4, 16), (1, 20)]

    def test_configured_expansion_budget_is_reachable(self) -> None:
        """With the user's config (6 expansions), at least 7 windows exist in
        a large document — the budget is no longer dead."""
        readjuster = _make_readjuster(
            context_window=6,
            retry_config={
                "max_context_expansion_attempts": 6,
                "max_low_certainty_retries": 15,
            },
        )
        windows = list(readjuster._generate_windows((5000, 5200), 1_000_000))
        assert len(windows) >= 7


# ---------------------------------------------------------------------------
# Gap enforcement: deleted spans stay unassigned
# ---------------------------------------------------------------------------


class TestEnforceMaxGapDeletedSpans:
    def _readjuster(self) -> LineRangeReadjuster:
        return _make_readjuster(retry_config={"max_gap_between_ranges": 2})

    def test_deleted_span_not_reabsorbed(self) -> None:
        """A gap created by deleting a no-content range must not be filled by
        extending the previous range."""
        readjuster = self._readjuster()
        survivors = [(1, 100), (201, 300)]
        result = readjuster._enforce_max_gap(survivors, [(101, 200)])
        assert result == [(1, 100), (201, 300)]

    def test_ordinary_gap_still_enforced(self) -> None:
        readjuster = self._readjuster()
        result = readjuster._enforce_max_gap([(1, 100), (201, 300)], [])
        assert result == [(1, 198), (201, 300)]

    def test_extension_capped_before_deleted_span(self) -> None:
        """Extension may cover adjustment slack but stops at a deleted span."""
        readjuster = self._readjuster()
        result = readjuster._enforce_max_gap([(1, 100), (201, 300)], [(150, 200)])
        assert result == [(1, 149), (201, 300)]


# ---------------------------------------------------------------------------
# Overlap removal: single-range sanitization
# ---------------------------------------------------------------------------


class TestRemoveOverlapsSingleRange:
    def test_single_inverted_range_clamped(self) -> None:
        """The single-range early return used to write inverted ranges to
        disk verbatim; they must be clamped like any other range."""
        result = LineRangeReadjuster._remove_overlaps([(450, 160)])
        assert result == [(450, 450)]

    def test_single_valid_range_passthrough(self) -> None:
        assert LineRangeReadjuster._remove_overlaps([(10, 60)]) == [(10, 60)]


# ---------------------------------------------------------------------------
# Chunk slicing: absolute indices, no truncation, no premature finalization
# ---------------------------------------------------------------------------


def _slicing_env(tmp_path: Path) -> tuple[Path, Path]:
    text_file = tmp_path / "sample.txt"
    text_file.write_text(
        "\n".join(f"Line {i}" for i in range(1, 31)) + "\n",
        encoding="utf-8",
    )
    lr_file = tmp_path / "sample_line_ranges.txt"
    lr_file.write_text("(1, 10)\n(11, 20)\n(21, 30)\n", encoding="utf-8")
    return text_file, lr_file


class TestChunkSlicing:
    @pytest.mark.asyncio
    async def test_last_n_uses_absolute_ids_and_keeps_all_ranges(
        self, tmp_path: Path
    ) -> None:
        """--last-n-chunks 1 on a 3-range file: only range 3 is processed,
        its custom_id is absolute (range-3, not range-1), the ranges file
        keeps all 3 ranges, and the header is NOT finalized."""
        text_file, lr_file = _slicing_env(tmp_path)
        readjuster = _make_readjuster()

        processed: list[int] = []

        async def mock_process_range(**kwargs: Any) -> RangeResult:
            idx = kwargs["range_index"]
            processed.append(idx)
            return _fake_range_result(idx, kwargs["original_range"], (23, 30))

        with (
            patch.object(
                readjuster, "_process_single_range", side_effect=mock_process_range
            ),
            patch("modules.line_ranges.readjuster.ProviderConfig") as mock_provider,
            patch(
                "modules.line_ranges.readjuster.open_extractor",
                new_callable=lambda: _async_noop_context,
            ),
            patch(
                "modules.line_ranges.readjuster.resolve_context_for_readjustment",
                return_value=(None, None),
            ),
        ):
            mock_provider._detect_provider.return_value = "openai"
            mock_provider._get_api_key.return_value = "fake-key"

            adjusted = await readjuster.ensure_adjusted_line_ranges(
                text_file=text_file,
                line_ranges_file=lr_file,
                boundary_type="TestSchema",
                last_n_chunks=1,
            )

        assert processed == [3]
        assert adjusted == [(1, 10), (11, 20), (23, 30)]

        # File keeps all ranges — sliced runs used to truncate it.
        file_ranges = lr_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(file_ranges) == 3

        temp_jsonl = tmp_path / "sample_line_ranges_adjust_temp.jsonl"
        records = list(read_jsonl_records(temp_jsonl))
        custom_ids = [r["custom_id"] for r in records if "custom_id" in r]
        assert custom_ids == ["sample_line_ranges-range-3"]

        header = read_jsonl_header(temp_jsonl)
        assert header is not None
        assert header.get("completed_at") is None  # partial coverage
        assert header.get("total_ranges") == 3  # full count, not slice count

    @pytest.mark.asyncio
    async def test_full_run_finalizes_with_final_fingerprint(
        self, tmp_path: Path
    ) -> None:
        text_file, lr_file = _slicing_env(tmp_path)
        readjuster = _make_readjuster()

        async def mock_process_range(**kwargs: Any) -> RangeResult:
            idx = kwargs["range_index"]
            original = kwargs["original_range"]
            return _fake_range_result(idx, original, original)

        with (
            patch.object(
                readjuster, "_process_single_range", side_effect=mock_process_range
            ),
            patch("modules.line_ranges.readjuster.ProviderConfig") as mock_provider,
            patch(
                "modules.line_ranges.readjuster.open_extractor",
                new_callable=lambda: _async_noop_context,
            ),
            patch(
                "modules.line_ranges.readjuster.resolve_context_for_readjustment",
                return_value=(None, None),
            ),
        ):
            mock_provider._detect_provider.return_value = "openai"
            mock_provider._get_api_key.return_value = "fake-key"

            await readjuster.ensure_adjusted_line_ranges(
                text_file=text_file,
                line_ranges_file=lr_file,
                boundary_type="TestSchema",
            )

        from modules.infra.jsonl import compute_ranges_fingerprint

        temp_jsonl = tmp_path / "sample_line_ranges_adjust_temp.jsonl"
        header = read_jsonl_header(temp_jsonl)
        assert header is not None
        assert header.get("completed_at") is not None
        # The stored final fingerprint matches the REWRITTEN ranges file.
        assert header.get("final_ranges_fingerprint") == compute_ranges_fingerprint(
            lr_file
        )


# ---------------------------------------------------------------------------
# Prompt payload: sentinel, boundary type, cache-friendly retry guidance
# ---------------------------------------------------------------------------


class TestPromptPayload:
    def _real_prompt_readjuster(self) -> LineRangeReadjuster:
        """Readjuster with the real prompt template (placeholders included)."""
        with patch(
            "modules.line_ranges.readjuster.detect_capabilities",
            return_value=MagicMock(supports_prompt_caching=False),
        ):
            return LineRangeReadjuster(
                {"extraction_model": {"name": "gpt-4o"}},
                context_window=3,
            )

    @pytest.mark.asyncio
    async def test_sentinel_and_boundary_type_and_retry_placement(self) -> None:
        readjuster = self._real_prompt_readjuster()
        raw_lines = _make_raw_lines(50)

        payload = {
            "contains_no_semantic_boundary": False,
            "needs_more_context": False,
            "boundary_already_on_target": True,
            "certainty": 90,
            "semantic_marker": "",
        }
        mock_chunk = AsyncMock(return_value={"output_text": json.dumps(payload)})

        with patch(
            "modules.line_ranges.readjuster.process_text_chunk", mock_chunk
        ):
            await readjuster._run_model(
                extractor=MagicMock(),
                raw_lines=raw_lines,
                original_range=(20, 35),
                context_window=(17, 23),
                window_index=0,
                boundary_type="HistoricalRecipes",
                context="A cookbook.",
                failed_markers=["bad marker one"],
            )

        kwargs = mock_chunk.call_args.kwargs
        user_message = kwargs["text_chunk"]
        system_message = kwargs["system_message"]

        # Sentinel sits directly above the current start line (line 20).
        assert f"{BOUNDARY_SENTINEL}\nfiller line 20" in user_message
        # Boundary type is rendered into the system prompt.
        assert "HistoricalRecipes" in system_message
        assert "{{BOUNDARY_TYPE}}" not in system_message
        # Failed-marker guidance lives in the user message (prompt caching),
        # not in the system prompt.
        assert "bad marker one" in user_message
        assert "bad marker one" not in system_message

    @pytest.mark.asyncio
    async def test_no_sentinel_when_start_outside_window(self) -> None:
        readjuster = self._real_prompt_readjuster()
        raw_lines = _make_raw_lines(3000)

        payload = {
            "contains_no_semantic_boundary": True,
            "needs_more_context": False,
            "boundary_already_on_target": False,
            "certainty": 90,
            "semantic_marker": "",
        }
        mock_chunk = AsyncMock(return_value={"output_text": json.dumps(payload)})

        with patch(
            "modules.line_ranges.readjuster.process_text_chunk", mock_chunk
        ):
            await readjuster._run_model(
                extractor=MagicMock(),
                raw_lines=raw_lines,
                original_range=(100, 2600),
                context_window=(1101, 2100),  # later verify window
                window_index=-2,
                boundary_type="TestSchema",
                context=None,
            )

        assert BOUNDARY_SENTINEL not in mock_chunk.call_args.kwargs["text_chunk"]


# ---------------------------------------------------------------------------
# Token counting: newlines included
# ---------------------------------------------------------------------------


class TestNewlineTokenCounting:
    def test_line_tokens_include_newline(self) -> None:
        """Chunks are joined with newlines downstream, so per-line counts
        must include the newline token or chunks overshoot their target."""
        encoding = tiktoken.get_encoding("cl100k_base")
        line = "some recipe text with several words"
        per_line = len(encoding.encode(line + "\n"))

        # Chunk budget fits exactly three lines WITH newlines; without them
        # (the old undercount) a fourth line would slip in.
        lines = [line] * 6
        strategy = TokenBasedChunking(
            tokens_per_chunk=per_line * 3,
            model_name="unknown-model-cl100k-fallback",
            text_processor=TextProcessor(),
        )
        ranges = strategy.get_line_ranges(lines)
        assert ranges == [(1, 3), (4, 6)]

    def test_empty_lines_count_one_token(self) -> None:
        encoding = tiktoken.get_encoding("cl100k_base")
        assert len(encoding.encode("\n")) == 1

        strategy = TokenBasedChunking(
            tokens_per_chunk=3,
            model_name="unknown-model-cl100k-fallback",
            text_processor=TextProcessor(),
        )
        ranges = strategy.get_line_ranges(["", "", "", "", "", ""])
        assert ranges == [(1, 3), (4, 6)]
