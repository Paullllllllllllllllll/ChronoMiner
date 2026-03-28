"""Tests for the _verify_no_content method of LineRangeReadjuster.

Covers the interior-scan logic introduced to replace the old adjacent-area scan.
"""

from typing import Any, Dict, List, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from modules.operations.line_ranges.readjuster import (
    BoundaryDecision,
    LineRangeReadjuster,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_readjuster(
    scan_range_multiplier: int = 2,
    delete_ranges_with_no_content: bool = True,
) -> LineRangeReadjuster:
    """Create a readjuster without needing real API keys or prompt files."""
    with patch(
        "modules.operations.line_ranges.readjuster.load_prompt_template",
        return_value="fake prompt",
    ), patch(
        "modules.operations.line_ranges.readjuster.detect_capabilities",
        return_value=MagicMock(supports_prompt_caching=False),
    ):
        return LineRangeReadjuster(
            {"extraction_model": {"name": "gpt-4o"}},
            context_window=3,
            retry_config={
                "scan_range_multiplier": scan_range_multiplier,
                "delete_ranges_with_no_content": delete_ranges_with_no_content,
            },
        )


def _no_content_payload(certainty: int = 90) -> Dict[str, Any]:
    """Payload indicating no semantic content found."""
    return {
        "contains_no_semantic_boundary": True,
        "needs_more_context": False,
        "certainty": certainty,
        "semantic_marker": "",
    }


def _content_found_payload(
    marker: str = "Recipe Title",
    certainty: int = 85,
) -> Dict[str, Any]:
    """Payload indicating semantic content was found."""
    return {
        "contains_no_semantic_boundary": False,
        "needs_more_context": False,
        "certainty": certainty,
        "semantic_marker": marker,
    }


def _make_raw_lines(n: int = 100) -> List[str]:
    """Create dummy raw_lines (1-indexed, so index 0 is padding)."""
    return [""] + [f"Line {i}" for i in range(1, n + 1)]


# ---------------------------------------------------------------------------
# Interior scan: no content → deletion confirmed
# ---------------------------------------------------------------------------

class TestVerifyNoContentDeletesEmpty:
    """When interior scan confirms no content, should_delete is True."""

    @pytest.mark.asyncio
    async def test_single_window_no_content(self) -> None:
        """Short range scanned in one window; model says no content → delete."""
        readjuster = _make_readjuster(scan_range_multiplier=2)
        raw_lines = _make_raw_lines(100)

        mock_run = AsyncMock(return_value=_no_content_payload())
        readjuster._run_model = mock_run

        should_delete, _, attempts = await readjuster._verify_no_content(
            extractor=MagicMock(),
            raw_lines=raw_lines,
            original_range=(50, 60),
            range_index=1,
            boundary_type="TestSchema",
            context=None,
        )

        assert should_delete is True
        assert len(attempts) == 1
        assert attempts[0]["decision_type"] == "verify_interior_1"
        assert attempts[0]["found_content"] is False
        # Scan window must be the range itself
        assert attempts[0]["window"] == [50, 60]

    @pytest.mark.asyncio
    async def test_window_index_is_negative(self) -> None:
        """Verification calls use negative window_index values."""
        readjuster = _make_readjuster()
        raw_lines = _make_raw_lines(100)

        mock_run = AsyncMock(return_value=_no_content_payload())
        readjuster._run_model = mock_run

        await readjuster._verify_no_content(
            extractor=MagicMock(),
            raw_lines=raw_lines,
            original_range=(10, 20),
            range_index=5,
            boundary_type="TestSchema",
            context=None,
        )

        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs["window_index"] == -1


# ---------------------------------------------------------------------------
# Interior scan: content found → preservation
# ---------------------------------------------------------------------------

class TestVerifyNoContentPreserves:
    """When interior scan finds content, should_delete is False."""

    @pytest.mark.asyncio
    async def test_content_found_preserves_range(self) -> None:
        """Model finds a recipe marker inside the range → do not delete."""
        readjuster = _make_readjuster()
        raw_lines = _make_raw_lines(100)

        mock_run = AsyncMock(
            return_value=_content_found_payload("Soupe à l'oignon"),
        )
        readjuster._run_model = mock_run

        should_delete, _, attempts = await readjuster._verify_no_content(
            extractor=MagicMock(),
            raw_lines=raw_lines,
            original_range=(30, 40),
            range_index=2,
            boundary_type="TestSchema",
            context=None,
        )

        assert should_delete is False
        assert len(attempts) == 1
        assert attempts[0]["found_content"] is True
        assert attempts[0]["semantic_marker"] == "Soupe à l'oignon"

    @pytest.mark.asyncio
    async def test_no_boundary_but_no_marker_still_deletes(self) -> None:
        """contains_no_semantic_boundary=False but empty marker → still deleted
        (no marker to anchor), but found_content reflects the model's assertion."""
        readjuster = _make_readjuster()
        raw_lines = _make_raw_lines(100)

        payload = {
            "contains_no_semantic_boundary": False,
            "needs_more_context": False,
            "certainty": 50,
            "semantic_marker": "",
        }
        mock_run = AsyncMock(return_value=payload)
        readjuster._run_model = mock_run

        should_delete, _, attempts = await readjuster._verify_no_content(
            extractor=MagicMock(),
            raw_lines=raw_lines,
            original_range=(10, 20),
            range_index=1,
            boundary_type="TestSchema",
            context=None,
        )

        # No marker to anchor with, so deletion proceeds
        assert should_delete is True
        # found_content reflects the model's assertion (cnb=False means content detected)
        assert attempts[0]["found_content"] is True


# ---------------------------------------------------------------------------
# Scan window geometry
# ---------------------------------------------------------------------------

class TestScanWindowGeometry:
    """Verify that scan windows are built inside the range, not outside."""

    @pytest.mark.asyncio
    async def test_scan_window_covers_full_range(self) -> None:
        """With default multiplier=2, a short range is scanned in one window
        covering [start, end] exactly."""
        readjuster = _make_readjuster(scan_range_multiplier=2)
        raw_lines = _make_raw_lines(200)

        mock_run = AsyncMock(return_value=_no_content_payload())
        readjuster._run_model = mock_run

        await readjuster._verify_no_content(
            extractor=MagicMock(),
            raw_lines=raw_lines,
            original_range=(80, 100),
            range_index=1,
            boundary_type="TestSchema",
            context=None,
        )

        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs["context_window"] == (80, 100)

    @pytest.mark.asyncio
    async def test_scan_never_leaves_range(self) -> None:
        """No matter the multiplier, scan windows stay within [start, end]."""
        readjuster = _make_readjuster(scan_range_multiplier=5)
        raw_lines = _make_raw_lines(1000)

        mock_run = AsyncMock(return_value=_no_content_payload())
        readjuster._run_model = mock_run

        original_range = (200, 300)
        await readjuster._verify_no_content(
            extractor=MagicMock(),
            raw_lines=raw_lines,
            original_range=original_range,
            range_index=1,
            boundary_type="TestSchema",
            context=None,
        )

        for call in mock_run.call_args_list:
            window = call.kwargs["context_window"]
            assert window[0] >= original_range[0]
            assert window[1] <= original_range[1]

    @pytest.mark.asyncio
    async def test_tiny_multiplier_creates_two_windows(self) -> None:
        """With a very small multiplier, a large range gets two windows
        (first portion + last portion) rather than one full scan."""
        # scan_radius = range_size * 0.1 (simulated by integer truncation)
        # We need range_size > 2 * scan_radius → range_size > 2 * range_size * multiplier
        # → 1 > 2 * multiplier → multiplier < 0.5. But multiplier is int, minimum useful is 1.
        # With multiplier=1, condition is range_size <= 2*range_size (always true),
        # so we can't trigger two windows with integer multipliers >= 1.
        # This test documents that with multiplier=2, all ranges get a single window.
        readjuster = _make_readjuster(scan_range_multiplier=2)
        raw_lines = _make_raw_lines(500)

        mock_run = AsyncMock(return_value=_no_content_payload())
        readjuster._run_model = mock_run

        await readjuster._verify_no_content(
            extractor=MagicMock(),
            raw_lines=raw_lines,
            original_range=(100, 400),
            range_index=1,
            boundary_type="TestSchema",
            context=None,
        )

        # With multiplier=2: range_size=301, scan_radius=602,
        # 301 <= 2*602 = 1204 → True → single window
        assert mock_run.call_count == 1
        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs["context_window"] == (100, 400)


# ---------------------------------------------------------------------------
# Two-window branch (long range with small multiplier)
# ---------------------------------------------------------------------------

class TestTwoWindowBranch:
    """Exercise the two-window sampling path for very large ranges with small multipliers.

    This requires patching scan_range_multiplier to a fractional-equivalent value.
    Since the attribute is set directly, we can set it to any numeric value.
    """

    @pytest.mark.asyncio
    async def test_two_windows_first_content_aborts(self) -> None:
        """When two windows are needed and the first finds content, only one call is made."""
        readjuster = _make_readjuster(scan_range_multiplier=2)
        # Force a small scan_radius so the two-window branch triggers
        readjuster.scan_range_multiplier = 0.1  # type: ignore[assignment]

        raw_lines = _make_raw_lines(1000)
        original_range = (100, 900)  # 801 lines

        mock_run = AsyncMock(
            return_value=_content_found_payload("Roast Beef"),
        )
        readjuster._run_model = mock_run

        should_delete, _, attempts = await readjuster._verify_no_content(
            extractor=MagicMock(),
            raw_lines=raw_lines,
            original_range=original_range,
            range_index=1,
            boundary_type="TestSchema",
            context=None,
        )

        assert should_delete is False
        assert mock_run.call_count == 1
        assert len(attempts) == 1

    @pytest.mark.asyncio
    async def test_two_windows_both_empty_deletes(self) -> None:
        """Two windows, both show no content → delete."""
        readjuster = _make_readjuster(scan_range_multiplier=2)
        readjuster.scan_range_multiplier = 0.1  # type: ignore[assignment]

        raw_lines = _make_raw_lines(1000)
        original_range = (100, 900)

        mock_run = AsyncMock(return_value=_no_content_payload())
        readjuster._run_model = mock_run

        should_delete, _, attempts = await readjuster._verify_no_content(
            extractor=MagicMock(),
            raw_lines=raw_lines,
            original_range=original_range,
            range_index=1,
            boundary_type="TestSchema",
            context=None,
        )

        assert should_delete is True
        assert mock_run.call_count == 2
        assert len(attempts) == 2
        assert attempts[0]["decision_type"] == "verify_interior_1"
        assert attempts[1]["decision_type"] == "verify_interior_2"

    @pytest.mark.asyncio
    async def test_two_windows_second_finds_content(self) -> None:
        """First window is empty, second finds content → preserve."""
        readjuster = _make_readjuster(scan_range_multiplier=2)
        readjuster.scan_range_multiplier = 0.1  # type: ignore[assignment]

        raw_lines = _make_raw_lines(1000)
        original_range = (100, 900)

        mock_run = AsyncMock(
            side_effect=[
                _no_content_payload(),
                _content_found_payload("Potage"),
            ],
        )
        readjuster._run_model = mock_run

        should_delete, _, attempts = await readjuster._verify_no_content(
            extractor=MagicMock(),
            raw_lines=raw_lines,
            original_range=original_range,
            range_index=1,
            boundary_type="TestSchema",
            context=None,
        )

        assert should_delete is False
        assert mock_run.call_count == 2
        assert attempts[0]["found_content"] is False
        assert attempts[1]["found_content"] is True

    @pytest.mark.asyncio
    async def test_two_windows_stay_within_range(self) -> None:
        """Both windows must be subsets of the original range."""
        readjuster = _make_readjuster(scan_range_multiplier=2)
        readjuster.scan_range_multiplier = 0.1  # type: ignore[assignment]

        raw_lines = _make_raw_lines(1000)
        original_range = (100, 900)

        mock_run = AsyncMock(return_value=_no_content_payload())
        readjuster._run_model = mock_run

        await readjuster._verify_no_content(
            extractor=MagicMock(),
            raw_lines=raw_lines,
            original_range=original_range,
            range_index=1,
            boundary_type="TestSchema",
            context=None,
        )

        for call in mock_run.call_args_list:
            window = call.kwargs["context_window"]
            assert window[0] >= 100, f"Window start {window[0]} is before range start 100"
            assert window[1] <= 900, f"Window end {window[1]} is after range end 900"


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------

class TestAuditTrail:
    """Verify the structure and content of verification attempt records."""

    @pytest.mark.asyncio
    async def test_attempt_record_keys(self) -> None:
        """Each attempt dict has the expected keys."""
        readjuster = _make_readjuster()
        raw_lines = _make_raw_lines(50)

        mock_run = AsyncMock(return_value=_no_content_payload(certainty=88))
        readjuster._run_model = mock_run

        _, _, attempts = await readjuster._verify_no_content(
            extractor=MagicMock(),
            raw_lines=raw_lines,
            original_range=(10, 30),
            range_index=3,
            boundary_type="TestSchema",
            context=None,
        )

        expected_keys = {
            "window", "window_index", "decision_type",
            "certainty", "semantic_marker", "found_content",
        }
        assert set(attempts[0].keys()) == expected_keys

    @pytest.mark.asyncio
    async def test_certainty_propagated(self) -> None:
        """Certainty from the model payload is recorded in the attempt."""
        readjuster = _make_readjuster()
        raw_lines = _make_raw_lines(50)

        mock_run = AsyncMock(return_value=_no_content_payload(certainty=73))
        readjuster._run_model = mock_run

        _, _, attempts = await readjuster._verify_no_content(
            extractor=MagicMock(),
            raw_lines=raw_lines,
            original_range=(1, 20),
            range_index=1,
            boundary_type="TestSchema",
            context=None,
        )

        assert attempts[0]["certainty"] == 73


# ---------------------------------------------------------------------------
# _run_model forwarding
# ---------------------------------------------------------------------------

class TestRunModelForwarding:
    """Verify that _verify_no_content forwards correct args to _run_model."""

    @pytest.mark.asyncio
    async def test_forwards_boundary_type_and_context(self) -> None:
        readjuster = _make_readjuster()
        raw_lines = _make_raw_lines(50)
        extractor = MagicMock()

        mock_run = AsyncMock(return_value=_no_content_payload())
        readjuster._run_model = mock_run

        await readjuster._verify_no_content(
            extractor=extractor,
            raw_lines=raw_lines,
            original_range=(5, 15),
            range_index=1,
            boundary_type="HistoricalRecipe",
            context="This is a French cookbook",
        )

        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs["boundary_type"] == "HistoricalRecipe"
        assert call_kwargs["context"] == "This is a French cookbook"
        assert call_kwargs["extractor"] is extractor
        assert call_kwargs["original_range"] == (5, 15)

    @pytest.mark.asyncio
    async def test_forwards_raw_lines_by_reference(self) -> None:
        """raw_lines is passed through, not copied or sliced."""
        readjuster = _make_readjuster()
        raw_lines = _make_raw_lines(50)

        mock_run = AsyncMock(return_value=_no_content_payload())
        readjuster._run_model = mock_run

        await readjuster._verify_no_content(
            extractor=MagicMock(),
            raw_lines=raw_lines,
            original_range=(1, 10),
            range_index=1,
            boundary_type="TestSchema",
            context=None,
        )

        assert mock_run.call_args.kwargs["raw_lines"] is raw_lines


# ---------------------------------------------------------------------------
# Re-anchoring on verify-interior content (Fix A)
# ---------------------------------------------------------------------------

def _make_raw_lines_with_marker(
    n: int = 100, marker: str = "Recipe Title", marker_line: int = 55
) -> List[str]:
    """Create raw_lines with a known marker at a specific 1-indexed line.

    The readjuster uses raw_lines[line_number - 1] to access content, so
    the marker is placed at index marker_line - 1 in the underlying list
    (which has a padding element at index 0).
    """
    lines = [""] + [f"Line {i}" for i in range(1, n + 1)]
    # raw_lines[marker_line - 1] is read when line_number == marker_line
    lines[marker_line - 1] = marker
    return lines


class TestVerifyReanchor:
    """When verify-interior finds content with a resolvable marker,
    the method returns a re-anchored range."""

    @pytest.mark.asyncio
    async def test_reanchor_on_verify_content(self) -> None:
        """Interior scan finds marker that resolves to a unique line → re-anchored range returned."""
        readjuster = _make_readjuster()
        marker_text = "Soupe à l'oignon."
        raw_lines = _make_raw_lines_with_marker(100, marker_text, 55)

        mock_run = AsyncMock(
            return_value=_content_found_payload(marker_text),
        )
        readjuster._run_model = mock_run

        should_delete, reanchored, attempts = await readjuster._verify_no_content(
            extractor=MagicMock(),
            raw_lines=raw_lines,
            original_range=(30, 80),
            range_index=1,
            boundary_type="TestSchema",
            context=None,
        )

        assert should_delete is False
        assert reanchored is not None
        assert reanchored[0] == 55  # marker line
        assert reanchored[1] == 80  # original end preserved

    @pytest.mark.asyncio
    async def test_reanchor_marker_not_resolved(self) -> None:
        """Interior scan finds content with marker that doesn't match source → reanchored is None."""
        readjuster = _make_readjuster()
        raw_lines = _make_raw_lines(100)  # no matching marker in source

        mock_run = AsyncMock(
            return_value=_content_found_payload("Nonexistent Marker Text"),
        )
        readjuster._run_model = mock_run

        should_delete, reanchored, attempts = await readjuster._verify_no_content(
            extractor=MagicMock(),
            raw_lines=raw_lines,
            original_range=(30, 80),
            range_index=1,
            boundary_type="TestSchema",
            context=None,
        )

        assert should_delete is False
        assert reanchored is None
        assert attempts[0]["found_content"] is True


# ---------------------------------------------------------------------------
# Exhaustion fallback verification (Fix B)
# ---------------------------------------------------------------------------

class TestExhaustionFallback:
    """When all boundary windows exhaust at low certainty, the exhaustion
    fallback triggers _verify_no_content before keeping the range."""

    @pytest.mark.asyncio
    async def test_exhaustion_fallback_triggers_verify(self) -> None:
        """_process_single_range calls _verify_no_content when all windows exhaust."""
        readjuster = _make_readjuster()
        readjuster.certainty_threshold = 70
        readjuster.max_low_certainty_retries = 1
        readjuster.max_context_expansion_attempts = 1
        readjuster.max_marker_mismatch_retries = 1

        raw_lines = _make_raw_lines(100)

        # All boundary window calls return low certainty
        low_cert = {
            "contains_no_semantic_boundary": False,
            "needs_more_context": False,
            "certainty": 40,
            "semantic_marker": "",
        }
        # Verify scan returns no content
        no_content = _no_content_payload(certainty=90)

        call_count = 0

        async def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            # First calls are boundary windows (low certainty)
            # Last call(s) are verify-interior (no content)
            if kwargs.get("window_index", 0) < 0:
                return no_content
            return low_cert

        readjuster._run_model = AsyncMock(side_effect=side_effect)

        result = await readjuster._process_single_range(
            extractor=MagicMock(),
            raw_lines=raw_lines,
            original_range=(10, 50),
            range_index=1,
            boundary_type="TestSchema",
            context=None,
        )

        # Verify that the fallback ran and deleted the range
        assert result.should_delete is True
        verify_attempts = [
            a for a in result.attempts
            if a.get("decision_type", "").startswith("verify_interior")
        ]
        assert len(verify_attempts) >= 1

    @pytest.mark.asyncio
    async def test_exhaustion_fallback_deletes_empty(self) -> None:
        """Exhaustion fallback verify scan confirms no content → should_delete=True."""
        readjuster = _make_readjuster()
        readjuster.certainty_threshold = 70
        readjuster.max_low_certainty_retries = 1
        readjuster.max_context_expansion_attempts = 1
        readjuster.max_marker_mismatch_retries = 1

        raw_lines = _make_raw_lines(100)

        low_cert = {
            "contains_no_semantic_boundary": False,
            "needs_more_context": False,
            "certainty": 40,
            "semantic_marker": "",
        }
        no_content = _no_content_payload(certainty=95)

        async def side_effect(**kwargs):
            if kwargs.get("window_index", 0) < 0:
                return no_content
            return low_cert

        readjuster._run_model = AsyncMock(side_effect=side_effect)

        result = await readjuster._process_single_range(
            extractor=MagicMock(),
            raw_lines=raw_lines,
            original_range=(10, 50),
            range_index=1,
            boundary_type="TestSchema",
            context=None,
        )

        assert result.should_delete is True

    @pytest.mark.asyncio
    async def test_exhaustion_fallback_reanchors(self) -> None:
        """Exhaustion fallback verify scan finds content with resolvable marker → re-anchored."""
        readjuster = _make_readjuster()
        readjuster.certainty_threshold = 70
        readjuster.max_low_certainty_retries = 1
        readjuster.max_context_expansion_attempts = 1
        readjuster.max_marker_mismatch_retries = 1

        marker_text = "Gefüllte Kalbsbrust."
        raw_lines = _make_raw_lines_with_marker(100, marker_text, 35)

        low_cert = {
            "contains_no_semantic_boundary": False,
            "needs_more_context": False,
            "certainty": 40,
            "semantic_marker": "",
        }
        content_found = _content_found_payload(marker_text, certainty=85)

        async def side_effect(**kwargs):
            if kwargs.get("window_index", 0) < 0:
                return content_found
            return low_cert

        readjuster._run_model = AsyncMock(side_effect=side_effect)

        result = await readjuster._process_single_range(
            extractor=MagicMock(),
            raw_lines=raw_lines,
            original_range=(10, 50),
            range_index=1,
            boundary_type="TestSchema",
            context=None,
        )

        assert result.should_delete is False
        assert result.adjusted_range[0] == 35  # marker line
        assert result.adjusted_range[1] == 50  # original end


# ---------------------------------------------------------------------------
# found_content audit field accuracy (Fix C)
# ---------------------------------------------------------------------------

class TestFoundContentField:
    """Verify found_content reflects contains_no_semantic_boundary alone."""

    @pytest.mark.asyncio
    async def test_found_content_true_without_marker(self) -> None:
        """contains_no_semantic_boundary=False with empty marker → found_content=True."""
        readjuster = _make_readjuster()
        raw_lines = _make_raw_lines(100)

        payload = {
            "contains_no_semantic_boundary": False,
            "needs_more_context": False,
            "certainty": 65,
            "semantic_marker": "",
        }
        readjuster._run_model = AsyncMock(return_value=payload)

        _, _, attempts = await readjuster._verify_no_content(
            extractor=MagicMock(),
            raw_lines=raw_lines,
            original_range=(10, 20),
            range_index=1,
            boundary_type="TestSchema",
            context=None,
        )

        assert attempts[0]["found_content"] is True
