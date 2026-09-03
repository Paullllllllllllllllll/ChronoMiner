"""A marker that resolves past the range end advances to the next window.

Before this fix the readjuster treated a real-text marker lying beyond the
range end exactly like a hallucinated one: it re-asked the same window with
"do not reuse" guidance up to ``max_marker_mismatch_retries`` times, and the
model, correctly, kept naming the same next boundary. On works with entries
longer than a chunk that burned the whole retry budget on every window.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from modules.line_ranges.readjuster import BoundaryDecision, LineRangeReadjuster

pytestmark = pytest.mark.unit

_RAW_LINES = [f"Entry number {i} begins here" for i in range(1, 41)]
_RETRY_CONFIG: dict[str, Any] = {
    "certainty_threshold": 70,
    "max_low_certainty_retries": 3,
    "max_marker_mismatch_retries": 15,
    "max_context_expansion_attempts": 3,
    "delete_ranges_with_no_content": True,
}


def _make_readjuster() -> LineRangeReadjuster:
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
        # context_window=20 on a 40-line file yields exactly two windows for a
        # range starting at line 5: (1, 25) and then the whole document.
        return LineRangeReadjuster(
            {"extraction_model": {"name": "gpt-4o"}},
            context_window=20,
            retry_config=_RETRY_CONFIG,
        )


def _payload(marker: str) -> dict[str, Any]:
    return {
        "contains_no_semantic_boundary": False,
        "needs_more_context": False,
        "boundary_already_on_target": False,
        "certainty": 95,
        "semantic_marker": marker,
    }


async def _process(readjuster: LineRangeReadjuster, marker: str) -> tuple[Any, int]:
    calls = 0

    async def fake_run(**_kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return _payload(marker)

    async def fake_verify(**_kwargs: Any) -> tuple[bool, None, list[Any]]:
        return False, None, []

    with (
        patch.object(readjuster, "_run_model", side_effect=fake_run),
        patch.object(readjuster, "_verify_no_content", side_effect=fake_verify),
    ):
        result = await readjuster._process_single_range(
            extractor=object(),  # type: ignore[arg-type]
            raw_lines=_RAW_LINES,
            original_range=(5, 10),
            range_index=1,
            boundary_type="TestSchema",
            context=None,
        )
    return result, calls


@pytest.mark.asyncio
async def test_marker_beyond_range_end_advances_window_not_retry() -> None:
    """One call per window, never the 15-deep mismatch loop, range kept."""
    readjuster = _make_readjuster()
    result, calls = await _process(readjuster, "Entry number 11 begins here")

    assert calls == 2
    assert result.adjusted_range == (5, 10)
    assert result.should_delete is False
    assert [a["decision_type"] for a in result.attempts] == [
        "marker_beyond_end",
        "marker_beyond_end",
    ]
    assert all(a["matched_line"] == 11 for a in result.attempts)


@pytest.mark.asyncio
async def test_marker_inside_range_still_applies() -> None:
    readjuster = _make_readjuster()
    result, calls = await _process(readjuster, "Entry number 3 begins here")

    assert calls == 1
    assert result.adjusted_range == (3, 10)
    assert result.attempts[-1]["decision_type"] == "marker_found"


def test_validate_and_apply_still_rejects_beyond_end() -> None:
    """The legacy validator keeps its contract: a beyond-end match is None."""
    readjuster = _make_readjuster()
    decision = BoundaryDecision.from_payload(_payload("Entry number 11 begins here"))
    assert (
        readjuster._validate_and_apply_decision(
            decision=decision,
            raw_lines=_RAW_LINES,
            context_window=(1, 25),
            fallback_range=(5, 10),
        )
        is None
    )
    assert (
        readjuster._resolve_marker_line(
            decision=decision,
            raw_lines=_RAW_LINES,
            context_window=(1, 25),
            fallback_range=(5, 10),
        )
        == 11
    )
