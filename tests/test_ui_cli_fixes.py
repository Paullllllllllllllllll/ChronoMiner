"""Regression tests for CLI/UI robustness fixes.

Covers two offline hardening fixes:

* The interactive line-range prompt (``ChunkHandler.adjust_line_ranges``)
  degrades gracefully when stdin is unavailable instead of crashing with an
  ``EOFError``.
* ``check_and_wait_for_token_limit`` lets ``asyncio.CancelledError`` propagate
  (so a mid-wait Ctrl+C is not misreported as a token-budget stop).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from modules.infra.chunking import ChunkHandler, TextProcessor
from modules.infra.token_tracker import check_and_wait_for_token_limit


@pytest.mark.unit
def test_adjust_line_ranges_survives_missing_stdin():
    """EOFError from input() is caught: remaining chunks keep their defaults."""
    handler = ChunkHandler(
        model_name="gpt-4o",
        default_tokens_per_chunk=100,
        text_processor=TextProcessor(),
    )
    with patch("builtins.input", side_effect=EOFError):
        result = handler.adjust_line_ranges(
            initial_ranges=[(1, 5)],
            original_start_line=1,
            total_processed_lines=10,
        )
    # No crash; the current default range is kept and the remainder is filled
    # in without further prompting.
    assert result == [(1, 5), (6, 10)]


@pytest.mark.unit
def test_adjust_line_ranges_survives_keyboard_interrupt():
    handler = ChunkHandler(
        model_name="gpt-4o",
        default_tokens_per_chunk=100,
        text_processor=TextProcessor(),
    )
    with patch("builtins.input", side_effect=KeyboardInterrupt):
        result = handler.adjust_line_ranges(
            initial_ranges=[(1, 5)],
            original_start_line=1,
            total_processed_lines=10,
        )
    assert result == [(1, 5), (6, 10)]


@pytest.mark.asyncio
async def test_check_and_wait_propagates_cancellation():
    """A cancellation during the wait must propagate, not return False."""
    fake_tracker = SimpleNamespace(
        enabled=True,
        is_limit_reached=lambda: True,
        get_stats=lambda: {"tokens_used_today": 100, "daily_limit": 100},
        get_reset_time=lambda: datetime.now(UTC) + timedelta(minutes=5),
        get_seconds_until_reset=lambda: 300,
        describe_pool_block=lambda: None,
    )
    with (
        patch(
            "modules.infra.token_tracker.get_token_tracker",
            return_value=fake_tracker,
        ),
        patch("asyncio.sleep", side_effect=asyncio.CancelledError),
        pytest.raises(asyncio.CancelledError),
    ):
        await check_and_wait_for_token_limit(ui=None, logger=None)
