"""Extended tests for modules/core/token_tracker.py.

Covers the methods and paths not exercised by the existing test_token_tracker.py:
- get_stats, get_usage_percentage, get_seconds_until_reset, get_reset_time
- can_use_tokens, is_limit_reached, get_tokens_remaining
- _load_state error/date-change paths, _save_state error path
- check_token_limit_enabled, check_and_wait_for_token_limit
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from modules.core.token_tracker import (
    DailyTokenTracker,
    check_and_wait_for_token_limit,
    check_token_limit_enabled,
    get_token_tracker,
)


# ---------------------------------------------------------------------------
# DailyTokenTracker — additional method coverage
# ---------------------------------------------------------------------------

class TestTokenTrackerGetStats:
    def test_returns_all_keys(self, tmp_path):
        tracker = DailyTokenTracker(daily_limit=10000, enabled=True, state_file=tmp_path / "state.json")
        stats = tracker.get_stats()

        assert "enabled" in stats
        assert "daily_limit" in stats
        assert "tokens_used_today" in stats
        assert "tokens_remaining" in stats
        assert "usage_percentage" in stats
        assert "limit_reached" in stats
        assert "seconds_until_reset" in stats
        assert "reset_time" in stats
        assert "current_date" in stats

    def test_stats_reflect_usage(self, tmp_path):
        tracker = DailyTokenTracker(daily_limit=1000, enabled=True, state_file=tmp_path / "state.json")
        tracker.add_tokens(400)
        stats = tracker.get_stats()

        assert stats["tokens_used_today"] == 400
        assert stats["tokens_remaining"] == 600
        assert stats["usage_percentage"] == 40.0
        assert stats["limit_reached"] is False

    def test_stats_limit_reached(self, tmp_path):
        tracker = DailyTokenTracker(daily_limit=100, enabled=True, state_file=tmp_path / "state.json")
        tracker.add_tokens(100)
        stats = tracker.get_stats()

        assert stats["limit_reached"] is True
        assert stats["tokens_remaining"] == 0
        assert stats["usage_percentage"] == 100.0


class TestGetUsagePercentage:
    def test_disabled_returns_zero(self, tmp_path):
        tracker = DailyTokenTracker(daily_limit=1000, enabled=False, state_file=tmp_path / "s.json")
        assert tracker.get_usage_percentage() == 0.0

    def test_zero_limit_returns_zero(self, tmp_path):
        tracker = DailyTokenTracker(daily_limit=0, enabled=True, state_file=tmp_path / "s.json")
        assert tracker.get_usage_percentage() == 0.0

    def test_over_limit(self, tmp_path):
        tracker = DailyTokenTracker(daily_limit=100, enabled=True, state_file=tmp_path / "s.json")
        tracker.add_tokens(150)
        assert tracker.get_usage_percentage() == 150.0


class TestGetSecondsUntilReset:
    def test_returns_positive_integer(self, tmp_path):
        tracker = DailyTokenTracker(daily_limit=1000, enabled=True, state_file=tmp_path / "s.json")
        seconds = tracker.get_seconds_until_reset()
        assert isinstance(seconds, int)
        assert 0 < seconds <= 86400


class TestGetResetTime:
    def test_returns_future_midnight(self, tmp_path):
        tracker = DailyTokenTracker(daily_limit=1000, enabled=True, state_file=tmp_path / "s.json")
        reset_time = tracker.get_reset_time()
        now = datetime.now()
        assert reset_time > now
        assert reset_time.hour == 0
        assert reset_time.minute == 0


class TestCanUseTokens:
    def test_disabled_always_true(self, tmp_path):
        tracker = DailyTokenTracker(daily_limit=100, enabled=False, state_file=tmp_path / "s.json")
        assert tracker.can_use_tokens(999999) is True

    def test_with_remaining(self, tmp_path):
        tracker = DailyTokenTracker(daily_limit=1000, enabled=True, state_file=tmp_path / "s.json")
        assert tracker.can_use_tokens(500) is True
        assert tracker.can_use_tokens(0) is True

    def test_exceeds_remaining(self, tmp_path):
        tracker = DailyTokenTracker(daily_limit=100, enabled=True, state_file=tmp_path / "s.json")
        tracker.add_tokens(90)
        assert tracker.can_use_tokens(20) is False
        assert tracker.can_use_tokens(10) is True

    def test_exact_limit(self, tmp_path):
        tracker = DailyTokenTracker(daily_limit=100, enabled=True, state_file=tmp_path / "s.json")
        tracker.add_tokens(100)
        assert tracker.can_use_tokens(0) is False
        assert tracker.can_use_tokens(1) is False


class TestIsLimitReached:
    def test_disabled_never_reached(self, tmp_path):
        tracker = DailyTokenTracker(daily_limit=0, enabled=False, state_file=tmp_path / "s.json")
        assert tracker.is_limit_reached() is False

    def test_below_limit(self, tmp_path):
        tracker = DailyTokenTracker(daily_limit=1000, enabled=True, state_file=tmp_path / "s.json")
        tracker.add_tokens(500)
        assert tracker.is_limit_reached() is False

    def test_at_limit(self, tmp_path):
        tracker = DailyTokenTracker(daily_limit=100, enabled=True, state_file=tmp_path / "s.json")
        tracker.add_tokens(100)
        assert tracker.is_limit_reached() is True


class TestGetTokensRemaining:
    def test_disabled_returns_limit(self, tmp_path):
        tracker = DailyTokenTracker(daily_limit=5000, enabled=False, state_file=tmp_path / "s.json")
        assert tracker.get_tokens_remaining() == 5000

    def test_returns_correct_remaining(self, tmp_path):
        tracker = DailyTokenTracker(daily_limit=1000, enabled=True, state_file=tmp_path / "s.json")
        tracker.add_tokens(300)
        assert tracker.get_tokens_remaining() == 700

    def test_clamps_to_zero(self, tmp_path):
        tracker = DailyTokenTracker(daily_limit=100, enabled=True, state_file=tmp_path / "s.json")
        tracker.add_tokens(200)
        assert tracker.get_tokens_remaining() == 0


# ---------------------------------------------------------------------------
# _load_state edge cases
# ---------------------------------------------------------------------------

class TestLoadState:
    def test_corrupted_state_file(self, tmp_path):
        state_file = tmp_path / "state.json"
        state_file.write_text("not valid json", encoding="utf-8")

        tracker = DailyTokenTracker(daily_limit=1000, enabled=True, state_file=state_file)
        assert tracker._tokens_used_today == 0

    def test_different_day_resets(self, tmp_path):
        state_file = tmp_path / "state.json"
        state_file.write_text(
            json.dumps({"date": "2020-01-01", "tokens_used": 5000}),
            encoding="utf-8",
        )

        tracker = DailyTokenTracker(daily_limit=10000, enabled=True, state_file=state_file)
        assert tracker._tokens_used_today == 0

    def test_same_day_restores(self, tmp_path):
        state_file = tmp_path / "state.json"
        today = datetime.now().strftime("%Y-%m-%d")
        state_file.write_text(
            json.dumps({"date": today, "tokens_used": 3000}),
            encoding="utf-8",
        )

        tracker = DailyTokenTracker(daily_limit=10000, enabled=True, state_file=state_file)
        assert tracker._tokens_used_today == 3000


# ---------------------------------------------------------------------------
# _save_state error path
# ---------------------------------------------------------------------------

class TestSaveState:
    def test_save_state_creates_file(self, tmp_path):
        state_file = tmp_path / "subdir" / "state.json"
        state_file.parent.mkdir(parents=True)

        tracker = DailyTokenTracker(daily_limit=1000, enabled=True, state_file=state_file)
        tracker.add_tokens(100)

        assert state_file.exists()
        data = json.loads(state_file.read_text(encoding="utf-8"))
        assert data["tokens_used"] == 100

    def test_save_state_error_handled(self, tmp_path):
        tracker = DailyTokenTracker(daily_limit=1000, enabled=True, state_file=tmp_path / "s.json")

        with patch.object(Path, "with_suffix", side_effect=OSError("disk full")):
            # Should not raise — errors are logged but swallowed
            tracker._save_state()


# ---------------------------------------------------------------------------
# _check_and_reset_if_new_day
# ---------------------------------------------------------------------------

class TestCheckAndResetIfNewDay:
    def test_resets_on_new_day(self, tmp_path):
        tracker = DailyTokenTracker(daily_limit=1000, enabled=True, state_file=tmp_path / "s.json")
        tracker.add_tokens(500)
        assert tracker._tokens_used_today == 500

        # Simulate date change
        tracker._current_date = "2020-01-01"
        used = tracker.get_tokens_used_today()
        assert used == 0


# ---------------------------------------------------------------------------
# add_tokens edge cases
# ---------------------------------------------------------------------------

class TestAddTokens:
    def test_disabled_no_op(self, tmp_path):
        tracker = DailyTokenTracker(daily_limit=1000, enabled=False, state_file=tmp_path / "s.json")
        tracker.add_tokens(500)
        assert tracker._tokens_used_today == 0

    def test_negative_tokens_ignored(self, tmp_path):
        tracker = DailyTokenTracker(daily_limit=1000, enabled=True, state_file=tmp_path / "s.json")
        tracker.add_tokens(-100)
        assert tracker._tokens_used_today == 0

    def test_zero_tokens_ignored(self, tmp_path):
        tracker = DailyTokenTracker(daily_limit=1000, enabled=True, state_file=tmp_path / "s.json")
        tracker.add_tokens(0)
        assert tracker._tokens_used_today == 0


# ---------------------------------------------------------------------------
# Module-level functions
# ---------------------------------------------------------------------------

class TestCheckTokenLimitEnabled:
    def test_returns_bool(self):
        result = check_token_limit_enabled()
        assert isinstance(result, bool)


class TestCheckAndWaitForTokenLimit:
    def test_returns_true_when_not_reached(self):
        result = check_and_wait_for_token_limit()
        assert result is True

    def test_returns_true_when_disabled(self, tmp_path):
        import modules.core.token_tracker as tt
        tt._tracker_instance = DailyTokenTracker(
            daily_limit=100, enabled=False, state_file=tmp_path / "s.json"
        )
        result = check_and_wait_for_token_limit()
        assert result is True

    def test_keyboard_interrupt_returns_false(self, tmp_path):
        import modules.core.token_tracker as tt
        tracker = DailyTokenTracker(
            daily_limit=100, enabled=True, state_file=tmp_path / "s.json"
        )
        tracker.add_tokens(200)  # exceed limit
        tt._tracker_instance = tracker

        with patch("time.sleep", side_effect=KeyboardInterrupt):
            result = check_and_wait_for_token_limit()
        assert result is False

    def test_with_ui_object(self, tmp_path):
        import modules.core.token_tracker as tt
        tracker = DailyTokenTracker(
            daily_limit=100, enabled=True, state_file=tmp_path / "s.json"
        )
        tracker.add_tokens(200)
        tt._tracker_instance = tracker

        ui = MagicMock()
        with patch("time.sleep", side_effect=KeyboardInterrupt):
            result = check_and_wait_for_token_limit(ui=ui)
        assert result is False
        ui.print_warning.assert_called()

    def test_limit_resets_during_wait(self, tmp_path):
        import modules.core.token_tracker as tt
        tracker = DailyTokenTracker(
            daily_limit=100, enabled=True, state_file=tmp_path / "s.json"
        )
        tracker.add_tokens(200)
        tt._tracker_instance = tracker

        call_count = 0

        def mock_sleep(seconds):
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                # Simulate day change by resetting tracker
                tracker._current_date = "2020-01-01"

        with patch("time.sleep", side_effect=mock_sleep):
            result = check_and_wait_for_token_limit()
        assert result is True
