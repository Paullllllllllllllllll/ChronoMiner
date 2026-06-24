from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import modules.infra.token_tracker as _tt_module
from modules.infra.token_tracker import (
    DailyTokenTracker,
    check_and_wait_for_token_limit,
    check_token_limit_enabled,
)

# Captured at import time, before the autouse _reset_token_tracker fixture
# rebinds the module attribute to a tmp-path lambda.
_REAL_DEFAULT_STATE_FILE = _tt_module._default_token_tracker_file


@pytest.mark.unit
def test_default_state_file_resolves_cwd_at_call_time(tmp_path, monkeypatch):
    """Regression (hygiene): the default state-file directory must be resolved
    when requested, not anchored to the cwd at import time."""
    monkeypatch.chdir(tmp_path)
    resolved = _REAL_DEFAULT_STATE_FILE()
    assert resolved == tmp_path / _tt_module._TOKEN_TRACKER_FILENAME


class TestTokenTrackerPersistence:
    """Test token tracker state persistence."""

    @pytest.mark.unit
    def test_persists_state(self, tmp_path: Path):
        """Test that token state is persisted across instances."""
        state = tmp_path / "state.json"
        t1 = DailyTokenTracker(daily_limit=100, enabled=True, state_file=state)
        t1.add_tokens(10)

        t2 = DailyTokenTracker(daily_limit=100, enabled=True, state_file=state)
        assert t2.get_tokens_used_today() == 10

    @pytest.mark.unit
    def test_persists_multiple_updates(self, tmp_path: Path):
        """Test multiple token additions are persisted."""
        state = tmp_path / "state.json"
        t1 = DailyTokenTracker(daily_limit=1000, enabled=True, state_file=state)
        t1.add_tokens(10)
        t1.add_tokens(20)
        t1.add_tokens(30)

        t2 = DailyTokenTracker(daily_limit=1000, enabled=True, state_file=state)
        assert t2.get_tokens_used_today() == 60

    @pytest.mark.unit
    def test_handles_missing_state_file(self, tmp_path: Path):
        """Test initialization with no existing state file."""
        state = tmp_path / "nonexistent.json"
        t = DailyTokenTracker(daily_limit=100, enabled=True, state_file=state)

        assert t.get_tokens_used_today() == 0


class TestTokenTrackerDailyReset:
    """Test daily reset behavior."""

    @pytest.mark.unit
    def test_resets_on_new_day(self, tmp_path: Path, monkeypatch):
        """Test that tokens reset on a new day."""
        state = tmp_path / "state.json"
        t = DailyTokenTracker(daily_limit=100, enabled=True, state_file=state)
        t._current_date = "2000-01-01"
        t._tokens_used_today = 50

        monkeypatch.setattr(t, "_get_current_date_str", lambda: "2099-01-01")

        assert t.get_tokens_used_today() == 0

    @pytest.mark.unit
    def test_preserves_same_day(self, tmp_path: Path, monkeypatch):
        """Test that tokens are preserved on the same day."""
        state = tmp_path / "state.json"
        t = DailyTokenTracker(daily_limit=100, enabled=True, state_file=state)

        fixed_date = "2024-01-15"
        monkeypatch.setattr(t, "_get_current_date_str", lambda: fixed_date)
        t._current_date = fixed_date
        t._tokens_used_today = 50

        assert t.get_tokens_used_today() == 50


class TestTokenTrackerLimits:
    """Test token limit checking."""

    @pytest.mark.unit
    def test_can_use_tokens_when_under_limit(self, tmp_path: Path):
        """Test can_use_tokens returns True when under limit."""
        state = tmp_path / "state.json"
        t = DailyTokenTracker(daily_limit=100, enabled=True, state_file=state)
        t._tokens_used_today = 50

        assert t.can_use_tokens() is True

    @pytest.mark.unit
    def test_cannot_use_tokens_when_at_limit(self, tmp_path: Path):
        """Test can_use_tokens returns False when at limit."""
        state = tmp_path / "state.json"
        t = DailyTokenTracker(daily_limit=100, enabled=True, state_file=state)
        t._tokens_used_today = 100

        assert t.can_use_tokens() is False

    @pytest.mark.unit
    def test_is_limit_reached(self, tmp_path: Path):
        """Test is_limit_reached method."""
        state = tmp_path / "state.json"
        t = DailyTokenTracker(daily_limit=100, enabled=True, state_file=state)

        t._tokens_used_today = 50
        assert t.is_limit_reached() is False

        t._tokens_used_today = 100
        assert t.is_limit_reached() is True

    @pytest.mark.unit
    def test_disabled_tracker_always_allows(self, tmp_path: Path):
        """Test disabled tracker always allows tokens."""
        state = tmp_path / "state.json"
        t = DailyTokenTracker(daily_limit=100, enabled=False, state_file=state)
        t._tokens_used_today = 1000  # Way over limit

        assert t.can_use_tokens() is True
        assert t.is_limit_reached() is False


class TestTokenTrackerStats:
    """Test token tracker statistics."""

    @pytest.mark.unit
    def test_get_stats(self, tmp_path: Path):
        """Test get_stats returns correct information."""
        state = tmp_path / "state.json"
        t = DailyTokenTracker(daily_limit=1000, enabled=True, state_file=state)
        t._tokens_used_today = 250

        stats = t.get_stats()

        assert stats["daily_limit"] == 1000
        assert stats["tokens_used_today"] == 250
        assert stats["tokens_remaining"] == 750
        assert stats["usage_percentage"] == 25.0

    @pytest.mark.unit
    def test_get_tokens_remaining(self, tmp_path: Path):
        """Test tokens remaining calculation."""
        state = tmp_path / "state.json"
        t = DailyTokenTracker(daily_limit=100, enabled=True, state_file=state)
        t._tokens_used_today = 30

        assert t.get_tokens_remaining() == 70


# ---------------------------------------------------------------------------
# Extended coverage: get_stats, get_usage_percentage, get_seconds_until_reset,
# get_reset_time, can_use_tokens, is_limit_reached, get_tokens_remaining,
# _load_state error/date-change paths, _save_state error path,
# check_token_limit_enabled, check_and_wait_for_token_limit
# ---------------------------------------------------------------------------


class TestTokenTrackerGetStatsExtended:
    def test_returns_all_keys(self, tmp_path):
        tracker = DailyTokenTracker(
            daily_limit=10000, enabled=True, state_file=tmp_path / "state.json"
        )
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
        tracker = DailyTokenTracker(
            daily_limit=1000, enabled=True, state_file=tmp_path / "state.json"
        )
        tracker.add_tokens(400)
        stats = tracker.get_stats()

        assert stats["tokens_used_today"] == 400
        assert stats["tokens_remaining"] == 600
        assert stats["usage_percentage"] == 40.0
        assert stats["limit_reached"] is False

    def test_stats_limit_reached(self, tmp_path):
        tracker = DailyTokenTracker(
            daily_limit=100, enabled=True, state_file=tmp_path / "state.json"
        )
        tracker.add_tokens(100)
        stats = tracker.get_stats()

        assert stats["limit_reached"] is True
        assert stats["tokens_remaining"] == 0
        assert stats["usage_percentage"] == 100.0


class TestGetUsagePercentage:
    def test_disabled_returns_zero(self, tmp_path):
        tracker = DailyTokenTracker(
            daily_limit=1000, enabled=False, state_file=tmp_path / "s.json"
        )
        assert tracker.get_usage_percentage() == 0.0

    def test_zero_limit_returns_zero(self, tmp_path):
        tracker = DailyTokenTracker(
            daily_limit=0, enabled=True, state_file=tmp_path / "s.json"
        )
        assert tracker.get_usage_percentage() == 0.0

    def test_over_limit(self, tmp_path):
        tracker = DailyTokenTracker(
            daily_limit=100, enabled=True, state_file=tmp_path / "s.json"
        )
        tracker.add_tokens(150)
        assert tracker.get_usage_percentage() == 150.0


class TestGetSecondsUntilReset:
    def test_returns_positive_integer(self, tmp_path):
        tracker = DailyTokenTracker(
            daily_limit=1000, enabled=True, state_file=tmp_path / "s.json"
        )
        seconds = tracker.get_seconds_until_reset()
        assert isinstance(seconds, int)
        assert 0 < seconds <= 86400


class TestGetResetTime:
    def test_returns_future_midnight(self, tmp_path):
        tracker = DailyTokenTracker(
            daily_limit=1000, enabled=True, state_file=tmp_path / "s.json"
        )
        reset_time = tracker.get_reset_time()
        now = datetime.now()
        assert reset_time > now
        assert reset_time.hour == 0
        assert reset_time.minute == 0


class TestCanUseTokensExtended:
    def test_disabled_always_true(self, tmp_path):
        tracker = DailyTokenTracker(
            daily_limit=100, enabled=False, state_file=tmp_path / "s.json"
        )
        assert tracker.can_use_tokens(999999) is True

    def test_with_remaining(self, tmp_path):
        tracker = DailyTokenTracker(
            daily_limit=1000, enabled=True, state_file=tmp_path / "s.json"
        )
        assert tracker.can_use_tokens(500) is True
        assert tracker.can_use_tokens(0) is True

    def test_exceeds_remaining(self, tmp_path):
        tracker = DailyTokenTracker(
            daily_limit=100, enabled=True, state_file=tmp_path / "s.json"
        )
        tracker.add_tokens(90)
        assert tracker.can_use_tokens(20) is False
        assert tracker.can_use_tokens(10) is True

    def test_exact_limit(self, tmp_path):
        tracker = DailyTokenTracker(
            daily_limit=100, enabled=True, state_file=tmp_path / "s.json"
        )
        tracker.add_tokens(100)
        assert tracker.can_use_tokens(0) is False
        assert tracker.can_use_tokens(1) is False


class TestIsLimitReachedExtended:
    def test_disabled_never_reached(self, tmp_path):
        tracker = DailyTokenTracker(
            daily_limit=0, enabled=False, state_file=tmp_path / "s.json"
        )
        assert tracker.is_limit_reached() is False

    def test_below_limit(self, tmp_path):
        tracker = DailyTokenTracker(
            daily_limit=1000, enabled=True, state_file=tmp_path / "s.json"
        )
        tracker.add_tokens(500)
        assert tracker.is_limit_reached() is False

    def test_at_limit(self, tmp_path):
        tracker = DailyTokenTracker(
            daily_limit=100, enabled=True, state_file=tmp_path / "s.json"
        )
        tracker.add_tokens(100)
        assert tracker.is_limit_reached() is True


class TestGetTokensRemainingExtended:
    def test_disabled_returns_limit(self, tmp_path):
        tracker = DailyTokenTracker(
            daily_limit=5000, enabled=False, state_file=tmp_path / "s.json"
        )
        assert tracker.get_tokens_remaining() == 5000

    def test_returns_correct_remaining(self, tmp_path):
        tracker = DailyTokenTracker(
            daily_limit=1000, enabled=True, state_file=tmp_path / "s.json"
        )
        tracker.add_tokens(300)
        assert tracker.get_tokens_remaining() == 700

    def test_clamps_to_zero(self, tmp_path):
        tracker = DailyTokenTracker(
            daily_limit=100, enabled=True, state_file=tmp_path / "s.json"
        )
        tracker.add_tokens(200)
        assert tracker.get_tokens_remaining() == 0


class TestLoadState:
    def test_corrupted_state_file(self, tmp_path):
        state_file = tmp_path / "state.json"
        state_file.write_text("not valid json", encoding="utf-8")

        tracker = DailyTokenTracker(
            daily_limit=1000, enabled=True, state_file=state_file
        )
        assert tracker._tokens_used_today == 0

    def test_different_day_resets(self, tmp_path):
        state_file = tmp_path / "state.json"
        state_file.write_text(
            json.dumps({"date": "2020-01-01", "tokens_used": 5000}),
            encoding="utf-8",
        )

        tracker = DailyTokenTracker(
            daily_limit=10000, enabled=True, state_file=state_file
        )
        assert tracker._tokens_used_today == 0

    def test_same_day_restores(self, tmp_path):
        state_file = tmp_path / "state.json"
        today = datetime.now().strftime("%Y-%m-%d")
        state_file.write_text(
            json.dumps({"date": today, "tokens_used": 3000}),
            encoding="utf-8",
        )

        tracker = DailyTokenTracker(
            daily_limit=10000, enabled=True, state_file=state_file
        )
        assert tracker._tokens_used_today == 3000


class TestSaveState:
    def test_save_state_creates_file(self, tmp_path):
        state_file = tmp_path / "subdir" / "state.json"
        state_file.parent.mkdir(parents=True)

        tracker = DailyTokenTracker(
            daily_limit=1000, enabled=True, state_file=state_file
        )
        tracker.add_tokens(100)

        assert state_file.exists()
        data = json.loads(state_file.read_text(encoding="utf-8"))
        assert data["tokens_used"] == 100

    def test_save_state_error_handled(self, tmp_path):
        tracker = DailyTokenTracker(
            daily_limit=1000, enabled=True, state_file=tmp_path / "s.json"
        )

        with patch.object(Path, "with_suffix", side_effect=OSError("disk full")):
            # Should not raise — errors are logged but swallowed
            tracker._save_state()


class TestCheckAndResetIfNewDay:
    def test_resets_on_new_day(self, tmp_path):
        tracker = DailyTokenTracker(
            daily_limit=1000, enabled=True, state_file=tmp_path / "s.json"
        )
        tracker.add_tokens(500)
        assert tracker._tokens_used_today == 500

        # Simulate date change
        tracker._current_date = "2020-01-01"
        used = tracker.get_tokens_used_today()
        assert used == 0


class TestAddTokens:
    def test_disabled_no_op(self, tmp_path):
        tracker = DailyTokenTracker(
            daily_limit=1000, enabled=False, state_file=tmp_path / "s.json"
        )
        tracker.add_tokens(500)
        assert tracker._tokens_used_today == 0

    def test_negative_tokens_ignored(self, tmp_path):
        tracker = DailyTokenTracker(
            daily_limit=1000, enabled=True, state_file=tmp_path / "s.json"
        )
        tracker.add_tokens(-100)
        assert tracker._tokens_used_today == 0

    def test_zero_tokens_ignored(self, tmp_path):
        tracker = DailyTokenTracker(
            daily_limit=1000, enabled=True, state_file=tmp_path / "s.json"
        )
        tracker.add_tokens(0)
        assert tracker._tokens_used_today == 0


class TestCheckTokenLimitEnabled:
    def test_returns_bool(self):
        result = check_token_limit_enabled()
        assert isinstance(result, bool)


class TestCheckAndWaitForTokenLimit:
    def test_returns_true_when_not_reached(self):
        result = asyncio.run(check_and_wait_for_token_limit())
        assert result is True

    def test_returns_true_when_disabled(self, tmp_path):
        import modules.infra.token_tracker as tt

        tt._tracker_instance = DailyTokenTracker(
            daily_limit=100, enabled=False, state_file=tmp_path / "s.json"
        )
        result = asyncio.run(check_and_wait_for_token_limit())
        assert result is True

    def test_keyboard_interrupt_returns_false(self, tmp_path):
        import modules.infra.token_tracker as tt

        tracker = DailyTokenTracker(
            daily_limit=100, enabled=True, state_file=tmp_path / "s.json"
        )
        tracker.add_tokens(200)
        tt._tracker_instance = tracker

        with patch("asyncio.sleep", side_effect=KeyboardInterrupt):
            result = asyncio.run(check_and_wait_for_token_limit())
        assert result is False

    def test_with_ui_object(self, tmp_path):
        import modules.infra.token_tracker as tt

        tracker = DailyTokenTracker(
            daily_limit=100, enabled=True, state_file=tmp_path / "s.json"
        )
        tracker.add_tokens(200)
        tt._tracker_instance = tracker

        ui = MagicMock()
        with patch("asyncio.sleep", side_effect=KeyboardInterrupt):
            result = asyncio.run(check_and_wait_for_token_limit(ui=ui))
        assert result is False
        ui.print_warning.assert_called()

    def test_limit_resets_during_wait(self, tmp_path):
        import modules.infra.token_tracker as tt

        tracker = DailyTokenTracker(
            daily_limit=100, enabled=True, state_file=tmp_path / "s.json"
        )
        tracker.add_tokens(200)
        tt._tracker_instance = tracker

        call_count = 0

        async def mock_sleep(seconds):
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                tracker._current_date = "2020-01-01"

        with patch("asyncio.sleep", side_effect=mock_sleep):
            result = asyncio.run(check_and_wait_for_token_limit())
        assert result is True


class TestChunkReservation:
    """Chunk-level reservation gate: try_reserve / release / EWMA estimate."""

    @pytest.mark.unit
    def test_disabled_returns_zero_and_never_denies(self, tmp_path):
        t = DailyTokenTracker(
            daily_limit=100,
            enabled=False,
            state_file=tmp_path / "s.json",
            chunk_estimate_seed=10,
        )
        assert t.try_reserve() == 0
        assert t.try_reserve(999_999) == 0
        # release is a no-op when disabled (must not raise)
        t.release(0)

    @pytest.mark.unit
    def test_reserve_within_and_beyond_budget(self, tmp_path):
        t = DailyTokenTracker(
            daily_limit=100,
            enabled=True,
            state_file=tmp_path / "s.json",
            chunk_estimate_seed=10,
        )
        # Seed EWMA is 10, so a bare reservation claims 10.
        assert t.try_reserve() == 10
        # An explicit estimate above the EWMA claims the estimate.
        assert t.try_reserve(50) == 50
        # reserved is now 60; remaining headroom is 40, so a 50 will not fit.
        assert t.try_reserve(50) is None
        # 40 fits exactly, bringing reserved to the limit.
        assert t.try_reserve(40) == 40
        assert t.try_reserve(1) is None

    @pytest.mark.unit
    def test_release_restores_headroom(self, tmp_path):
        t = DailyTokenTracker(
            daily_limit=100,
            enabled=True,
            state_file=tmp_path / "s.json",
            chunk_estimate_seed=10,
        )
        assert t.try_reserve(100) == 100
        assert t.try_reserve(1) is None
        t.release(100)
        assert t.try_reserve(50) == 50

    @pytest.mark.unit
    def test_committed_plus_reserved_never_exceeds_limit(self, tmp_path):
        """Concurrency safety: admission subtracts both committed usage and
        outstanding reservations, so workers cannot collectively overshoot."""
        # smoothing=0 freezes the EWMA at the seed so this test isolates the
        # committed-plus-reserved arithmetic from estimate drift.
        t = DailyTokenTracker(
            daily_limit=100,
            enabled=True,
            state_file=tmp_path / "s.json",
            chunk_estimate_seed=10,
            estimate_smoothing=0.0,
        )
        t.add_tokens(60)  # committed usage
        assert t.try_reserve(30) == 30  # 60 + 30 = 90, fits
        assert t.try_reserve(20) is None  # 90 + 20 = 110, denied
        assert t.try_reserve(10) == 10  # 90 + 10 = 100, exact fit

    @pytest.mark.unit
    def test_add_tokens_updates_ewma(self, tmp_path):
        t = DailyTokenTracker(
            daily_limit=10**9,
            enabled=True,
            state_file=tmp_path / "s.json",
            chunk_estimate_seed=100,
            estimate_smoothing=0.5,
        )
        # EWMA = 0.5 * 1100 + 0.5 * 100 = 600
        t.add_tokens(1100)
        assert t.try_reserve() == 600

    @pytest.mark.unit
    def test_reserve_uses_max_of_estimate_and_ewma(self, tmp_path):
        t = DailyTokenTracker(
            daily_limit=10**9,
            enabled=True,
            state_file=tmp_path / "s.json",
            chunk_estimate_seed=100,
        )
        # Estimate above the EWMA wins.
        assert t.try_reserve(5000) == 5000
        # Estimate below the EWMA floors at the EWMA seed (100).
        assert t.try_reserve(10) == 100
