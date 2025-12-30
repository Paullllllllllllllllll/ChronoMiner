from __future__ import annotations

from pathlib import Path

import pytest

from modules.core.token_tracker import DailyTokenTracker


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
