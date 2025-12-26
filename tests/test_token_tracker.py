from __future__ import annotations

from pathlib import Path

import pytest

from modules.core.token_tracker import DailyTokenTracker


@pytest.mark.unit
def test_token_tracker_persists_state(tmp_path: Path):
    state = tmp_path / "state.json"
    t1 = DailyTokenTracker(daily_limit=100, enabled=True, state_file=state)
    t1.add_tokens(10)

    t2 = DailyTokenTracker(daily_limit=100, enabled=True, state_file=state)
    assert t2.get_tokens_used_today() == 10


@pytest.mark.unit
def test_token_tracker_resets_on_new_day(tmp_path: Path, monkeypatch):
    state = tmp_path / "state.json"
    t = DailyTokenTracker(daily_limit=100, enabled=True, state_file=state)
    t._current_date = "2000-01-01"
    t._tokens_used_today = 50

    monkeypatch.setattr(t, "_get_current_date_str", lambda: "2099-01-01")

    assert t.get_tokens_used_today() == 0
