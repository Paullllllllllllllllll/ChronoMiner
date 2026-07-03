"""Tests for the shared cross-tool token-budget integration.

These exercise DailyTokenTracker's optional ``shared_enabled`` path: the
vendored :mod:`modules.infra.shared_ledger` wired in as the daily budget's
persistence and combined-total source. The DISABLED path (default) is covered
by ``tests/test_token_tracker.py`` and must stay bit-for-bit today's behaviour;
here every tracker is constructed with an explicit scratch ledger directory so
nothing touches ``~/.chronopipeline``.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from modules.infra.shared_ledger import LEDGER_FILENAME, SharedTokenLedger
from modules.infra.token_tracker import (
    DailyTokenTracker,
    check_and_wait_for_token_limit,
)


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _write_ledger(ledger_dir: Path, tools: dict[str, int]) -> None:
    """Write the ledger JSON directly (simulates another process/tool)."""
    ledger_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "schema_version": 1,
        "date": _today(),
        "tools": tools,
        "last_updated": datetime.now().isoformat(),
    }
    (ledger_dir / LEDGER_FILENAME).write_text(json.dumps(data), encoding="utf-8")


def _make(
    *,
    ledger_dir: Path,
    state_file: Path,
    daily_limit: int = 10_000_000,
    enabled: bool = True,
    shared_enabled: bool = True,
    seed: int = 1,
    smoothing: float = 0.0,
) -> DailyTokenTracker:
    return DailyTokenTracker(
        daily_limit=daily_limit,
        enabled=enabled,
        state_file=state_file,
        chunk_estimate_seed=seed,
        estimate_smoothing=smoothing,
        shared_enabled=shared_enabled,
        shared_ledger_dir=ledger_dir,
    )


class TestDisabledUnchanged:
    def test_no_ledger_file_and_private_state_used(self, tmp_path: Path) -> None:
        ledger_dir = tmp_path / "ledger"
        state_file = tmp_path / "state.json"
        t = _make(
            ledger_dir=ledger_dir,
            state_file=state_file,
            daily_limit=100,
            shared_enabled=False,
        )
        t.add_tokens(50)
        t.flush()

        # Disabled: zero ledger I/O, private state file carries the count.
        assert not (ledger_dir / LEDGER_FILENAME).exists()
        assert state_file.exists()
        data = json.loads(state_file.read_text(encoding="utf-8"))
        assert data["tokens_used"] == 50
        assert t.get_tokens_used_today() == 50


class TestEnabledPersistence:
    def test_add_then_flush_pushes_delta(self, tmp_path: Path) -> None:
        ledger_dir = tmp_path / "ledger"
        t = _make(ledger_dir=ledger_dir, state_file=tmp_path / "s.json")

        t.add_tokens(1234)
        t.flush()  # forces the accumulated delta into the ledger

        ledger = SharedTokenLedger("chronominer", ledger_dir=ledger_dir)
        assert ledger.read_breakdown() == {"chronominer": 1234}
        assert ledger.read_combined() == 1234
        # Combined view (only this tool present) equals our own usage.
        assert t.get_tokens_used_today() == 1234
        assert t.get_own_tokens_used_today() == 1234

    def test_stats_expose_breakdown(self, tmp_path: Path) -> None:
        ledger_dir = tmp_path / "ledger"
        SharedTokenLedger("chronotranscriber", ledger_dir=ledger_dir).sync(700)
        t = _make(ledger_dir=ledger_dir, state_file=tmp_path / "s.json")
        t.add_tokens(300)
        t.flush()

        stats = t.get_stats()
        assert stats["shared_budget_enabled"] is True
        assert stats["own_tokens_used_today"] == 300
        assert stats["combined_tokens_used_today"] == 1000
        assert stats["shared_breakdown"] == {
            "chronominer": 300,
            "chronotranscriber": 700,
        }


class TestForeignUsageClosesGate:
    def test_combined_total_enforced(self, tmp_path: Path) -> None:
        ledger_dir = tmp_path / "ledger"
        # Another tool has already burned most of the shared daily budget.
        SharedTokenLedger("chronotranscriber", ledger_dir=ledger_dir).sync(98)

        t = _make(
            ledger_dir=ledger_dir,
            state_file=tmp_path / "s.json",
            daily_limit=100,
        )
        # Seeded at init: combined baseline is the foreign 98.
        assert t.get_tokens_used_today() == 98

        # A small own contribution tips the COMBINED total over the cap.
        t.add_tokens(5)
        assert t.is_limit_reached() is True
        # Admission refreshes near the cap and denies.
        assert t.try_reserve(1) is None


class TestForcedAndDebouncedRefresh:
    def test_eager_refresh_near_limit(self, tmp_path: Path) -> None:
        ledger_dir = tmp_path / "ledger"
        t = _make(
            ledger_dir=ledger_dir,
            state_file=tmp_path / "s.json",
            daily_limit=100,
        )
        t.add_tokens(85)
        t.flush()  # own field = 85; cached combined = 85

        # Another process raises the combined total behind our back.
        SharedTokenLedger("autoexcerpter", ledger_dir=ledger_dir).sync(10)

        # Cached value is stale (still 85 -> remaining 15).
        assert t.get_tokens_remaining() == 15
        # Cached combined (85) exceeds 80% of 100 -> try_reserve refreshes first.
        t.try_reserve(1)
        assert t.get_tokens_remaining() == 5  # 100 - refreshed combined 95

    def test_below_80_picked_up_on_debounced_sync(self, tmp_path: Path) -> None:
        ledger_dir = tmp_path / "ledger"
        t = _make(
            ledger_dir=ledger_dir,
            state_file=tmp_path / "s.json",
            daily_limit=1000,
        )
        t.add_tokens(100)
        t.flush()  # own field 100; cached combined 100 (< 80% of 1000)

        SharedTokenLedger("chronotranscriber", ledger_dir=ledger_dir).sync(200)

        # Make the debounce interval look elapsed so the next add syncs.
        t._last_ledger_sync_monotonic = 0.0
        t.add_tokens(1)  # debounced sync fires inline, refreshing combined

        assert t.get_tokens_used_today() == 301  # 101 own + 200 foreign


class TestSeeding:
    def test_seeds_from_legacy_private_state(self, tmp_path: Path) -> None:
        ledger_dir = tmp_path / "ledger"
        state_file = tmp_path / "s.json"
        # Legacy private state for TODAY with prior usage.
        state_file.write_text(
            json.dumps({"date": _today(), "tokens_used": 500}),
            encoding="utf-8",
        )

        _make(ledger_dir=ledger_dir, state_file=state_file)

        ledger = SharedTokenLedger("chronominer", ledger_dir=ledger_dir)
        assert ledger.read_breakdown() == {"chronominer": 500}

    def test_repeated_init_does_not_double_seed(self, tmp_path: Path) -> None:
        ledger_dir = tmp_path / "ledger"
        state_file = tmp_path / "s.json"
        state_file.write_text(
            json.dumps({"date": _today(), "tokens_used": 500}),
            encoding="utf-8",
        )

        _make(ledger_dir=ledger_dir, state_file=state_file)
        _make(ledger_dir=ledger_dir, state_file=state_file)  # second process

        ledger = SharedTokenLedger("chronominer", ledger_dir=ledger_dir)
        assert ledger.read_breakdown() == {"chronominer": 500}


class _FakeLedger:
    """Ledger stand-in whose I/O can be toggled to degrade and recover."""

    def __init__(self) -> None:
        self.field = 0
        self.fail = True
        self.foreign = 0

    def seed(self, own: int) -> int | None:
        if self.fail:
            return None
        self.field = max(self.field, int(own))
        return self.field + self.foreign

    def sync(self, delta: int) -> int | None:
        if self.fail:
            return None
        self.field += max(0, int(delta))
        return self.field + self.foreign

    def read_breakdown(self) -> dict[str, int] | None:
        if self.fail:
            return None
        return {"chronominer": self.field}

    def read_combined(self) -> int | None:
        if self.fail:
            return None
        return self.field + self.foreign


class TestDegradedMode:
    def test_accumulates_while_degraded_then_lands_on_recovery(
        self, tmp_path: Path
    ) -> None:
        t = _make(
            ledger_dir=tmp_path / "ledger",
            state_file=tmp_path / "s.json",
            daily_limit=10_000_000,
        )
        # Swap in a degraded ledger and reset shared state to pre-seed.
        fake = _FakeLedger()
        with t._lock:
            t._ledger = fake
            t._seeded = False
            t._combined_total = 0
            t._unsynced_delta = 0
            t._ledger_degraded = False
            t._last_ledger_sync_monotonic = 0.0

        t.add_tokens(100)  # debounced sync fires; seed fails -> degraded
        assert t._ledger_degraded is True
        # Standalone fallback keeps the tracker fully functional.
        assert t.get_tokens_used_today() == 100
        assert t.is_limit_reached() is False

        t.add_tokens(50)  # delta keeps accumulating while degraded
        assert t.get_own_tokens_used_today() == 150

        # Ledger recovers; a forced sync must land the full accumulated amount.
        fake.fail = False
        t.sync_ledger_now()

        assert fake.field == 150  # accumulated 100 + 50 landed
        assert t._ledger_degraded is False
        assert t.get_tokens_used_today() == 150


class TestWaitLoopForcedSync:
    def test_foreign_reset_unblocks_within_a_poll(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        import modules.infra.token_tracker as tt

        ledger_dir = tmp_path / "ledger"
        # Another tool has exhausted the shared budget.
        SharedTokenLedger("chronotranscriber", ledger_dir=ledger_dir).sync(120)

        t = _make(
            ledger_dir=ledger_dir,
            state_file=tmp_path / "s.json",
            daily_limit=100,
        )
        tt._tracker_instance = t
        assert t.is_limit_reached() is True  # combined 120 > 100

        # Keep the configured limit fixed so only the ledger can unblock us.
        monkeypatch.setattr(tt, "_read_configured_daily_limit", lambda: None)

        calls = {"n": 0}

        async def mock_sleep(_seconds: float) -> None:
            calls["n"] += 1
            # Simulate the other tool resetting (its field drops to 0).
            _write_ledger(ledger_dir, {"chronominer": 0, "chronotranscriber": 0})

        with patch("asyncio.sleep", side_effect=mock_sleep):
            result = asyncio.run(check_and_wait_for_token_limit())

        assert result is True
        assert calls["n"] >= 1
        assert t.is_limit_reached() is False
