"""Token usage tracking with daily limits and timezone-aware reset.

This module provides a thread-safe token tracker that:
- Counts total tokens used from OpenAI API responses
- Enforces configurable daily token limits
- Automatically resets at midnight in the local timezone
- Persists state to disk to survive application restarts
- Thread-safe for concurrent API calls

Usage:
    from modules.infra.token_tracker import get_token_tracker

    tracker = get_token_tracker()

    # Check if we can proceed
    if tracker.can_use_tokens():
        # Make API call
        response = api_call()

        # Report usage
        tokens = response.get("usage", {}).get("total_tokens", 0)
        tracker.add_tokens(tokens)

    # Check if limit is reached
    if tracker.is_limit_reached():
        wait_time = tracker.get_seconds_until_reset()
        # Wait or defer processing
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import json
import os
import shutil
import threading
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from modules.infra.logger import setup_logger

if TYPE_CHECKING:
    from modules.infra.shared_ledger import SharedTokenLedger

logger = setup_logger(__name__)

# Default token tracker state-file name. The directory is resolved lazily at
# first use (see _default_token_tracker_file) rather than anchored to the cwd
# at import time.
_TOKEN_TRACKER_FILENAME = ".chronominer_token_state.json"

# User-level state directory (cross-process, CWD-independent). Overridable via
# paths_config general.state_dir. The former CWD-relative file is adopted once
# if the user-level file is absent (see _adopt_legacy_state).
_STATE_DIRNAME = ".chronominer"

# Minimum seconds between on-disk state writes. add_tokens() fires per API call
# (up to ~20/s under concurrency); without debouncing that rewrites the state
# file that often. In-memory counts stay exact; the debounced write plus an
# atexit flush bound disk churn while keeping cross-process enforcement
# best-effort.
_STATE_WRITE_DEBOUNCE_S = 1.0

# Minimum seconds between shared-ledger syncs on the debounced (add_tokens) path.
# Mirrors _STATE_WRITE_DEBOUNCE_S but is deliberately longer: a ledger sync takes
# an OS file lock, so it is heavier than a private-file write and must never run
# on the event-loop hot path. Forced syncs (near-limit admission, wait-loop
# polls, exit flush) bypass this interval.
_LEDGER_SYNC_DEBOUNCE_S = 2.0

# This tool's field name in the shared cross-tool ledger.
_LEDGER_TOOL_NAME = "chronominer"


def _resolve_state_dir() -> Path:
    """Resolve the state directory: config override, else ``~/.chronominer``."""
    try:
        from modules.config.loader import get_config_loader

        paths_config = get_config_loader().get_paths_config()
        override = (paths_config.get("general", {}) or {}).get("state_dir")
        if override:
            return Path(str(override)).expanduser()
    except Exception:
        # Config not available / not loaded yet: fall back to the user dir.
        pass
    return Path.home() / _STATE_DIRNAME


def _default_token_tracker_file() -> Path:
    """Resolve the default user-level state-file path."""
    return _resolve_state_dir() / _TOKEN_TRACKER_FILENAME


# Singleton instance
_tracker_instance: DailyTokenTracker | None = None
_tracker_lock = threading.Lock()


class DailyTokenTracker:
    """
    Thread-safe daily token usage tracker with persistent state.

    Tracks token usage across API calls and enforces daily limits with
    automatic reset at midnight in the local timezone.
    """

    def __init__(
        self,
        daily_limit: int,
        enabled: bool = True,
        state_file: Path | None = None,
        chunk_estimate_seed: int = 25_000,
        estimate_smoothing: float = 0.3,
        shared_enabled: bool = False,
        shared_ledger_dir: str | Path | None = None,
    ):
        """
        Initialize the token tracker.

        Args:
            daily_limit: Maximum tokens allowed per day.
            enabled: Whether token limiting is enabled.
            state_file: Path to persistent state file
                (default: .chronominer_token_state.json).
            chunk_estimate_seed: Cold-start estimate (in tokens) of how many
                tokens one chunk/page consumes, used by try_reserve() before
                any actual usage has been observed this run.
            estimate_smoothing: EWMA smoothing factor (0-1) applied to observed
                per-call token usage; higher reacts faster to recent calls.
            shared_enabled: Opt-in cross-tool combined budget. When True the
                daily limit is enforced against the COMBINED usage of every
                participating ChronoPipeline tool via a shared on-disk ledger,
                and that ledger replaces the private state file as persistence.
                When False (default) behaviour is bit-for-bit the private
                per-tool tracker with no ledger I/O whatsoever.
            shared_ledger_dir: Directory holding the shared ledger. Empty/None
                means the ledger default (``~/.chronopipeline``).
        """
        self.daily_limit = daily_limit
        self.enabled = enabled
        using_default = state_file is None
        self.state_file = state_file or _default_token_tracker_file()
        with contextlib.suppress(Exception):
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
        # Legacy CWD-file adoption applies only to the default user-level
        # location; an explicitly supplied state_file is honoured verbatim.
        if using_default:
            self._adopt_legacy_state()

        # Debounced-write bookkeeping.
        self._last_write_monotonic: float = 0.0
        self._pending_write: bool = False
        atexit.register(self._flush_on_exit)

        # Thread safety
        self._lock = threading.Lock()

        # Token tracking state
        self._current_date: str = ""  # Format: YYYY-MM-DD
        self._tokens_used_today: int = 0

        # Chunk-level reservation state (in-memory only; transient per run).
        # _tokens_reserved is headroom claimed by in-flight calls that have not
        # yet committed actual usage via add_tokens(); the admission check in
        # try_reserve() subtracts both committed and reserved tokens so that
        # concurrent workers cannot collectively overshoot the daily limit.
        self._tokens_reserved: int = 0
        self._seed: int = max(1, int(chunk_estimate_seed))
        self._alpha: float = min(1.0, max(0.0, float(estimate_smoothing)))
        self._ewma: float = float(self._seed)

        # Shared cross-tool ledger state (only touched when shared_enabled).
        # The ledger is constructed lazily on first use so a disabled tracker
        # performs zero ledger I/O. _unsynced_delta accumulates committed
        # tokens not yet pushed to the ledger; _combined_total caches the
        # last-known combined usage across all tools. Budget math while enabled
        # uses (_combined_total + _unsynced_delta) as the effective usage.
        self._shared_enabled: bool = bool(shared_enabled)
        self._shared_ledger_dir: str | Path | None = shared_ledger_dir or None
        self._ledger: SharedTokenLedger | None = None
        self._ledger_construct_failed: bool = False
        self._ledger_tool_name: str = _LEDGER_TOOL_NAME
        self._unsynced_delta: int = 0
        self._combined_total: int = 0
        self._seeded: bool = False
        self._ledger_degraded: bool = False
        self._ledger_sync_in_flight: bool = False
        self._last_ledger_sync_monotonic: float = 0.0

        # Load existing state from disk
        self._load_state()

        # Seed the shared ledger once at init so the combined baseline (this
        # tool's prior same-day usage plus any concurrent tools) is known
        # before the first admission check. Best-effort: a degraded ledger
        # simply leaves the tracker in standalone mode.
        if self._shared_enabled:
            with contextlib.suppress(Exception):
                self.sync_ledger_now()

        logger.info(
            f"Token tracker initialized: enabled={enabled}, "
            f"daily_limit={daily_limit:,}, "
            f"current_usage={self._tokens_used_today:,}, "
            f"shared_budget={self._shared_enabled}"
        )

    def flush(self) -> None:
        """Force-persist any pending debounced state write.

        When the shared budget is enabled this also forces a final ledger sync
        so the last accumulated delta lands before exit. If a background sync is
        in flight, wait briefly (bounded) for it to clear so the final push is
        not skipped by the in-flight guard.
        """
        if self._shared_enabled:
            for _ in range(50):
                with self._lock:
                    busy = self._ledger_sync_in_flight
                if not busy:
                    break
                time.sleep(0.02)
            with contextlib.suppress(Exception):
                self.sync_ledger_now()
        with self._lock:
            if self._pending_write:
                self._save_state()

    def set_daily_limit(self, new_limit: int) -> None:
        """Update the daily token limit at runtime.

        Used by the wait loop so a user editing ``concurrency_config.yaml``
        mid-wait (raising ``daily_token_limit.daily_tokens``) lifts the cap
        without a restart. A no-op when the value is unchanged.
        """
        new_limit = int(new_limit)
        with self._lock:
            if new_limit != self.daily_limit:
                logger.info(
                    "Daily token limit updated: %s -> %s",
                    f"{self.daily_limit:,}",
                    f"{new_limit:,}",
                )
                self.daily_limit = new_limit

    # ------------------------------------------------------------------
    # Shared cross-tool ledger integration
    # ------------------------------------------------------------------

    @staticmethod
    def _running_on_event_loop() -> bool:
        """True when called from a thread with a running asyncio event loop.

        Ledger I/O takes an OS file lock and must never block the event loop,
        so on-loop callers dispatch the sync to a background thread while
        off-loop callers (init, tests, atexit) run it inline for determinism.
        """
        try:
            asyncio.get_running_loop()
            return True
        except RuntimeError:
            return False

    def _get_or_create_ledger_locked(self) -> SharedTokenLedger | None:
        """Construct the shared ledger lazily. Must hold ``self._lock``.

        Construction touches no filesystem (the ledger defers all I/O to its
        locked merge), so this cannot fail for a bad directory; a genuinely
        invalid tool name is the only ValueError, and it is latched so we do
        not retry forever.
        """
        if self._ledger is None and not self._ledger_construct_failed:
            try:
                from modules.infra.shared_ledger import SharedTokenLedger

                self._ledger = SharedTokenLedger(
                    self._ledger_tool_name,
                    ledger_dir=self._shared_ledger_dir,
                )
            except Exception as exc:  # pragma: no cover - defensive
                self._ledger_construct_failed = True
                logger.warning("Could not construct shared token ledger: %s", exc)
        return self._ledger

    def _effective_used_locked(self) -> int:
        """Return the usage figure the budget is enforced against. Hold lock.

        Enabled and healthy: the last-known combined total across all tools
        plus this tool's not-yet-synced delta, so our own in-flight usage is
        never undercounted. Disabled or degraded: the private per-tool count,
        i.e. exactly today's standalone semantics.
        """
        if self._shared_enabled and not self._ledger_degraded:
            return self._combined_total + self._unsynced_delta
        return self._tokens_used_today

    def _due_for_ledger_sync_locked(self, force: bool) -> bool:
        """Whether a ledger sync should be dispatched now. Must hold the lock."""
        if not self._shared_enabled:
            return False
        if self._ledger_sync_in_flight:
            return False
        if force:
            return True
        elapsed = time.monotonic() - self._last_ledger_sync_monotonic
        return elapsed >= _LEDGER_SYNC_DEBOUNCE_S

    def _perform_ledger_sync(self) -> None:
        """Run a ledger sync off the event loop where one is running.

        On an event loop, dispatch to a daemon thread so the OS file lock never
        blocks the loop. Off-loop (init, tests, atexit), run inline so the
        refreshed combined total is visible to the immediately following check.
        """
        if self._running_on_event_loop():
            threading.Thread(target=self.sync_ledger_now, daemon=True).start()
        else:
            self.sync_ledger_now()

    def sync_ledger_now(self) -> None:
        """Seed-or-sync the shared ledger, writing back the combined total.

        Discipline: snapshot the delta under the tracker lock, call the ledger
        (seed or sync) with the lock RELEASED, then write the returned combined
        total back under the lock. The ledger has its own internal mutex; we
        never hold the tracker lock across a ledger call so the hot path cannot
        stall on ledger I/O. Degradation (ledger returns None) leaves the
        tracker in standalone mode with the unsynced delta preserved so a
        transient failure self-heals on a later sync.
        """
        if not self._shared_enabled:
            return

        with self._lock:
            if self._ledger_sync_in_flight:
                return
            self._ledger_sync_in_flight = True
            ledger = self._get_or_create_ledger_locked()
            need_seed = not self._seeded
            own_committed = self._tokens_used_today
            delta = self._unsynced_delta
            self._last_ledger_sync_monotonic = time.monotonic()

        try:
            if ledger is None:
                with self._lock:
                    self._ledger_degraded = True
                return

            own_field: int | None = None
            if need_seed:
                combined = ledger.seed(own_committed)
                if combined is not None:
                    breakdown = ledger.read_breakdown()
                    if breakdown is not None:
                        own_field = int(
                            breakdown.get(self._ledger_tool_name, own_committed)
                        )
            else:
                combined = ledger.sync(delta)

            with self._lock:
                if combined is None:
                    # Degraded: keep the unsynced delta so the full accumulated
                    # amount is pushed once the ledger recovers.
                    self._ledger_degraded = True
                else:
                    self._ledger_degraded = False
                    self._combined_total = combined
                    if need_seed:
                        self._seeded = True
                        baseline = own_field if own_field is not None else own_committed
                        if baseline > self._tokens_used_today:
                            self._tokens_used_today = baseline
                        # Any delta committed during the seed round is preserved
                        # for the next sync; the baseline is now in the ledger.
                        self._unsynced_delta = max(
                            0, self._tokens_used_today - baseline
                        )
                    else:
                        # Subtract only what we pushed; deltas that arrived
                        # mid-sync remain queued for the next push.
                        self._unsynced_delta = max(0, self._unsynced_delta - delta)
        finally:
            with self._lock:
                self._ledger_sync_in_flight = False

    def _maybe_forced_refresh_before_admit(self) -> None:
        """Force a ledger refresh before a reservation when it matters.

        Triggers a forced (debounce-bypassing) sync when the shared budget is
        not yet seeded, or when the cached combined total already exceeds 80%
        of the daily limit, so admission near the cap sees the freshest
        cross-tool usage. Off-loop this runs inline (fresh value visible to the
        caller); on-loop it dispatches and the value converges shortly after.
        """
        if not self._shared_enabled:
            return
        trigger = False
        with self._lock:
            near_limit = (
                self.daily_limit > 0 and self._combined_total > 0.8 * self.daily_limit
            )
            if (not self._seeded or near_limit) and self._due_for_ledger_sync_locked(
                force=True
            ):
                trigger = True
        if trigger:
            self._perform_ledger_sync()

    def _adopt_legacy_state(self) -> None:
        """One-time adoption of the legacy CWD state file.

        If the user-level state file does not yet exist but a legacy
        ``.chronominer_token_state.json`` sits in the current directory, copy it
        across so a prior day's usage is not silently reset on upgrade.
        """
        try:
            if self.state_file.exists():
                return
            legacy = Path.cwd() / _TOKEN_TRACKER_FILENAME
            if legacy.exists() and legacy.resolve() != self.state_file.resolve():
                shutil.copy2(legacy, self.state_file)
                logger.info(
                    "Adopted legacy token state %s -> %s", legacy, self.state_file
                )
        except Exception as exc:
            logger.debug("Legacy token-state adoption skipped: %s", exc)

    def _flush_on_exit(self) -> None:
        """atexit hook: persist a pending debounced write."""
        try:
            with self._lock:
                if self._pending_write:
                    self._save_state()
        except Exception:
            pass

    def _get_current_date_str(self) -> str:
        """Get current date as string in YYYY-MM-DD format."""
        return datetime.now().strftime("%Y-%m-%d")

    def _load_state(self) -> None:
        """Load token usage state from disk."""
        if not self.state_file.exists():
            # No existing state, initialize fresh
            self._current_date = self._get_current_date_str()
            self._tokens_used_today = 0
            logger.debug("No existing token state file found, starting fresh")
            return

        try:
            with open(self.state_file, encoding="utf-8") as f:
                state = json.load(f)

            saved_date = state.get("date", "")
            saved_tokens = state.get("tokens_used", 0)

            current_date = self._get_current_date_str()

            if saved_date == current_date:
                # Same day, restore token count
                self._current_date = saved_date
                self._tokens_used_today = saved_tokens
                logger.info(
                    f"Loaded token state for {current_date}: "
                    f"{self._tokens_used_today:,} tokens used"
                )
            else:
                # Different day, reset counter
                self._current_date = current_date
                self._tokens_used_today = 0
                logger.info(
                    f"New day detected (was {saved_date}, now {current_date}). "
                    "Token counter reset to 0."
                )
                # Save the reset state
                self._save_state()

        except Exception as e:
            logger.warning(f"Error loading token state from {self.state_file}: {e}")
            # Initialize fresh on error
            self._current_date = self._get_current_date_str()
            self._tokens_used_today = 0

    def _save_state(self) -> None:
        """Save current token usage state to disk (immediate, atomic).

        Writes to a per-process-unique temp file in the same directory before
        an atomic ``replace()``, so concurrent processes can never collide on
        the temp path. A fixed ``.tmp`` name previously let one process's write
        clobber another's mid-flight. The temp file is always removed in the
        finally block, even when the write or replace failed.
        """
        temp_file = self.state_file.with_name(
            f"{self.state_file.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
        )
        try:
            state = {
                "date": self._current_date,
                "tokens_used": self._tokens_used_today,
                "last_updated": datetime.now().isoformat(),
            }

            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)

            # Atomic replace of the destination.
            temp_file.replace(self.state_file)
            self._last_write_monotonic = time.monotonic()
            self._pending_write = False

        except Exception as e:
            logger.error(f"Error saving token state to {self.state_file}: {e}")
        finally:
            # Remove the temp file if it survived (write/replace failure). A
            # successful replace() already moved it, so exists() is False.
            with contextlib.suppress(Exception):
                if temp_file.exists():
                    temp_file.unlink()

    def _debounced_save(self) -> None:
        """Persist state only if at least ``_STATE_WRITE_DEBOUNCE_S`` elapsed
        since the last write; otherwise mark a pending write flushed at exit or
        on the next eligible call. Must be called under ``self._lock``."""
        now = time.monotonic()
        if now - self._last_write_monotonic >= _STATE_WRITE_DEBOUNCE_S:
            self._save_state()
        else:
            self._pending_write = True

    def _check_and_reset_if_new_day(self) -> None:
        """Check if it's a new day and reset counter if needed.

        Must be called under ``self._lock``.
        """
        current_date = self._get_current_date_str()

        if current_date != self._current_date:
            logger.info(
                f"New day detected: {current_date} (was {self._current_date}). "
                f"Resetting token counter from {self._tokens_used_today:,} to 0."
            )
            self._current_date = current_date
            self._tokens_used_today = 0
            if self._shared_enabled:
                # The ledger rolls over internally; reset the local mirror and
                # force a re-seed on the next sync. The private file is left
                # untouched while the shared budget is the active persistence.
                self._unsynced_delta = 0
                self._combined_total = 0
                self._seeded = False
            else:
                self._save_state()

    def add_tokens(self, tokens: int) -> None:
        """
        Add tokens to the daily count.

        Args:
            tokens: Number of tokens to add.
        """
        if not self.enabled or tokens <= 0:
            return

        do_ledger_sync = False
        with self._lock:
            self._check_and_reset_if_new_day()
            self._tokens_used_today += tokens
            # Update the rolling per-call estimate used by try_reserve().
            self._ewma = self._alpha * tokens + (1.0 - self._alpha) * self._ewma

            if self._shared_enabled:
                # Ledger is the active persistence: accumulate the delta and
                # decide (under the lock) whether a debounced sync is due. The
                # actual ledger I/O runs outside the lock via _perform below.
                self._unsynced_delta += tokens
                do_ledger_sync = self._due_for_ledger_sync_locked(force=False)
            else:
                # Debounced: in-memory count is exact; disk write is throttled.
                self._debounced_save()

            logger.debug(
                f"Added {tokens:,} tokens. "
                f"Daily total: {self._tokens_used_today:,}/{self.daily_limit:,}"
            )

        if do_ledger_sync:
            self._perform_ledger_sync()

    def try_reserve(self, estimate: int | None = None) -> int | None:
        """Reserve estimated tokens for one chunk/page before launching it.

        The estimate is the larger of the caller-supplied hint (e.g. a tiktoken
        input count for text) and the rolling EWMA of observed per-call usage,
        so the reservation tracks reality and never drops below the average.

        Returns the reserved amount, ``0`` when limiting is disabled (admit
        freely, nothing to release), or ``None`` when the remaining budget
        cannot cover the estimate (caller should stop admitting new work). A
        non-zero reservation must be matched by a later :meth:`release` of the
        same amount once the call completes.
        """
        if not self.enabled:
            return 0

        # Forced pre-admission refresh when near the combined cap (or not yet
        # seeded). Runs outside the tracker lock; inline off-loop so the fresh
        # combined total is visible to the admission check below.
        self._maybe_forced_refresh_before_admit()

        with self._lock:
            self._check_and_reset_if_new_day()
            est = max(int(estimate or 0), max(1, round(self._ewma)))
            available = (
                self.daily_limit - self._effective_used_locked() - self._tokens_reserved
            )
            if est > available:
                return None
            self._tokens_reserved += est
            return est

    def release(self, amount: int) -> None:
        """Release a reservation made by :meth:`try_reserve` after the call.

        Actual usage is committed separately via :meth:`add_tokens`; releasing
        only frees the transient headroom the reservation was holding.
        """
        if not self.enabled or amount <= 0:
            return

        with self._lock:
            self._tokens_reserved = max(0, self._tokens_reserved - amount)

    def get_tokens_used_today(self) -> int:
        """
        Get the number of tokens used today.

        With the shared budget enabled this is the COMBINED usage across all
        participating tools (the figure the daily limit is enforced against);
        otherwise it is this tool's private count. See
        :meth:`get_own_tokens_used_today` for the per-tool figure.

        Returns:
            Token count for current day.
        """
        with self._lock:
            self._check_and_reset_if_new_day()
            return self._effective_used_locked()

    def get_own_tokens_used_today(self) -> int:
        """Return this tool's private token count for today (never combined)."""
        with self._lock:
            self._check_and_reset_if_new_day()
            return self._tokens_used_today

    def get_tokens_remaining(self) -> int:
        """
        Get the number of tokens remaining for today.

        Returns:
            Remaining token count (0 if limit exceeded).
        """
        if not self.enabled:
            return self.daily_limit  # Unlimited

        with self._lock:
            self._check_and_reset_if_new_day()
            remaining = self.daily_limit - self._effective_used_locked()
            return max(0, remaining)

    def is_limit_reached(self) -> bool:
        """
        Check if the daily token limit has been reached.

        Returns:
            True if limit is reached or exceeded, False otherwise.
        """
        if not self.enabled:
            return False

        return self.get_tokens_remaining() == 0

    def can_use_tokens(self, estimated_tokens: int = 0) -> bool:
        """
        Check if we can use a certain number of tokens.

        Args:
            estimated_tokens: Estimated tokens needed (default: 0 for any usage).

        Returns:
            True if we can proceed, False if limit would be exceeded.
        """
        if not self.enabled:
            return True

        remaining = self.get_tokens_remaining()

        if estimated_tokens > 0:
            return remaining >= estimated_tokens
        else:
            # Just check if any tokens remain
            return remaining > 0

    def get_seconds_until_reset(self) -> int:
        """
        Get the number of seconds until the counter resets (midnight).

        Returns:
            Seconds until midnight (00:00:00 local time).
        """
        now = datetime.now()
        # Calculate next midnight
        tomorrow = now.date() + timedelta(days=1)
        midnight = datetime.combine(tomorrow, datetime.min.time())

        delta = midnight - now
        return int(delta.total_seconds())

    def get_reset_time(self) -> datetime:
        """
        Get the datetime when the counter will reset.

        Returns:
            Datetime of next midnight.
        """
        now = datetime.now()
        tomorrow = now.date() + timedelta(days=1)
        return datetime.combine(tomorrow, datetime.min.time())

    def get_usage_percentage(self) -> float:
        """
        Get current usage as percentage of daily limit.

        Returns:
            Percentage (0-100+) of daily limit used.
        """
        if not self.enabled or self.daily_limit == 0:
            return 0.0

        used = self.get_tokens_used_today()
        return (used / self.daily_limit) * 100.0

    def get_stats(self) -> dict:
        """
        Get comprehensive token usage statistics.

        Returns:
            Dictionary with usage stats.
        """
        used = self.get_tokens_used_today()
        remaining = self.get_tokens_remaining()
        percentage = self.get_usage_percentage()
        seconds_until_reset = self.get_seconds_until_reset()
        reset_time = self.get_reset_time()

        stats = {
            "enabled": self.enabled,
            "daily_limit": self.daily_limit,
            "tokens_used_today": used,
            "tokens_remaining": remaining,
            "usage_percentage": round(percentage, 2),
            "limit_reached": self.is_limit_reached(),
            "seconds_until_reset": seconds_until_reset,
            "reset_time": reset_time.isoformat(),
            "current_date": self._current_date,
        }
        stats.update(self._shared_stats())
        return stats

    def _shared_stats(self) -> dict[str, Any]:
        """Shared-budget stats: combined total, own count, and per-tool split.

        Returns an empty dict when the shared budget is disabled so callers see
        no change. ``read_breakdown`` is a lock-free ledger read run outside the
        tracker lock; ``tokens_used_today`` above already reflects the combined
        figure when enabled.
        """
        if not self._shared_enabled:
            return {}
        with self._lock:
            ledger = self._ledger
            own = self._tokens_used_today
            combined = self._effective_used_locked()
            degraded = self._ledger_degraded
        breakdown: dict[str, int] | None = None
        if ledger is not None:
            breakdown = ledger.read_breakdown()
        return {
            "shared_budget_enabled": True,
            "shared_budget_degraded": degraded,
            "own_tokens_used_today": own,
            "combined_tokens_used_today": combined,
            "shared_breakdown": breakdown or {},
        }


def check_token_limit_enabled() -> bool:
    """
    Check if daily token limit is enabled in configuration.

    This is the canonical implementation - do not duplicate in main scripts.

    Returns:
        True if token limiting is enabled, False otherwise.
    """
    tracker = get_token_tracker()
    return tracker.enabled


def _read_configured_daily_limit() -> int | None:
    """Read the currently-configured daily token limit fresh from disk.

    Bypasses the config cache (``force_reload=True``) so an edit to
    ``concurrency_config.yaml`` made while the wait loop is polling is
    observed. Returns ``None`` when the value is absent or the config cannot
    be read, so callers keep the old limit on failure.
    """
    from modules.config.loader import get_config_loader

    concurrency_config = get_config_loader(force_reload=True).get_concurrency_config()
    token_limit_config = (concurrency_config or {}).get("daily_token_limit", {}) or {}
    raw = token_limit_config.get("daily_tokens")
    if raw is None:
        return None
    return int(str(raw).replace("_", ""))


async def check_and_wait_for_token_limit(ui: Any = None, logger: Any = None) -> bool:
    """
    Check if daily token limit is reached and wait until next day if needed.

    This is the canonical implementation - do not duplicate in main scripts.

    Args:
        ui: Optional UserInterface instance for user feedback.
        logger: Optional logger instance. If None, uses module logger.

    Returns:
        True if processing can continue, False if user cancelled wait.
    """

    _logger = globals().get("logger") if logger is None else logger

    token_tracker = get_token_tracker()

    if not token_tracker.enabled or not token_tracker.is_limit_reached():
        return True

    # Token limit reached - need to wait until next day
    stats = token_tracker.get_stats()
    reset_time = token_tracker.get_reset_time()
    seconds_until_reset = token_tracker.get_seconds_until_reset()

    if _logger:
        _logger.warning(
            f"Daily token limit reached: "
            f"{stats['tokens_used_today']:,}/{stats['daily_limit']:,} tokens used"
        )
        _logger.info(
            f"Waiting until {reset_time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"({seconds_until_reset // 3600}h "
            f"{(seconds_until_reset % 3600) // 60}m) for token limit reset..."
        )

    if ui:
        ui.print_warning(
            f"\n⚠ Daily token limit reached: "
            f"{stats['tokens_used_today']:,}/{stats['daily_limit']:,} tokens used"
        )
        ui.print_info(
            f"Waiting until {reset_time.strftime('%Y-%m-%d %H:%M:%S')} for daily reset "
            f"({seconds_until_reset // 3600}h "
            f"{(seconds_until_reset % 3600) // 60}m remaining)"
        )
        ui.print_info("Press Ctrl+C to cancel and exit.")
    elif _logger:
        _logger.info("Press Ctrl+C to cancel and exit.")

    try:
        sleep_interval = 60
        elapsed = 0

        while elapsed < seconds_until_reset:
            interval = min(sleep_interval, max(0, seconds_until_reset - elapsed))
            await asyncio.sleep(interval)
            elapsed += interval

            # Forced ledger refresh each poll so another tool's usage or its
            # midnight reset is observed while we wait. Runs off the event loop
            # via to_thread; a no-op (and skipped) when the shared budget is
            # disabled, so single-tool waits are unchanged.
            if getattr(token_tracker, "_shared_enabled", False):
                try:
                    await asyncio.to_thread(token_tracker.sync_ledger_now)
                except Exception as exc:
                    if _logger:
                        _logger.debug(
                            "Shared ledger refresh during wait failed: %s", exc
                        )

            # Live re-read of the configured daily limit: a user raising
            # daily_token_limit.daily_tokens mid-wait lifts the cap without a
            # restart. A read failure keeps the current limit (debug-logged).
            try:
                new_limit = _read_configured_daily_limit()
                if new_limit is not None:
                    token_tracker.set_daily_limit(new_limit)
            except Exception as exc:
                if _logger:
                    _logger.debug(
                        "Could not refresh daily token limit during wait: %s", exc
                    )

            if not token_tracker.is_limit_reached():
                if _logger:
                    _logger.info("Token limit has been reset. Resuming processing.")
                if ui:
                    ui.print_success("Token limit has been reset. Resuming processing.")
                return True

        if _logger:
            _logger.info("Token limit has been reset. Resuming processing.")
        if ui:
            ui.print_success("\nToken limit has been reset. Resuming processing.")
        return True

    except (KeyboardInterrupt, asyncio.CancelledError):
        if _logger:
            _logger.info("Wait cancelled by user.")
        if ui:
            ui.print_warning("\nWait cancelled by user.")
        return False


def get_token_tracker(
    daily_limit: int | None = None, enabled: bool | None = None
) -> DailyTokenTracker:
    """
    Get the singleton token tracker instance.

    Args:
        daily_limit: Override daily limit (for initialization only).
        enabled: Override enabled flag (for initialization only).

    Returns:
        Global DailyTokenTracker instance.
    """
    global _tracker_instance

    if _tracker_instance is None:
        with _tracker_lock:
            # Double-check locking
            if _tracker_instance is None:
                # Load configuration for any values not supplied by the caller.
                from modules.config.loader import get_config_loader

                config_loader = get_config_loader()
                concurrency_config = config_loader.get_concurrency_config()
                token_limit_config = concurrency_config.get("daily_token_limit", {})

                if daily_limit is None:
                    daily_limit = int(
                        str(token_limit_config.get("daily_tokens", 10000000)).replace(
                            "_", ""
                        )
                    )
                if enabled is None:
                    enabled = token_limit_config.get("enabled", False)

                seed = int(
                    str(token_limit_config.get("chunk_estimate_seed", 25000)).replace(
                        "_", ""
                    )
                )
                smoothing = float(token_limit_config.get("estimate_smoothing", 0.3))

                # Optional opt-in cross-tool combined budget (default off).
                shared_config = concurrency_config.get("shared_token_budget", {}) or {}
                shared_enabled = bool(shared_config.get("enabled", False))
                shared_ledger_dir = shared_config.get("ledger_dir", "") or None

                _tracker_instance = DailyTokenTracker(
                    daily_limit=daily_limit,
                    enabled=enabled,
                    chunk_estimate_seed=seed,
                    estimate_smoothing=smoothing,
                    shared_enabled=shared_enabled,
                    shared_ledger_dir=shared_ledger_dir,
                )

    return _tracker_instance
