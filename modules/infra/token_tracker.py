"""Token usage tracking with daily limits and timezone-aware reset.

This module provides a thread-safe token tracker that:
- Counts total tokens used from OpenAI API responses
- Enforces configurable daily token limits, including per-(provider key,
  pool) daily caps for OpenAI free-tier models
- Automatically resets at 00:01 UTC, one minute after OpenAI's 00:00 UTC
  free-tier reset
- Persists state to disk to survive application restarts
- Thread-safe for concurrent API calls

Usage:
    from modules.infra.token_tracker import get_token_tracker

    tracker = get_token_tracker()

    # Check if we can proceed
    if tracker.can_use_tokens():
        # Make API call
        response = api_call()

        # Report usage (stamped with the key pool that served the call)
        tokens = response.get("usage", {}).get("total_tokens", 0)
        tracker.add_tokens(tokens, provider="openai",
                           key_env="OPENAI_API_KEY", model="gpt-4o")

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
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from modules.infra.logger import setup_logger
from modules.infra.shared_ledger import (
    DEFAULT_POOL_CAPS,
    UNATTRIBUTED_BUCKET,
    BucketKey,
    compile_pools,
    derive_pool,
)

if TYPE_CHECKING:
    from modules.infra.shared_ledger import (
        CompiledPools,
        SharedTokenLedger,
        UsageSnapshot,
    )

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

# One-minute safety buffer past OpenAI's 00:00 UTC free-tier reset, so the
# tracker never frees its budget before the upstream quota has actually reset.
# Mirrors modules.infra.shared_ledger._RESET_BUFFER (kept as a separate
# constant since this module is not vendored and may import the ledger only
# under TYPE_CHECKING).
_RESET_BUFFER = timedelta(minutes=1)


def _bucket_to_str(bucket: BucketKey) -> str:
    """Serialize a bucket to the private-state key ``provider|key_env|pool``.

    ``pool`` is the empty string for ``None`` so the round-trip is lossless.
    """
    return f"{bucket.provider}|{bucket.key_env}|{bucket.pool or ''}"


def _bucket_from_str(text: str) -> BucketKey | None:
    """Parse a ``provider|key_env|pool`` private-state key back to a bucket.

    Returns ``None`` for a malformed key (never crashes the load path).
    """
    parts = str(text).split("|")
    if len(parts) != 3:
        return None
    provider, key_env, pool = parts
    return BucketKey(provider, key_env, pool or None)


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
    automatic reset at 00:01 UTC (one minute after OpenAI's 00:00 UTC
    free-tier reset).

    Two enforcement gates run when a call is stamped with its serving key
    pool (provider + key env-var name + derived pool):

    - Per-key-pool gate (primary): each OpenAI free-tier key has its own
      daily cap per pool ("large" | "small"); a call is blocked when its
      bucket's usage would exceed that cap. Non-OpenAI providers and local
      endpoints (pool ``None``) are exempt.
    - Combined gate (secondary): the classic ``daily_tokens`` cap. Under
      the default ``scope="pooled"`` it applies only to pooled (OpenAI
      free-tier) calls, so a free local endpoint is never blocked by OpenAI
      usage; ``scope="all"`` restores the legacy block-everything behaviour.
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
        scope: str = "pooled",
        per_key_pool_caps_enabled: bool = True,
        pool_caps: dict[tuple[str, str], int] | None = None,
        provider_pools: dict[str, dict[str, tuple[str, ...]]] | None = None,
    ):
        """
        Initialize the token tracker.

        Args:
            daily_limit: Maximum tokens allowed per day (combined gate).
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
            scope: ``"pooled"`` (default) applies the combined gate only to
                calls whose model belongs to a defined pool; ``"all"`` applies
                it to every call (legacy block-everything behaviour).
            per_key_pool_caps_enabled: When True (default), enforce per-(key,
                pool) daily caps for models in a defined pool.
            pool_caps: Per-(provider, pool label) daily-cap overrides. A pool
                without an entry falls back to ``DEFAULT_POOL_CAPS`` for its
                label; a label unknown there is tracked but uncapped.
            provider_pools: Custom pool definitions (provider -> pool label ->
                model-name prefixes). When given for a provider they REPLACE
                the built-in pools for that provider; other providers keep the
                built-ins.
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

        # Per-key-pool enforcement configuration. _pool_caps is keyed by
        # (provider, pool label); resolution falls back to DEFAULT_POOL_CAPS
        # by label, then to "uncapped". _compiled_pools carries any configured
        # custom pool definitions (None = built-ins only).
        self._scope: str = "all" if str(scope).strip().lower() == "all" else "pooled"
        self._per_key_pool_caps_enabled: bool = bool(per_key_pool_caps_enabled)
        self._pool_caps: dict[tuple[str, str], int] = dict(pool_caps or {})
        self._compiled_pools: CompiledPools | None = (
            compile_pools(provider_pools) if provider_pools else None
        )

        # Debounced-write bookkeeping.
        self._last_write_monotonic: float = 0.0
        self._pending_write: bool = False
        atexit.register(self._flush_on_exit)

        # Thread safety
        self._lock = threading.Lock()

        # Token tracking state
        self._current_date: str = ""  # Format: YYYY-MM-DD
        self._tokens_used_today: int = 0
        # This tool's OWN per-bucket usage today (the private per-key counts).
        # Sum equals _tokens_used_today (legacy adoption puts the whole legacy
        # count in the unattributed bucket). Used for standalone/degraded
        # per-key enforcement and persisted in the state file.
        self._bucket_used_today: dict[BucketKey, int] = {}

        # Chunk-level reservation state (in-memory only; transient per run).
        # _tokens_reserved is headroom claimed per bucket by in-flight calls
        # that have not yet committed actual usage via add_tokens(); the
        # admission check in try_reserve() subtracts committed and reserved
        # tokens so concurrent workers cannot collectively overshoot a cap.
        self._tokens_reserved: dict[BucketKey, int] = {}
        self._seed: int = max(1, int(chunk_estimate_seed))
        self._alpha: float = min(1.0, max(0.0, float(estimate_smoothing)))
        self._ewma: float = float(self._seed)

        # The last bucket a reservation was denied on (and whether it was
        # stamped). The wait loop, which has no per-call stamp, consults this
        # so a per-key pool block is treated as limit-reached and the wait
        # actually waits (rather than spinning on the combined gate alone).
        self._last_blocked_bucket: BucketKey | None = None
        self._last_blocked_stamped: bool = False

        # Shared cross-tool ledger state (only touched when shared_enabled).
        # The ledger is constructed lazily on first use so a disabled tracker
        # performs zero ledger I/O. _unsynced_deltas accumulates committed
        # per-bucket tokens not yet pushed to the ledger; _combined_total caches
        # the last-known combined usage across all tools; _bucket_totals caches
        # the cross-tool aggregate per bucket (what per-key caps enforce).
        self._shared_enabled: bool = bool(shared_enabled)
        self._shared_ledger_dir: str | Path | None = shared_ledger_dir or None
        self._ledger: SharedTokenLedger | None = None
        self._ledger_construct_failed: bool = False
        self._ledger_tool_name: str = _LEDGER_TOOL_NAME
        self._unsynced_deltas: dict[BucketKey, int] = {}
        self._combined_total: int = 0
        self._bucket_totals: dict[BucketKey, int] = {}
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
            f"shared_budget={self._shared_enabled}, scope={self._scope}, "
            f"per_key_pool_caps={self._per_key_pool_caps_enabled}"
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
    # Bucket derivation and enforcement helpers
    # ------------------------------------------------------------------

    def _derive_bucket(
        self, provider: str | None, key_env: str | None, model: str | None
    ) -> tuple[BucketKey, bool]:
        """Derive the accounting bucket for a call and whether it was stamped.

        A call is "stamped" only when BOTH provider and key_env are known;
        otherwise it lands in the sentinel unattributed bucket (and keeps the
        legacy combined-cap-only semantics). Configured pool definitions
        (``_compiled_pools``) replace the built-ins for the providers they
        cover; other providers keep the built-in pools.
        """
        if provider and key_env:
            prov = str(provider).strip().lower()
            pool = derive_pool(prov, model, pools=self._compiled_pools)
            return BucketKey(prov, str(key_env), pool), True
        return UNATTRIBUTED_BUCKET, False

    def _effective_stamp_locked(
        self, provider: str | None, key_env: str | None, model: str | None
    ) -> tuple[BucketKey, bool]:
        """Resolve the bucket for a query. Must hold ``self._lock``.

        An explicit stamp wins; otherwise the last denied reservation's bucket
        is used (so the stampless wait loop reflects a per-key pool block);
        failing that, the unattributed bucket (legacy combined-only).
        """
        if provider and key_env:
            return self._derive_bucket(provider, key_env, model)
        if self._last_blocked_bucket is not None:
            return self._last_blocked_bucket, self._last_blocked_stamped
        return UNATTRIBUTED_BUCKET, False

    def _effective_used_locked(self) -> int:
        """Return the usage figure the combined gate is enforced against.

        Enabled and healthy shared budget: the last-known combined total across
        all tools plus this tool's not-yet-synced delta, so our own in-flight
        usage is never undercounted. Disabled or degraded: the private per-tool
        count, i.e. exactly today's standalone semantics.
        """
        if self._shared_enabled and not self._ledger_degraded:
            return self._combined_total + sum(self._unsynced_deltas.values())
        return self._tokens_used_today

    def _total_reserved_locked(self) -> int:
        """Sum of all outstanding reservations across buckets. Hold the lock."""
        return sum(self._tokens_reserved.values())

    def _combined_available_locked(self) -> int:
        """Remaining headroom under the combined gate (may be negative)."""
        return (
            self.daily_limit
            - self._effective_used_locked()
            - self._total_reserved_locked()
        )

    def _bucket_committed_locked(self, bucket: BucketKey) -> int:
        """Committed (non-reserved) usage for a bucket. Hold the lock.

        Uses the cross-tool ledger aggregate plus this tool's un-synced delta
        while the shared budget is healthy; the private per-bucket count in
        standalone or degraded mode.
        """
        if self._shared_enabled and not self._ledger_degraded:
            return self._bucket_totals.get(bucket, 0) + self._unsynced_deltas.get(
                bucket, 0
            )
        return self._bucket_used_today.get(bucket, 0)

    def _bucket_usage_locked(self, bucket: BucketKey) -> int:
        """Committed + reserved usage for a bucket (the pool-gate figure)."""
        return self._bucket_committed_locked(bucket) + self._tokens_reserved.get(
            bucket, 0
        )

    def _pool_cap_for(self, bucket: BucketKey) -> int | None:
        """The per-key daily cap for a bucket, or ``None`` when not gated.

        Resolution: configured cap for (provider, pool label), else the
        built-in ``DEFAULT_POOL_CAPS`` for the label, else ``None`` -- the
        pool is tracked in the ledger but uncapped.
        """
        if bucket.pool is None or not self._per_key_pool_caps_enabled:
            return None
        cap = self._pool_caps.get((bucket.provider, bucket.pool))
        if cap is None:
            cap = DEFAULT_POOL_CAPS.get(bucket.pool)
        return cap

    def set_pool_config(
        self,
        scope: str,
        per_key_pool_caps_enabled: bool,
        pool_caps: dict[tuple[str, str], int],
        provider_pools: dict[str, dict[str, tuple[str, ...]]],
    ) -> None:
        """Replace the pool enforcement configuration at runtime.

        Used by the wait loop so a user editing ``concurrency_config.yaml``
        mid-wait (raising a pool cap, remapping pools, or disabling per-key
        caps) takes effect without a restart.
        """
        compiled = compile_pools(provider_pools) if provider_pools else None
        with self._lock:
            self._scope = "all" if str(scope).strip().lower() == "all" else "pooled"
            self._per_key_pool_caps_enabled = bool(per_key_pool_caps_enabled)
            self._pool_caps = dict(pool_caps)
            self._compiled_pools = compiled

    def _blocked_for_locked(self, bucket: BucketKey, stamped: bool, est: int) -> bool:
        """Whether admitting ``est`` tokens for ``bucket`` is blocked.

        Unstamped calls keep the legacy combined-only gate. Stamped pool-None
        calls are never blocked under ``scope="pooled"`` (combined gate only
        under ``scope="all"``). Stamped pooled calls face the per-key pool gate
        (primary) and the combined gate (secondary).
        """
        if not self.enabled:
            return False
        if not stamped:
            return est > self._combined_available_locked()
        if bucket.pool is None:
            if self._scope == "all":
                return est > self._combined_available_locked()
            return False
        cap = self._pool_cap_for(bucket)
        if cap is not None and est > cap - self._bucket_usage_locked(bucket):
            return True
        return est > self._combined_available_locked()

    def _effective_limit_for_locked(
        self, bucket: BucketKey, stamped: bool
    ) -> int | None:
        """The effective per-call limit for a bucket; ``None`` = never blocks.

        Used by :meth:`estimate_exceeds_daily_limit`: a pooled bucket's limit is
        min(daily cap, per-key pool cap).
        """
        if not stamped:
            return self.daily_limit
        if bucket.pool is None:
            return self.daily_limit if self._scope == "all" else None
        limit = self.daily_limit
        cap = self._pool_cap_for(bucket)
        if cap is not None:
            limit = min(limit, cap)
        return limit

    def _remaining_locked(self, bucket: BucketKey, stamped: bool) -> int:
        """Remaining budget for a bucket. Hold the lock.

        Combined remaining for unstamped/all-scope calls; for a pooled bucket,
        the tighter of combined and per-key remaining; pool-None pooled-scope
        calls report the full limit (they are never blocked).
        """
        combined_remaining = max(0, self.daily_limit - self._effective_used_locked())
        if not stamped:
            return combined_remaining
        if bucket.pool is None:
            return combined_remaining if self._scope == "all" else self.daily_limit
        cap = self._pool_cap_for(bucket)
        if cap is not None:
            pool_remaining = max(0, cap - self._bucket_committed_locked(bucket))
            return min(combined_remaining, pool_remaining)
        return combined_remaining

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
        """Seed-or-sync the shared ledger, caching the returned snapshot.

        Discipline: snapshot the per-bucket deltas under the tracker lock, call
        the ledger (seed or sync) with the lock RELEASED, then write the
        returned totals back under the lock. We never hold the tracker lock
        across a ledger call so the hot path cannot stall on ledger I/O.
        Degradation (ledger returns None) leaves the tracker in standalone mode
        with the unsynced deltas preserved so a transient failure self-heals on
        a later sync.
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
            own_buckets_snapshot = dict(self._bucket_used_today)
            deltas_snapshot = dict(self._unsynced_deltas)
            self._last_ledger_sync_monotonic = time.monotonic()

        try:
            if ledger is None:
                with self._lock:
                    self._ledger_degraded = True
                return

            snapshot: UsageSnapshot | None
            if need_seed:
                snapshot = ledger.seed_usage(own_committed, own_buckets_snapshot)
            else:
                snapshot = ledger.sync_usage(deltas_snapshot)

            with self._lock:
                if snapshot is None:
                    # Degraded: keep the unsynced deltas so the full accumulated
                    # amount is pushed once the ledger recovers.
                    self._ledger_degraded = True
                else:
                    self._ledger_degraded = False
                    self._combined_total = snapshot.combined
                    self._bucket_totals = dict(snapshot.buckets)
                    if need_seed:
                        self._seeded = True
                        baseline = snapshot.own_total
                        if baseline > self._tokens_used_today:
                            self._tokens_used_today = baseline
                        # The seed adopted own_buckets_snapshot into the ledger
                        # (max semantics); anything added since remains queued.
                        new_deltas: dict[BucketKey, int] = {}
                        for b in set(self._bucket_used_today) | set(
                            own_buckets_snapshot
                        ):
                            rem = self._bucket_used_today.get(
                                b, 0
                            ) - own_buckets_snapshot.get(b, 0)
                            if rem > 0:
                                new_deltas[b] = rem
                        self._unsynced_deltas = new_deltas
                    else:
                        # Subtract only what we pushed, per bucket; deltas that
                        # arrived mid-sync remain queued for the next push.
                        for b, pushed in deltas_snapshot.items():
                            rem = self._unsynced_deltas.get(b, 0) - pushed
                            if rem > 0:
                                self._unsynced_deltas[b] = rem
                            else:
                                self._unsynced_deltas.pop(b, None)
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

    def read_ledger_usage(self) -> UsageSnapshot | None:
        """Lock-free cross-tool usage snapshot, or ``None`` when unavailable.

        Used by the wait loop to show per-key numbers across tools. Returns
        ``None`` in standalone/degraded mode (no cross-tool view).
        """
        if not self._shared_enabled:
            return None
        with self._lock:
            ledger = self._ledger
            degraded = self._ledger_degraded
        if ledger is None or degraded:
            return None
        return ledger.read_usage()

    def rebind_active_key_env(self, provider: str, new_key_env: str) -> bool:
        """Re-point the last blocked bucket at a freshly resolved key env var.

        Called by the wait loop each poll so a user remapping the provider's
        key (e.g. ``openai: OPENAI_API_KEY_2``) mid-wait switches the active
        bucket, letting the fresh key's headroom unblock the wait without a
        restart. Returns True when the active bucket changed.
        """
        prov = str(provider).strip().lower()
        with self._lock:
            bucket = self._last_blocked_bucket
            if (
                bucket is None
                or bucket.provider != prov
                or not new_key_env
                or bucket.key_env == new_key_env
            ):
                return False
            self._last_blocked_bucket = BucketKey(
                bucket.provider, str(new_key_env), bucket.pool
            )
        logger.info(
            "Active %s key remapped mid-wait: %s -> %s",
            prov,
            bucket.key_env,
            new_key_env,
        )
        return True

    def describe_pool_block(self) -> str | None:
        """Human-readable per-key pool-exhaustion message, or ``None``.

        Names the exhausted key/pool with its usage against the cap and lists
        the other keys' remaining headroom for the same pool (pulled from the
        cross-tool ledger). ``None`` when the last block was not a per-key pool
        block.
        """
        with self._lock:
            bucket = self._last_blocked_bucket
            if bucket is None or bucket.pool is None:
                return None
            cap = self._pool_cap_for(bucket)
            used = self._bucket_committed_locked(bucket)
            pool = bucket.pool
            provider = bucket.provider
            key_env = bucket.key_env
            shared = self._shared_enabled
            degraded = self._ledger_degraded

        cap_txt = f"{cap:,}" if cap is not None else "n/a"
        msg = (
            f"{key_env} {pool} pool exhausted ({used:,}/{cap_txt}) for "
            f"provider '{provider}'."
        )
        if not shared or degraded:
            return msg + " Cross-tool per-key view unavailable (standalone mode)."

        snapshot = self.read_ledger_usage()
        if snapshot is None:
            return msg + " Cross-tool per-key view unavailable."
        alternatives = []
        for other, other_used in sorted(snapshot.buckets.items()):
            if (
                other.provider == provider
                and other.pool == pool
                and other.key_env != key_env
            ):
                other_cap = self._pool_cap_for(other)
                remaining = (
                    max(0, other_cap - other_used) if other_cap is not None else None
                )
                if remaining is None:
                    alternatives.append(f"{other.key_env}: {other_used:,} used")
                else:
                    alternatives.append(
                        f"{other.key_env} {pool} pool: {other_used:,} used, "
                        f"{remaining:,} remaining"
                    )
        if alternatives:
            msg += " " + " ".join(alternatives)
        return msg

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
        """atexit hook: flush the pending state write and shared-ledger delta.

        Delegates to :meth:`flush`, which pushes any accumulated unsynced ledger
        delta (shared mode never sets ``_pending_write``, so a bare state save
        would drop it) and then persists a pending debounced write. flush() runs
        the final ledger sync inline off the event loop, which holds at
        interpreter shutdown (atexit runs on the main thread with no loop).
        """
        with contextlib.suppress(Exception):
            self.flush()

    def _get_current_date_str(self) -> str:
        """Get the current budget-day key (YYYY-MM-DD), buffered UTC.

        The day rolls over at 00:01 UTC rather than at exact UTC midnight, so
        the reset never fires before OpenAI's own 00:00 UTC free-tier reset
        has actually happened.
        """
        return (datetime.now(UTC) - _RESET_BUFFER).strftime("%Y-%m-%d")

    def _load_state(self) -> None:
        """Load token usage state from disk (including the per-bucket split)."""
        if not self.state_file.exists():
            # No existing state, initialize fresh
            self._current_date = self._get_current_date_str()
            self._tokens_used_today = 0
            self._bucket_used_today = {}
            logger.debug("No existing token state file found, starting fresh")
            return

        try:
            with open(self.state_file, encoding="utf-8") as f:
                state = json.load(f)

            saved_date = state.get("date", "")
            saved_tokens = int(state.get("tokens_used", 0) or 0)

            current_date = self._get_current_date_str()

            if saved_date == current_date:
                # Same day, restore token count and per-bucket split.
                self._current_date = saved_date
                self._tokens_used_today = saved_tokens
                self._bucket_used_today = self._load_buckets(state, saved_tokens)
                logger.info(
                    f"Loaded token state for {current_date}: "
                    f"{self._tokens_used_today:,} tokens used"
                )
            else:
                # Different day, reset counter
                self._current_date = current_date
                self._tokens_used_today = 0
                self._bucket_used_today = {}
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
            self._bucket_used_today = {}

    @staticmethod
    def _load_buckets(state: dict[str, Any], legacy_total: int) -> dict[BucketKey, int]:
        """Parse the ``buckets`` object, adopting a legacy count if absent.

        When the state predates per-bucket accounting the whole ``tokens_used``
        count is adopted into the unattributed bucket, so per-key enforcement
        still works best-effort from the private counts.
        """
        raw = state.get("buckets")
        if not isinstance(raw, dict):
            if legacy_total > 0:
                return {UNATTRIBUTED_BUCKET: legacy_total}
            return {}
        buckets: dict[BucketKey, int] = {}
        for key, value in raw.items():
            bucket = _bucket_from_str(key)
            if bucket is None:
                continue
            amount = int(value) if isinstance(value, (int, float)) else 0
            if amount > 0:
                buckets[bucket] = buckets.get(bucket, 0) + amount
        return buckets

    def _save_state(self) -> None:
        """Save current token usage state to disk (immediate, atomic).

        Writes to a per-process-unique temp file in the same directory before
        an atomic ``replace()``, so concurrent processes can never collide on
        the temp path. The temp file is always removed in the finally block,
        even when the write or replace failed. The legacy ``tokens_used`` int is
        kept for backward compatibility alongside the ``buckets`` split.
        """
        temp_file = self.state_file.with_name(
            f"{self.state_file.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
        )
        try:
            state = {
                "date": self._current_date,
                "tokens_used": self._tokens_used_today,
                "buckets": {
                    _bucket_to_str(bucket): tokens
                    for bucket, tokens in self._bucket_used_today.items()
                    if tokens > 0
                },
                "last_updated": datetime.now().isoformat(),
            }

            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)

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
        """Check if it's a new day and reset counters if needed.

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
            self._bucket_used_today = {}
            self._last_blocked_bucket = None
            self._last_blocked_stamped = False
            if self._shared_enabled:
                # The ledger rolls over internally; reset the local mirror and
                # force a re-seed on the next sync. The private file is left
                # untouched while the shared budget is the active persistence.
                self._unsynced_deltas = {}
                self._combined_total = 0
                self._bucket_totals = {}
                self._seeded = False
            else:
                self._save_state()

    def add_tokens(
        self,
        tokens: int,
        provider: str | None = None,
        key_env: str | None = None,
        model: str | None = None,
    ) -> None:
        """
        Add tokens to the daily count, stamped with the serving key pool.

        Args:
            tokens: Number of tokens to add.
            provider: Lowercase provider id (e.g. "openai") that served the call.
            key_env: NAME of the env var holding the key that served the call
                (never the key value).
            model: Model name, used to derive the OpenAI free-tier pool.
        """
        if not self.enabled or tokens <= 0:
            return

        bucket, _stamped = self._derive_bucket(provider, key_env, model)
        do_ledger_sync = False
        with self._lock:
            self._check_and_reset_if_new_day()
            self._tokens_used_today += tokens
            self._bucket_used_today[bucket] = (
                self._bucket_used_today.get(bucket, 0) + tokens
            )
            # Update the rolling per-call estimate used by try_reserve().
            self._ewma = self._alpha * tokens + (1.0 - self._alpha) * self._ewma

            if self._shared_enabled:
                # Ledger is the active persistence: accumulate the per-bucket
                # delta and decide (under the lock) whether a debounced sync is
                # due. The actual ledger I/O runs outside the lock via _perform.
                self._unsynced_deltas[bucket] = (
                    self._unsynced_deltas.get(bucket, 0) + tokens
                )
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

    def try_reserve(
        self,
        estimate: int | None = None,
        provider: str | None = None,
        key_env: str | None = None,
        model: str | None = None,
    ) -> int | None:
        """Reserve estimated tokens for one chunk/page before launching it.

        The estimate is the larger of the caller-supplied hint (e.g. a tiktoken
        input count for text) and the rolling EWMA of observed per-call usage,
        so the reservation tracks reality and never drops below the average.
        The reservation is charged against the call's bucket, so per-key pool
        caps and the combined cap are both honoured.

        Returns the reserved amount, ``0`` when limiting is disabled (admit
        freely, nothing to release), or ``None`` when the budget cannot cover
        the estimate (caller should stop admitting new work). A non-zero
        reservation must be matched by a later :meth:`release` of the same
        amount and bucket once the call completes.
        """
        if not self.enabled:
            return 0

        # Forced pre-admission refresh when near the combined cap (or not yet
        # seeded). Runs outside the tracker lock; inline off-loop so the fresh
        # combined total is visible to the admission check below.
        self._maybe_forced_refresh_before_admit()

        bucket, stamped = self._derive_bucket(provider, key_env, model)
        with self._lock:
            self._check_and_reset_if_new_day()
            est = max(int(estimate or 0), max(1, round(self._ewma)))
            if self._blocked_for_locked(bucket, stamped, est):
                self._last_blocked_bucket = bucket
                self._last_blocked_stamped = stamped
                return None
            self._tokens_reserved[bucket] = self._tokens_reserved.get(bucket, 0) + est
            if self._last_blocked_bucket == bucket:
                self._last_blocked_bucket = None
                self._last_blocked_stamped = False
            return est

    def release(
        self,
        amount: int,
        provider: str | None = None,
        key_env: str | None = None,
        model: str | None = None,
    ) -> None:
        """Release a reservation made by :meth:`try_reserve` after the call.

        Actual usage is committed separately via :meth:`add_tokens`; releasing
        only frees the transient headroom the reservation was holding, on the
        same bucket it was reserved against.
        """
        if not self.enabled or amount <= 0:
            return

        bucket, _stamped = self._derive_bucket(provider, key_env, model)
        with self._lock:
            new_value = max(0, self._tokens_reserved.get(bucket, 0) - amount)
            if new_value:
                self._tokens_reserved[bucket] = new_value
            else:
                self._tokens_reserved.pop(bucket, None)

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

    def get_tokens_remaining(
        self,
        provider: str | None = None,
        key_env: str | None = None,
        model: str | None = None,
    ) -> int:
        """
        Get the number of tokens remaining for today.

        Without a stamp this is the combined-cap remaining (legacy). With a
        stamp for a pooled bucket it is the tighter of the combined and per-key
        remaining.

        Returns:
            Remaining token count (0 if limit exceeded).
        """
        if not self.enabled:
            return self.daily_limit  # Unlimited

        with self._lock:
            self._check_and_reset_if_new_day()
            bucket, stamped = self._effective_stamp_locked(provider, key_env, model)
            return self._remaining_locked(bucket, stamped)

    def is_limit_reached(
        self,
        provider: str | None = None,
        key_env: str | None = None,
        model: str | None = None,
    ) -> bool:
        """
        Check if the daily token limit has been reached.

        Returns:
            True if limit is reached or exceeded, False otherwise.
        """
        if not self.enabled:
            return False

        return self.get_tokens_remaining(provider, key_env, model) == 0

    def would_block_next_page(
        self,
        provider: str | None = None,
        key_env: str | None = None,
        model: str | None = None,
    ) -> bool:
        """True when the remaining budget cannot cover the current per-chunk
        reservation estimate -- i.e. :meth:`try_reserve` would defer the next
        chunk even though :meth:`is_limit_reached` is still False.

        Mirrors the admission math in try_reserve without mutating reservation
        state, so the wait loop can treat "reservation-blocked near the cap" as
        limit-reached. Without an explicit stamp it uses the last denied
        reservation's bucket (so a per-key pool block is honoured while
        waiting). A disabled tracker never blocks.
        """
        if not self.enabled:
            return False

        with self._lock:
            self._check_and_reset_if_new_day()
            bucket, stamped = self._effective_stamp_locked(provider, key_env, model)
            est = max(1, round(self._ewma))
            return self._blocked_for_locked(bucket, stamped, est)

    def estimate_exceeds_daily_limit(
        self,
        provider: str | None = None,
        key_env: str | None = None,
        model: str | None = None,
    ) -> bool:
        """True when the current per-chunk reservation estimate alone exceeds
        the effective limit for the bucket, so even a full daily reset cannot
        admit the next chunk.

        For a pooled bucket the effective limit is min(daily cap, per-key pool
        cap); for a pool-None bucket under ``scope="pooled"`` there is no limit
        (returns False). A disabled tracker never blocks.
        """
        if not self.enabled:
            return False

        with self._lock:
            self._check_and_reset_if_new_day()
            bucket, stamped = self._effective_stamp_locked(provider, key_env, model)
            est = max(1, round(self._ewma))
            limit = self._effective_limit_for_locked(bucket, stamped)
            if limit is None:
                return False
            return est > limit

    def can_use_tokens(
        self,
        estimated_tokens: int = 0,
        provider: str | None = None,
        key_env: str | None = None,
        model: str | None = None,
    ) -> bool:
        """
        Check if we can use a certain number of tokens.

        Args:
            estimated_tokens: Estimated tokens needed (default: 0 for any usage).
            provider, key_env, model: Optional stamp for per-key enforcement.

        Returns:
            True if we can proceed, False if limit would be exceeded.
        """
        if not self.enabled:
            return True

        remaining = self.get_tokens_remaining(provider, key_env, model)

        if estimated_tokens > 0:
            return remaining >= estimated_tokens
        else:
            # Just check if any tokens remain
            return remaining > 0

    def get_seconds_until_reset(self) -> int:
        """
        Get the number of seconds until the counter resets (00:01 UTC).

        Returns:
            Seconds until the next 00:01 UTC reset.
        """
        now = datetime.now(UTC)
        delta = self.get_reset_time() - now
        return max(0, int(delta.total_seconds()))

    def get_reset_time(self) -> datetime:
        """
        Get the datetime when the counter will reset.

        Returns:
            Timezone-aware UTC datetime of the next 00:01 UTC reset (one
            minute after OpenAI's 00:00 UTC free-tier reset).
        """
        now = datetime.now(UTC)
        anchor = now - _RESET_BUFFER
        # datetime.min.time() is time(0, 0), i.e. midnight, without importing
        # the datetime.time class (which would shadow the stdlib time module
        # imported above for time.monotonic()/time.sleep()).
        next_midnight = datetime.combine(
            anchor.date() + timedelta(days=1), datetime.min.time(), tzinfo=UTC
        )
        return next_midnight + _RESET_BUFFER

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
            "scope": self._scope,
            "per_key_pool_caps_enabled": self._per_key_pool_caps_enabled,
            "pool_caps": (
                {
                    "defaults": dict(DEFAULT_POOL_CAPS),
                    "overrides": {
                        f"{prov}:{label}": cap
                        for (prov, label), cap in self._pool_caps.items()
                    },
                }
                if self._per_key_pool_caps_enabled
                else {}
            ),
            "buckets": self._bucket_stats(),
        }
        stats.update(self._shared_stats())
        return stats

    def _bucket_stats(self) -> list[dict[str, Any]]:
        """Per-bucket used/remaining rows (own and cross-tool where available).

        Only meaningful when per-key caps are active; returns an empty list when
        they are disabled so callers see no per-key rows.
        """
        if not self._per_key_pool_caps_enabled:
            return []
        with self._lock:
            keys = (
                set(self._bucket_used_today)
                | set(self._bucket_totals)
                | set(self._unsynced_deltas)
                | set(self._tokens_reserved)
            )
            keys.discard(UNATTRIBUTED_BUCKET)
            rows = []
            for bucket in sorted(
                keys, key=lambda b: (b.provider, b.key_env, b.pool or "")
            ):
                cap = self._pool_cap_for(bucket)
                cross_tool = (
                    self._bucket_totals.get(bucket, 0)
                    if self._shared_enabled and not self._ledger_degraded
                    else None
                )
                rows.append(
                    {
                        "provider": bucket.provider,
                        "key_env": bucket.key_env,
                        "pool": bucket.pool,
                        "own_used": self._bucket_used_today.get(bucket, 0),
                        "cross_tool_used": cross_tool,
                        "reserved": self._tokens_reserved.get(bucket, 0),
                        "pool_cap": cap,
                        "remaining": (
                            max(0, cap - self._bucket_usage_locked(bucket))
                            if cap is not None
                            else None
                        ),
                    }
                )
            return rows

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


def _describe_reset_time(reset_time: datetime) -> str:
    """Render an aware-UTC reset instant for user-facing messages.

    The actual reset always happens at 00:01 UTC regardless of the local
    offset, so the UTC anchor is always shown alongside the more readable
    local wall-clock time.
    """
    local = reset_time.astimezone()
    return f"{local.strftime('%Y-%m-%d %H:%M:%S')} local (00:01 UTC)"


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


def _resolve_openai_key_env_fresh() -> str | None:
    """Resolve the active OpenAI key env-var NAME fresh from disk.

    Forces a config reload so a mid-wait remap of ``api_keys_config.yaml``
    (``openai: OPENAI_API_KEY_2``) is picked up; the reload also refreshes the
    global config cache, so the next extractor built at a file/chunk boundary
    resolves the new key. Returns ``None`` on any failure.
    """
    try:
        from modules.config.loader import get_config_loader, resolve_api_key_env_var

        # Refresh the shared cache so subsequent client construction sees it.
        get_config_loader(force_reload=True)
        return resolve_api_key_env_var("openai")
    except Exception:
        return None


async def check_and_wait_for_token_limit(
    ui: Any = None, logger: Any = None, reservation_aware: bool = False
) -> bool:
    """
    Check if daily token limit is reached and wait until next day if needed.

    This is the canonical implementation - do not duplicate in main scripts.

    Args:
        ui: Optional UserInterface instance for user feedback.
        logger: Optional logger instance. If None, uses module logger.
        reservation_aware: When True, treat "remaining budget < the current
            per-chunk reservation estimate" as limit-reached (mirrors admission
            control via :meth:`DailyTokenTracker.would_block_next_page`), so the
            wait actually waits while chunks are reservation-blocked near the
            cap rather than returning instantly. The default (False) keeps the
            plain :meth:`is_limit_reached` semantics used by the per-file
            pre-gate callers.

    Returns:
        True if processing can continue. False if the wait cannot help: either
        the user cancelled (Ctrl+C) or -- for reservation-aware callers -- the
        per-chunk estimate exceeds the entire daily limit, so no reset frees
        enough budget. Callers must treat False as an honest give-up (mark the
        item partial/failed), never as success.
    """

    _logger = globals().get("logger") if logger is None else logger

    token_tracker = get_token_tracker()

    def _still_blocked() -> bool:
        # Reservation-aware callers (the mid-document re-pass loop) must keep
        # waiting while admission control defers chunks near the cap; plain
        # callers only wait once the budget is fully spent.
        if reservation_aware:
            return token_tracker.would_block_next_page()
        return token_tracker.is_limit_reached()

    if not token_tracker.enabled or not _still_blocked():
        return True

    # Fast-fail: if a single chunk's estimate exceeds the whole daily limit, a
    # reset cannot admit it -- waiting would burn ~48 h (two useless resets)
    # before the caller's stalled-resets safeguard fires. Give up now.
    if reservation_aware and token_tracker.estimate_exceeds_daily_limit():
        daily_limit = int(token_tracker.get_stats().get("daily_limit", 0))
        if _logger:
            _logger.warning(
                "Per-chunk token estimate exceeds the entire daily limit (%s); "
                "a daily reset cannot admit the next chunk. Not waiting.",
                f"{daily_limit:,}",
            )
        if ui:
            ui.print_warning(
                "A single chunk's token estimate exceeds the entire daily "
                "budget; not waiting. Raise daily_tokens to process the "
                "remaining chunks."
            )
        return False

    # Token limit reached - need to wait until next day
    stats = token_tracker.get_stats()
    reset_time = token_tracker.get_reset_time()
    seconds_until_reset = token_tracker.get_seconds_until_reset()

    # A per-key pool block (one OpenAI key's pool is exhausted while the combined
    # cap is not) names the exhausted key/pool and any alternative keys.
    pool_msg = None
    try:
        pool_msg = token_tracker.describe_pool_block()
    except Exception:  # pragma: no cover - defensive
        pool_msg = None

    if _logger:
        if pool_msg:
            _logger.warning("Per-key token pool exhausted: %s", pool_msg)
        else:
            _logger.warning(
                f"Daily token limit reached: "
                f"{stats['tokens_used_today']:,}/{stats['daily_limit']:,} tokens used"
            )
        _logger.info(
            f"Waiting until {_describe_reset_time(reset_time)} "
            f"({seconds_until_reset // 3600}h "
            f"{(seconds_until_reset % 3600) // 60}m) for token limit reset..."
        )

    if ui:
        if pool_msg:
            ui.print_warning(f"\n⚠ Per-key token pool exhausted: {pool_msg}")
        else:
            ui.print_warning(
                f"\n⚠ Daily token limit reached: "
                f"{stats['tokens_used_today']:,}/{stats['daily_limit']:,} tokens used"
            )
        ui.print_info(
            f"Waiting until {_describe_reset_time(reset_time)} for daily reset "
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
            # 00:01 UTC reset is observed while we wait. Runs off the event
            # loop via to_thread; a no-op (and skipped) when the shared budget
            # is disabled, so single-tool waits are unchanged.
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

            # Live re-read of the pool caps and pool definitions: a user
            # raising a per-key cap or remapping pools mid-wait takes effect
            # without a restart. A read failure keeps the current settings.
            try:
                _refresh_pool_config(token_tracker)
            except Exception as exc:
                if _logger:
                    _logger.debug("Could not refresh pool config during wait: %s", exc)

            # Re-resolve the active OpenAI key mapping fresh from disk so a
            # mid-wait remap (openai: OPENAI_API_KEY_2) switches the active
            # bucket and can unblock without a restart. The fresh config load
            # also refreshes the global cache, so the next extractor built at a
            # file/chunk boundary uses the new key.
            try:
                new_key_env = _resolve_openai_key_env_fresh()
                remapped = bool(new_key_env) and token_tracker.rebind_active_key_env(
                    "openai", new_key_env or ""
                )
                if remapped and _logger:
                    _logger.info(
                        "Active OpenAI key remapped to %s during wait; "
                        "the next file/chunk will use it.",
                        new_key_env,
                    )
            except Exception as exc:
                if _logger:
                    _logger.debug(
                        "Could not refresh the active key mapping during wait: %s",
                        exc,
                    )

            if not _still_blocked():
                if _logger:
                    _logger.info("Token limit has been reset. Resuming processing.")
                if ui:
                    ui.print_success("Token limit has been reset. Resuming processing.")
                return True

        # Countdown expired: the reset moment has passed. Re-check rather than
        # assume the budget is free -- if the per-chunk estimate exceeds the
        # whole daily limit, would_block_next_page() stays True even after
        # reset, and returning True here would spin the caller's re-pass loop
        # into a second full-day wait. Return False so the caller gives up
        # honestly.
        if _still_blocked():
            if _logger:
                _logger.warning(
                    "Token limit reset elapsed but the next chunk is still "
                    "blocked (per-chunk estimate exceeds the daily limit). "
                    "Giving up."
                )
            if ui:
                ui.print_warning(
                    "Token limit reset elapsed but the next chunk still cannot "
                    "be admitted; giving up. Raise daily_tokens to continue."
                )
            return False

        if _logger:
            _logger.info("Token limit has been reset. Resuming processing.")
        if ui:
            ui.print_success("\nToken limit has been reset. Resuming processing.")
        return True

    except KeyboardInterrupt:
        if _logger:
            _logger.info("Wait cancelled by user.")
        if ui:
            ui.print_warning("\nWait cancelled by user.")
        return False


def _coerce_cap(raw: Any) -> int | None:
    """Coerce a configured cap value to int (underscores tolerated)."""
    try:
        return int(str(raw).replace("_", ""))
    except (ValueError, TypeError):
        return None


def _parse_pool_caps_config(
    token_limit_config: dict[str, Any],
) -> tuple[
    str, bool, dict[tuple[str, str], int], dict[str, dict[str, tuple[str, ...]]]
]:
    """Parse scope / per-key-pool settings from the daily_token_limit block.

    Returns ``(scope, per_key_enabled, pool_caps, provider_pools)``.
    ``pool_caps`` is keyed by (provider, pool label); ``provider_pools``
    carries configured model-prefix lists (only entries that define
    ``models``). Each pool entry accepts EITHER a bare int (cap only; the
    model list comes from the built-ins for that provider+label) OR a mapping
    with optional ``cap`` and ``models`` keys. An absent ``per_key_pool_caps``
    block leaves per-key caps ENABLED with the built-in defaults;
    ``enabled: false`` turns them off. A bare integer ``daily_tokens`` keeps
    its combined-cap meaning untouched.
    """
    scope_raw = str(token_limit_config.get("scope", "pooled")).strip().lower()
    scope = "all" if scope_raw == "all" else "pooled"

    pkpc = token_limit_config.get("per_key_pool_caps")
    if not isinstance(pkpc, dict):
        # Block absent entirely -> caps apply with default values.
        return scope, True, {}, {}

    per_key_enabled = bool(pkpc.get("enabled", True))
    pool_caps: dict[tuple[str, str], int] = {}
    provider_pools: dict[str, dict[str, tuple[str, ...]]] = {}
    for provider, pools in pkpc.items():
        if provider == "enabled" or not isinstance(pools, dict):
            continue
        prov = str(provider).strip().lower()
        if not prov:
            continue
        for label, entry in pools.items():
            if not isinstance(label, str) or not label.strip():
                continue
            lab = label.strip()
            if isinstance(entry, dict):
                cap = _coerce_cap(entry.get("cap")) if "cap" in entry else None
                if cap is not None:
                    pool_caps[(prov, lab)] = cap
                models = entry.get("models")
                if isinstance(models, list):
                    prefixes = tuple(
                        str(m).strip()
                        for m in models
                        if isinstance(m, str) and m.strip()
                    )
                    if prefixes:
                        provider_pools.setdefault(prov, {})[lab] = prefixes
            else:
                # Bare value: cap only, built-in model list for the label.
                cap = _coerce_cap(entry)
                if cap is not None:
                    pool_caps[(prov, lab)] = cap
    return scope, per_key_enabled, pool_caps, provider_pools


def _refresh_pool_config(tracker: DailyTokenTracker) -> None:
    """Re-read pool caps / definitions fresh from disk and apply them.

    Called by the wait loop each poll so a user editing the
    ``daily_token_limit`` block mid-wait (raising a pool cap, remapping pools,
    changing scope, disabling per-key caps) takes effect without a restart.
    """
    from modules.config.loader import get_config_loader

    concurrency_config = get_config_loader(force_reload=True).get_concurrency_config()
    token_limit_config = (concurrency_config or {}).get("daily_token_limit", {}) or {}
    scope, per_key_enabled, pool_caps, provider_pools = _parse_pool_caps_config(
        token_limit_config
    )
    tracker.set_pool_config(scope, per_key_enabled, pool_caps, provider_pools)


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

                scope, per_key_enabled, pool_caps, provider_pools = (
                    _parse_pool_caps_config(token_limit_config)
                )

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
                    scope=scope,
                    per_key_pool_caps_enabled=per_key_enabled,
                    pool_caps=pool_caps,
                    provider_pools=provider_pools,
                )

    return _tracker_instance
