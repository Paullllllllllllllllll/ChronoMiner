"""Cross-tool shared daily token ledger (schema v2: per-key-pool rows).

One JSON ledger file, shared by every participating ChronoPipeline tool
(ChronoMiner, ChronoTranscriber, AutoExcerpter), accumulates each tool's
committed daily token usage under an OS file lock so that concurrently
running tools enforce shared daily budgets instead of N blind per-tool
budgets.

Schema v2 tracks two levels simultaneously:

- ``tools``: one integer per tool (the v1 field, unchanged semantics),
  always equal to the sum of that tool's ``usage`` rows.
- ``usage``: a list of rows keyed by (tool, provider, key_env, pool),
  where ``key_env`` is the NAME of the environment variable that served
  the call (never the key value) and ``pool`` is the OpenAI free-tier
  pool ("large" | "small") derived from the model name, or ``None`` for
  non-OpenAI providers and local endpoints. Usage a tool cannot
  attribute (legacy v1 counts, un-stamped call sites, v1 writers during
  a mixed-version rollout) lands in a per-tool "unattributed" row via
  reconciliation on every merge, so the day's combined count is never
  lost and ``tools[tool] == sum(rows of tool)`` always holds.

Design contract (kept deliberately small):

- A tool only ever merges deltas into its OWN ``tools`` field and its
  OWN ``usage`` rows, so cross-tool lost updates are impossible and
  same-tool concurrency merges additively.
- All mutation happens under an exclusive OS file lock
  (``msvcrt.locking`` on Windows, ``fcntl.flock`` on POSIX) held on a
  dedicated ``ledger.lock`` file, with a bounded acquisition timeout.
  OS locks die with their process, so a killed tool never wedges the
  ledger.
- Writes are atomic: per-process-unique temp file plus ``replace()``,
  with bounded retries for Windows AV/indexer interference.
- Any I/O failure degrades to standalone mode (methods return ``None``)
  with a single warning; the host tool falls back to its private
  counter and NEVER crashes because of the ledger.
- The day rolls over at 00:01 UTC (one minute after OpenAI's 00:00 UTC
  free-tier reset), matching the per-tool trackers.
- Backward compatibility: a v1 ledger (no ``usage`` key) is adopted in
  place -- per-tool totals are kept and reconciled into "unattributed"
  rows as each tool writes. v1 writers preserve the ``usage`` block
  untouched (verified: v1 ``_read_or_fresh`` keeps unknown keys), and
  the reconciliation absorbs the drift they cause, so mixed-version
  rollout cannot corrupt the file.

This module is vendored byte-identically into every participating repo
and therefore imports nothing from its host package. Do not add host
imports. Bump ``LEDGER_MODULE_VERSION`` on every semantic change and
re-copy the file (and its test) to all sibling repos.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import secrets
import sys
import threading
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import IO, Any, NamedTuple

logger = logging.getLogger(__name__)

LEDGER_SCHEMA_VERSION = 2
LEDGER_MODULE_VERSION = "2.1.0"

# One-minute safety buffer past OpenAI's 00:00 UTC free-tier reset, so the
# ledger never frees its budget before the upstream quota has actually reset.
_RESET_BUFFER = timedelta(minutes=1)

LEDGER_FILENAME = "token_ledger.json"
LOCK_FILENAME = "ledger.lock"
DEFAULT_LEDGER_DIRNAME = ".chronopipeline"

# Sentinel provider/key_env for usage a tool could not attribute to a
# concrete (provider, key_env) pair: legacy v1 totals, un-stamped call
# sites, and drift caused by v1 writers during a mixed-version rollout.
UNATTRIBUTED = "unattributed"

POOL_LARGE = "large"
POOL_SMALL = "small"

# Built-in fallback pool definitions. A "pool" is a named set of models
# that share one daily token allowance per API key; hosts may define
# their own pools (any provider, any label, any model prefix list) in
# tool config and pass them via ``compile_pools``. When no configured
# pools cover a provider, these built-ins apply: they mirror OpenAI's
# complimentary daily token program (two shared pools across the listed
# model families). Matching is longest-prefix against the model name; a
# prefix only matches at a separator boundary so e.g. "gpt-5.5" never
# claims "gpt-5.55", while a dated snapshot like "gpt-4o-2024-08-06"
# lands in gpt-4o's pool. Models with no match derive pool ``None``
# (uncapped bucket).
_LARGE_POOL_MODELS: tuple[str, ...] = (
    "gpt-5.6-sol",
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.2",
    "gpt-5.1",
    "gpt-5.1-codex",
    "gpt-5",
    "gpt-5-codex",
    "gpt-5-chat-latest",
    "gpt-4.5-preview",
    "gpt-4.1",
    "gpt-4o",
    "o3",
    "o1-preview",
    "o1",
)
_SMALL_POOL_MODELS: tuple[str, ...] = (
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "gpt-5.1-codex-mini",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o-mini",
    "o4-mini",
    "o1-mini",
    "codex-mini-latest",
)

# Default per-key daily caps for the built-in OpenAI pools (tokens/day):
# slightly below the program's ~1M/~10M allowances for safety headroom.
# Hosts override caps (and pools) in tool config.
DEFAULT_POOL_CAPS: dict[str, int] = {
    POOL_LARGE: 975_000,
    POOL_SMALL: 9_750_000,
}

# Built-in pool definitions by provider: provider -> pool label -> model
# name prefixes. The generic shape hosts can extend/replace via config.
DEFAULT_PROVIDER_POOLS: dict[str, dict[str, tuple[str, ...]]] = {
    "openai": {
        POOL_LARGE: _LARGE_POOL_MODELS,
        POOL_SMALL: _SMALL_POOL_MODELS,
    },
}

# Compiled pools: provider -> ((prefix, pool), ...) sorted longest-first.
CompiledPools = dict[str, tuple[tuple[str, str], ...]]


def compile_pools(
    provider_pools: Mapping[str, Mapping[str, Any]],
) -> CompiledPools:
    """Compile a provider->pool->model-prefixes mapping for derive_pool.

    Accepts any iterable of string prefixes per pool; non-string entries
    and empty prefixes are dropped (never-crash contract). Prefixes are
    lowercased and sorted longest-first per provider so the most specific
    pool wins.
    """
    compiled: CompiledPools = {}
    for provider, pools in provider_pools.items():
        if not isinstance(provider, str) or not provider.strip():
            continue
        pairs: list[tuple[str, str]] = []
        if not isinstance(pools, Mapping):
            continue
        for pool, prefixes in pools.items():
            if not isinstance(pool, str) or not pool.strip():
                continue
            if isinstance(prefixes, (str, bytes)) or not hasattr(prefixes, "__iter__"):
                continue
            for prefix in prefixes:
                if isinstance(prefix, str) and prefix.strip():
                    pairs.append((prefix.strip().lower(), pool.strip()))
        if pairs:
            compiled[provider.strip().lower()] = tuple(
                sorted(pairs, key=lambda item: len(item[0]), reverse=True)
            )
    return compiled


_DEFAULT_COMPILED_POOLS: CompiledPools = compile_pools(DEFAULT_PROVIDER_POOLS)

_PREFIX_SEPARATORS = ("-", ".", ":", "@")

_LOCK_TIMEOUT_S = 5.0
_LOCK_POLL_S = 0.05
_WRITE_RETRIES = 3
_WRITE_RETRY_DELAY_S = 0.1

if sys.platform == "win32":  # pragma: win32 cover
    import msvcrt

    def _lock_handle(handle: IO[bytes]) -> bool:
        """Try to lock one byte non-blockingly; True on success."""
        try:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False

    def _unlock_handle(handle: IO[bytes]) -> None:
        try:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:  # pragma: no cover - best-effort unlock
            pass

else:  # pragma: win32 no cover
    import fcntl

    def _lock_handle(handle: IO[bytes]) -> bool:
        """Try to flock exclusively, non-blockingly; True on success."""
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False

    def _unlock_handle(handle: IO[bytes]) -> None:
        with contextlib.suppress(OSError):  # pragma: no cover - best effort
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class BucketKey(NamedTuple):
    """Identity of one accounting bucket: which key pool served the call.

    ``provider`` and ``key_env`` are lowercase provider id and the NAME of
    the env var holding the key (never the key value). ``pool`` is
    "large" | "small" for OpenAI free-tier models, else ``None``.
    """

    provider: str
    key_env: str
    pool: str | None


UNATTRIBUTED_BUCKET = BucketKey(UNATTRIBUTED, UNATTRIBUTED, None)


class UsageSnapshot(NamedTuple):
    """Point-in-time view of today's ledger for budget enforcement.

    ``combined`` is the all-tools total (the v1 figure). ``own_total`` is
    the calling tool's total. ``buckets`` aggregates attributed rows
    ACROSS tools per (provider, key_env, pool) -- the figure a per-key
    pool cap is enforced against. ``own_buckets`` is the calling tool's
    per-bucket split (includes its unattributed row).
    """

    combined: int
    own_total: int
    buckets: dict[BucketKey, int]
    own_buckets: dict[BucketKey, int]


def derive_pool(
    provider: str | None,
    model: str | None,
    pools: CompiledPools | None = None,
) -> str | None:
    """Derive the daily-allowance pool label for a model, else ``None``.

    ``pools`` is a compiled provider->prefix table from
    :func:`compile_pools`; when ``None`` (or when it does not cover the
    provider) the built-in defaults apply, so zero-config installs keep
    pool enforcement for providers with a known allowance program. A
    prefix only matches exactly or at a separator boundary. Unknown
    providers, empty models, and unlisted models return ``None``.
    """
    if not provider or not model:
        return None
    provider_key = str(provider).strip().lower()
    table = None
    if pools is not None:
        table = pools.get(provider_key)
    if table is None:
        table = _DEFAULT_COMPILED_POOLS.get(provider_key)
    if not table:
        return None
    name = str(model).strip().lower()
    # Tolerate router-style prefixes such as "openai/gpt-4o".
    if "/" in name:
        name = name.rsplit("/", 1)[-1]
    for prefix, pool in table:
        if name == prefix:
            return pool
        if name.startswith(prefix) and name[len(prefix)] in _PREFIX_SEPARATORS:
            return pool
    return None


def default_ledger_dir() -> Path:
    """Return the default shared ledger directory (``~/.chronopipeline``)."""
    return Path.home() / DEFAULT_LEDGER_DIRNAME


def _today() -> str:
    return (datetime.now(UTC) - _RESET_BUFFER).strftime("%Y-%m-%d")


def _coerce_int(value: Any) -> int:
    """Coerce a stored numeric field to int; anything else becomes 0.

    A hand-edited or corrupt ledger value must not crash the extraction
    call path (never-crash contract).
    """
    return int(value) if isinstance(value, (int, float)) else 0


def _parse_usage_rows(data: Mapping[str, Any]) -> dict[tuple[str, BucketKey], int]:
    """Parse the ``usage`` list into a merge-friendly dict.

    Malformed rows are dropped silently (never-crash contract); rows with
    the same key are summed so a duplicated row cannot double-write.
    """
    rows: dict[tuple[str, BucketKey], int] = {}
    raw = data.get("usage")
    if not isinstance(raw, list):
        return rows
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        tool = entry.get("tool")
        provider = entry.get("provider")
        key_env = entry.get("key_env")
        pool = entry.get("pool")
        if not isinstance(tool, str) or not tool.strip():
            continue
        if not isinstance(provider, str) or not provider.strip():
            continue
        if not isinstance(key_env, str) or not key_env.strip():
            continue
        if pool is not None and not isinstance(pool, str):
            continue
        key = (tool.strip().lower(), BucketKey(provider, key_env, pool))
        rows[key] = rows.get(key, 0) + _coerce_int(entry.get("tokens"))
    return rows


def _serialize_usage_rows(
    rows: Mapping[tuple[str, BucketKey], int],
) -> list[dict[str, Any]]:
    """Serialize rows back to the on-disk list, sorted deterministically."""
    ordered = sorted(
        rows.items(),
        key=lambda item: (
            item[0][0],
            item[0][1].provider,
            item[0][1].key_env,
            item[0][1].pool or "",
        ),
    )
    return [
        {
            "tool": tool,
            "provider": bucket.provider,
            "key_env": bucket.key_env,
            "pool": bucket.pool,
            "tokens": tokens,
        }
        for (tool, bucket), tokens in ordered
        if tokens > 0
    ]


class SharedTokenLedger:
    """Locked read-merge-write access to the shared daily token ledger.

    Thread-safe within a process (internal mutex) and safe across
    processes (OS file lock). All public methods degrade to ``None`` on
    I/O failure instead of raising.
    """

    def __init__(
        self,
        tool_name: str,
        ledger_dir: str | Path | None = None,
        lock_timeout: float = _LOCK_TIMEOUT_S,
    ) -> None:
        if not tool_name or not tool_name.strip():
            raise ValueError("tool_name must be a non-empty string")
        self.tool_name = tool_name.strip().lower()
        directory = (
            Path(str(ledger_dir)).expanduser() if ledger_dir else default_ledger_dir()
        )
        self.ledger_dir = directory
        self.ledger_path = directory / LEDGER_FILENAME
        self.lock_path = directory / LOCK_FILENAME
        self.lock_timeout = lock_timeout
        self._mutex = threading.Lock()
        self._warned_degraded = False

    # ------------------------------------------------------------------
    # Public API -- legacy (v1-compatible, combined totals)
    # ------------------------------------------------------------------

    def sync(self, delta: int) -> int | None:
        """Merge ``delta`` tokens into this tool's field; return combined.

        Legacy un-stamped path: the delta lands in this tool's total and,
        via reconciliation, in its "unattributed" usage row. Returns the
        combined total across all tools, or ``None`` when the ledger is
        unavailable (caller should fall back to its private counter).
        """
        if delta < 0:
            delta = 0

        def mutate(data: dict[str, Any]) -> None:
            tools = data["tools"]
            tools[self.tool_name] = _coerce_int(tools.get(self.tool_name)) + delta

        data = self._locked_mutate(mutate)
        if data is None:
            return None
        return self._sum_tools(data)

    def seed(self, own_committed_today: int) -> int | None:
        """Adopt pre-enable same-day usage exactly once per day.

        Sets this tool's field to ``max(field, own_committed_today)`` so
        enabling the shared budget mid-day never forgets prior usage and
        repeated seeding never double-counts. Returns the combined total
        or ``None`` when degraded.
        """
        own = max(0, int(own_committed_today))

        def mutate(data: dict[str, Any]) -> None:
            tools = data["tools"]
            tools[self.tool_name] = max(_coerce_int(tools.get(self.tool_name)), own)

        data = self._locked_mutate(mutate)
        if data is None:
            return None
        return self._sum_tools(data)

    def read_combined(self) -> int | None:
        """Lock-free combined total for today (display / cheap checks).

        Atomic replacement guarantees no torn reads; the value may lag
        in-flight merges by design. Returns 0 for a stale-dated ledger
        and ``None`` when the file is unreadable or absent.
        """
        try:
            data = json.loads(self.ledger_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if data.get("date") != _today():
            return 0
        return self._sum_tools(data)

    def read_breakdown(self) -> dict[str, int] | None:
        """Per-tool breakdown for today, lock-free; ``None`` if unreadable."""
        try:
            data = json.loads(self.ledger_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if data.get("date") != _today():
            return {}
        tools = data.get("tools")
        if not isinstance(tools, dict):
            return {}
        return {
            str(name): int(value)
            for name, value in tools.items()
            if isinstance(value, (int, float))
        }

    # ------------------------------------------------------------------
    # Public API -- v2 (per-key-pool usage rows)
    # ------------------------------------------------------------------

    def sync_usage(self, deltas: Mapping[BucketKey, int]) -> UsageSnapshot | None:
        """Merge per-bucket deltas into this tool's rows; return a snapshot.

        Each delta (clamped to >= 0) is added to this tool's row for its
        bucket, and the sum of all deltas is added to this tool's total,
        keeping ``tools[tool] == sum(rows of tool)`` in one locked write.
        Returns ``None`` when the ledger is unavailable.
        """
        clean: dict[BucketKey, int] = {}
        for bucket, delta in deltas.items():
            amount = max(0, int(delta))
            if amount > 0:
                clean[BucketKey(*bucket)] = clean.get(BucketKey(*bucket), 0) + amount

        def mutate(data: dict[str, Any]) -> None:
            tools = data["tools"]
            tools[self.tool_name] = _coerce_int(tools.get(self.tool_name)) + sum(
                clean.values()
            )
            rows = _parse_usage_rows(data)
            for bucket, amount in clean.items():
                key = (self.tool_name, bucket)
                rows[key] = rows.get(key, 0) + amount
            data["usage"] = _serialize_usage_rows(rows)

        data = self._locked_mutate(mutate)
        if data is None:
            return None
        return self._snapshot(data)

    def seed_usage(
        self,
        own_total: int,
        own_buckets: Mapping[BucketKey, int] | None = None,
    ) -> UsageSnapshot | None:
        """Adopt pre-enable same-day per-bucket usage exactly once per day.

        Applies ``max(row, seeded)`` per bucket and ``max(total, seeded)``
        to this tool's total, so repeated seeding never double-counts.
        Returns a snapshot or ``None`` when degraded.
        """
        total = max(0, int(own_total))
        seeded: dict[BucketKey, int] = {}
        for bucket, value in (own_buckets or {}).items():
            amount = max(0, int(value))
            if amount > 0:
                seeded[BucketKey(*bucket)] = amount

        def mutate(data: dict[str, Any]) -> None:
            tools = data["tools"]
            tools[self.tool_name] = max(_coerce_int(tools.get(self.tool_name)), total)
            rows = _parse_usage_rows(data)
            for bucket, amount in seeded.items():
                key = (self.tool_name, bucket)
                rows[key] = max(rows.get(key, 0), amount)
            data["usage"] = _serialize_usage_rows(rows)

        data = self._locked_mutate(mutate)
        if data is None:
            return None
        return self._snapshot(data)

    def read_usage(self) -> UsageSnapshot | None:
        """Lock-free usage snapshot for today; ``None`` when unreadable.

        A stale-dated ledger yields an empty snapshot (new day, nothing
        used). The value may lag in-flight merges by design.
        """
        try:
            data = json.loads(self.ledger_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict):
            return None
        if data.get("date") != _today():
            return UsageSnapshot(0, 0, {}, {})
        return self._snapshot(data)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _snapshot(self, data: Mapping[str, Any]) -> UsageSnapshot:
        """Build a UsageSnapshot from parsed ledger data (today's)."""
        rows = _parse_usage_rows(data)
        buckets: dict[BucketKey, int] = {}
        own_buckets: dict[BucketKey, int] = {}
        for (tool, bucket), tokens in rows.items():
            if tool == self.tool_name:
                own_buckets[bucket] = own_buckets.get(bucket, 0) + tokens
            if bucket.provider == UNATTRIBUTED:
                continue
            buckets[bucket] = buckets.get(bucket, 0) + tokens
        tools = data.get("tools")
        own_total = (
            _coerce_int(tools.get(self.tool_name)) if isinstance(tools, dict) else 0
        )
        return UsageSnapshot(
            combined=self._sum_tools(data),
            own_total=own_total,
            buckets=buckets,
            own_buckets=own_buckets,
        )

    def _reconcile_own_rows(self, data: dict[str, Any]) -> None:
        """Force ``tools[tool] == sum(rows of tool)`` for the own tool.

        Any excess of the tool total over the row sum (v1 adoption, drift
        from v1 writers, un-stamped legacy sync) is absorbed into this
        tool's "unattributed" row. If rows exceed the total (hand-edited
        file), the total is raised to the row sum instead -- usage is
        never silently discarded in either direction.
        """
        rows = _parse_usage_rows(data)
        own_rows = {
            key: tokens for key, tokens in rows.items() if key[0] == self.tool_name
        }
        tools = data["tools"]
        total = _coerce_int(tools.get(self.tool_name))
        row_sum = sum(own_rows.values())
        if total > row_sum:
            key = (self.tool_name, UNATTRIBUTED_BUCKET)
            rows[key] = rows.get(key, 0) + (total - row_sum)
        elif row_sum > total:
            tools[self.tool_name] = row_sum
        data["usage"] = _serialize_usage_rows(rows)

    def _locked_mutate(
        self, mutate: Callable[[dict[str, Any]], None]
    ) -> dict[str, Any] | None:
        """Locked read-modify-write; returns post-write data or ``None``.

        Runs ``mutate`` on the freshly-read (or fresh) ledger dict, then
        reconciles this tool's rows against its total before the atomic
        write, so every write path leaves the invariant intact.
        """
        with self._mutex:
            handle = self._acquire_lock()
            if handle is None:
                self._warn_degraded("could not acquire the ledger lock")
                return None
            try:
                data = self._read_or_fresh()
                mutate(data)
                self._reconcile_own_rows(data)
                data["last_updated"] = datetime.now().isoformat()
                self._write_atomic(data)
                return data
            except (OSError, ValueError, TypeError) as exc:
                self._warn_degraded(f"ledger I/O failed: {exc}")
                return None
            finally:
                _unlock_handle(handle)
                with contextlib.suppress(OSError):  # pragma: no cover
                    handle.close()

    def _acquire_lock(self) -> IO[bytes] | None:
        """Open the lock file and acquire the OS lock within the timeout."""
        try:
            self.ledger_dir.mkdir(parents=True, exist_ok=True)
            handle: IO[bytes] = open(self.lock_path, "a+b")  # noqa: SIM115
        except OSError:
            return None
        deadline = time.monotonic() + max(0.0, self.lock_timeout)
        while True:
            if _lock_handle(handle):
                return handle
            if time.monotonic() >= deadline:
                with contextlib.suppress(OSError):  # pragma: no cover
                    handle.close()
                return None
            time.sleep(_LOCK_POLL_S)

    def _read_or_fresh(self) -> dict[str, Any]:
        """Read the ledger, resetting on corruption or day rollover.

        A same-day v1 file (no ``usage`` key) is adopted in place: its
        per-tool totals are kept and the missing rows are supplied by
        reconciliation on this write. Unknown keys are preserved so
        future schema additions survive round-trips.
        """
        try:
            data = json.loads(self.ledger_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return self._fresh()
        except (OSError, ValueError):
            logger.warning(
                "Shared token ledger at %s was unreadable or corrupt; "
                "starting a fresh ledger for today.",
                self.ledger_path,
            )
            return self._fresh()
        if not isinstance(data, dict) or data.get("date") != _today():
            return self._fresh()
        tools = data.get("tools")
        if not isinstance(tools, dict):
            data["tools"] = {}
        if not isinstance(data.get("usage"), list):
            data["usage"] = []
        data["schema_version"] = LEDGER_SCHEMA_VERSION
        return data

    def _fresh(self) -> dict[str, Any]:
        return {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "date": _today(),
            "tools": {},
            "usage": [],
            "last_updated": datetime.now().isoformat(),
        }

    def _write_atomic(self, data: dict[str, Any]) -> None:
        """Write via per-process-unique temp + replace, with retries."""
        temp_path = self.ledger_dir / (
            f"{LEDGER_FILENAME}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
        )
        try:
            temp_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            last_error: OSError | None = None
            for _attempt in range(_WRITE_RETRIES):
                try:
                    temp_path.replace(self.ledger_path)
                    return
                except (PermissionError, FileNotFoundError) as exc:
                    last_error = exc
                    time.sleep(_WRITE_RETRY_DELAY_S)
            if last_error is not None:
                raise last_error
        finally:
            if temp_path.exists():
                with contextlib.suppress(OSError):  # pragma: no cover
                    temp_path.unlink()

    @staticmethod
    def _sum_tools(data: Mapping[str, Any]) -> int:
        tools = data.get("tools")
        if not isinstance(tools, dict):
            return 0
        return sum(
            int(value) for value in tools.values() if isinstance(value, (int, float))
        )

    def _warn_degraded(self, reason: str) -> None:
        if not self._warned_degraded:
            self._warned_degraded = True
            logger.warning(
                "Shared token budget degraded to standalone mode (%s). "
                "This tool falls back to its private daily counter.",
                reason,
            )


__all__ = [
    "DEFAULT_POOL_CAPS",
    "DEFAULT_PROVIDER_POOLS",
    "CompiledPools",
    "compile_pools",
    "LEDGER_FILENAME",
    "LEDGER_MODULE_VERSION",
    "LEDGER_SCHEMA_VERSION",
    "POOL_LARGE",
    "POOL_SMALL",
    "UNATTRIBUTED",
    "UNATTRIBUTED_BUCKET",
    "BucketKey",
    "SharedTokenLedger",
    "UsageSnapshot",
    "default_ledger_dir",
    "derive_pool",
]
