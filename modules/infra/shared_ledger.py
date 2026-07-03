"""Cross-tool shared daily token ledger.

One JSON ledger file, shared by every participating ChronoPipeline tool
(ChronoMiner, ChronoTranscriber, AutoExcerpter), accumulates each tool's
committed daily token usage under an OS file lock so that concurrently
running tools enforce ONE combined daily budget instead of N blind
per-tool budgets.

Design contract (kept deliberately small):

- The ledger stores one integer per tool under ``tools``; a tool only
  ever merges deltas into its OWN field, so cross-tool lost updates are
  impossible and same-tool concurrency merges additively.
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
- The day rolls over at LOCAL midnight, matching the per-tool trackers.

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
from datetime import datetime
from pathlib import Path
from typing import IO, Any

logger = logging.getLogger(__name__)

LEDGER_SCHEMA_VERSION = 1
LEDGER_MODULE_VERSION = "1.0.0"

LEDGER_FILENAME = "token_ledger.json"
LOCK_FILENAME = "ledger.lock"
DEFAULT_LEDGER_DIRNAME = ".chronopipeline"

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


def default_ledger_dir() -> Path:
    """Return the default shared ledger directory (``~/.chronopipeline``)."""
    return Path.home() / DEFAULT_LEDGER_DIRNAME


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


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
            Path(str(ledger_dir)).expanduser()
            if ledger_dir
            else default_ledger_dir()
        )
        self.ledger_dir = directory
        self.ledger_path = directory / LEDGER_FILENAME
        self.lock_path = directory / LOCK_FILENAME
        self.lock_timeout = lock_timeout
        self._mutex = threading.Lock()
        self._warned_degraded = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sync(self, delta: int) -> int | None:
        """Merge ``delta`` tokens into this tool's field; return combined.

        Under the file lock: read the ledger, roll the day over if the
        stored date is stale, add ``delta`` (>= 0) to this tool's field,
        write atomically, and return the combined total across all
        tools. Returns ``None`` when the ledger is unavailable (caller
        should fall back to its private counter).
        """
        if delta < 0:
            delta = 0
        return self._merge(lambda current: current + delta)

    def seed(self, own_committed_today: int) -> int | None:
        """Adopt pre-enable same-day usage exactly once per day.

        Sets this tool's field to ``max(field, own_committed_today)`` so
        enabling the shared budget mid-day never forgets prior usage and
        repeated seeding never double-counts. Returns the combined total
        or ``None`` when degraded.
        """
        own = max(0, int(own_committed_today))
        return self._merge(lambda current: max(current, own))

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
    # Internals
    # ------------------------------------------------------------------

    def _merge(self, update: Any) -> int | None:
        """Locked read-modify-write applying ``update`` to the own field."""
        with self._mutex:
            handle = self._acquire_lock()
            if handle is None:
                self._warn_degraded("could not acquire the ledger lock")
                return None
            try:
                data = self._read_or_fresh()
                tools = data["tools"]
                current = int(tools.get(self.tool_name, 0) or 0)
                tools[self.tool_name] = int(update(current))
                data["last_updated"] = datetime.now().isoformat()
                self._write_atomic(data)
                return self._sum_tools(data)
            except OSError as exc:
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
        """Read the ledger, resetting on corruption or day rollover."""
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
        data["schema_version"] = LEDGER_SCHEMA_VERSION
        return data

    def _fresh(self) -> dict[str, Any]:
        return {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "date": _today(),
            "tools": {},
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
    def _sum_tools(data: dict[str, Any]) -> int:
        tools = data.get("tools")
        if not isinstance(tools, dict):
            return 0
        return sum(
            int(value)
            for value in tools.values()
            if isinstance(value, (int, float))
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
    "LEDGER_FILENAME",
    "LEDGER_MODULE_VERSION",
    "LEDGER_SCHEMA_VERSION",
    "SharedTokenLedger",
    "default_ledger_dir",
]
