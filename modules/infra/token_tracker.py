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

import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from modules.infra.logger import setup_logger

logger = setup_logger(__name__)

# Default token tracker state-file name. The directory is resolved lazily at
# first use (see _default_token_tracker_file) rather than anchored to the cwd
# at import time, which would bind the path to whatever directory happened to
# be current when this module was first imported.
_TOKEN_TRACKER_FILENAME = ".chronominer_token_state.json"


def _default_token_tracker_file() -> Path:
    """Resolve the default state-file path against the current directory."""
    return Path.cwd() / _TOKEN_TRACKER_FILENAME


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
        """
        self.daily_limit = daily_limit
        self.enabled = enabled
        self.state_file = state_file or _default_token_tracker_file()

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

        # Load existing state from disk
        self._load_state()

        logger.info(
            f"Token tracker initialized: enabled={enabled}, "
            f"daily_limit={daily_limit:,}, "
            f"current_usage={self._tokens_used_today:,}"
        )

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
        """Save current token usage state to disk."""
        try:
            state = {
                "date": self._current_date,
                "tokens_used": self._tokens_used_today,
                "last_updated": datetime.now().isoformat(),
            }

            # Write atomically using a temp file
            temp_file = self.state_file.with_suffix(".tmp")
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)

            # Replace original file
            temp_file.replace(self.state_file)

        except Exception as e:
            logger.error(f"Error saving token state to {self.state_file}: {e}")

    def _check_and_reset_if_new_day(self) -> None:
        """Check if it's a new day and reset counter if needed."""
        current_date = self._get_current_date_str()

        if current_date != self._current_date:
            logger.info(
                f"New day detected: {current_date} (was {self._current_date}). "
                f"Resetting token counter from {self._tokens_used_today:,} to 0."
            )
            self._current_date = current_date
            self._tokens_used_today = 0
            self._save_state()

    def add_tokens(self, tokens: int) -> None:
        """
        Add tokens to the daily count.

        Args:
            tokens: Number of tokens to add.
        """
        if not self.enabled or tokens <= 0:
            return

        with self._lock:
            self._check_and_reset_if_new_day()
            self._tokens_used_today += tokens
            # Update the rolling per-call estimate used by try_reserve().
            self._ewma = self._alpha * tokens + (1.0 - self._alpha) * self._ewma
            self._save_state()

            logger.debug(
                f"Added {tokens:,} tokens. "
                f"Daily total: {self._tokens_used_today:,}/{self.daily_limit:,}"
            )

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

        with self._lock:
            self._check_and_reset_if_new_day()
            est = max(int(estimate or 0), max(1, round(self._ewma)))
            available = (
                self.daily_limit - self._tokens_used_today - self._tokens_reserved
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

        Returns:
            Token count for current day.
        """
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
            remaining = self.daily_limit - self._tokens_used_today
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

        return {
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


def check_token_limit_enabled() -> bool:
    """
    Check if daily token limit is enabled in configuration.

    This is the canonical implementation - do not duplicate in main scripts.

    Returns:
        True if token limiting is enabled, False otherwise.
    """
    tracker = get_token_tracker()
    return tracker.enabled


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
    import asyncio

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

                _tracker_instance = DailyTokenTracker(
                    daily_limit=daily_limit,
                    enabled=enabled,
                    chunk_estimate_seed=seed,
                    estimate_smoothing=smoothing,
                )

    return _tracker_instance
