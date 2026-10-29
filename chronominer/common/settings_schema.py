"""Settings schema: the typed blocks every tool's settings file shares.

:class:`Reader` reads typed values from a parsed settings mapping and names
the key in every error. The block readers parse the shared top-level
sections: ``api_keys``, ``endpoints``, ``timeouts``, ``rate_limits``,
``retry``, ``concurrency``, ``images``, ``prompt_caching``, ``state_dir`` and
``logs_dir``. Each tool decides which sections it accepts and injects its own
code defaults: a block reader takes the tool's base value (a subclass with
other defaults or extra fields included) and returns the same type, and the
keys a tool does not accept stay errors. A missing key keeps the base value.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from .page_types import DEFAULT_PROFILE, PROFILES
from .retry import RetryPolicy
from .settings_loader import SettingsError

__all__ = [
    "ANTHROPIC_CACHE_TTLS",
    "BACKOFF_BASE",
    "BACKOFF_CAP",
    "BACKOFF_MULTIPLIER",
    "CONCURRENCY_KEYS",
    "DEFAULT_API_KEYS",
    "JITTER",
    "LOG_FILE_NAME",
    "PAGE_TIMEOUT_HEADROOM",
    "BlockParser",
    "Concurrency",
    "Decisions",
    "Endpoint",
    "Images",
    "PromptCaching",
    "Reader",
    "RetrySettings",
    "Timeouts",
    "ValueKind",
    "check_blocks",
    "log_file",
    "read_api_keys",
    "read_concurrency",
    "read_decisions",
    "read_endpoints",
    "read_extensions",
    "read_images",
    "read_logs_dir",
    "read_page_timeout",
    "read_prompt_caching",
    "read_rate_limits",
    "read_retry",
    "read_state_dir",
    "read_timeouts",
    "state_directory",
]

DEFAULT_API_KEYS: Mapping[str, str] = MappingProxyType(
    {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }
)
"""The environment variable holding each provider's key."""

LOG_FILE_NAME = "application.log"
ANTHROPIC_CACHE_TTLS = ("5m", "1h")

# Backoff of the retry block's transient-error ladder: 2 s doubling per retry
# plus up to 1 s of jitter, capped at 120 s (a server's Retry-After included).
BACKOFF_BASE = 2.0
BACKOFF_MULTIPLIER = 2.0
BACKOFF_CAP = 120.0
JITTER = (0.0, 1.0)
PAGE_TIMEOUT_HEADROOM = 300.0

type BlockParser = Callable[[Any], Any]
"""Turns the parsed YAML value of an extra top-level block into its setting."""

type ValueKind = Literal["number", "integer", "positive", "flag"]
"""The :class:`Reader` method that reads one value of a flat block."""


class Reader:
    """Typed reads from a parsed settings mapping with errors naming the key.

    *where* names the file in error messages; relative paths resolve against
    *base*, by default the current directory.
    """

    def __init__(self, where: str, base: Path | None = None) -> None:
        self.where = where
        self.base = base if base is not None else Path.cwd()

    def fail(self, path: str, message: str) -> SettingsError:
        """Return the error for the key at *path*."""
        return SettingsError(f"{self.where}: {path}: {message}")

    def section(
        self, raw: Any, path: str, allowed: frozenset[str] | None = None
    ) -> Mapping[str, Any]:
        """Return a mapping section ({} when absent), checking its keys."""
        if raw is None:
            return {}
        if not isinstance(raw, Mapping):
            raise self.fail(path, "must be a mapping")
        if allowed is not None:
            self.keys(raw, allowed, path)
        return raw

    def keys(self, raw: Mapping[str, Any], allowed: frozenset[str], path: str) -> None:
        """Raise when *raw* holds a key outside *allowed*."""
        unknown = sorted(str(key) for key in raw if key not in allowed)
        if unknown:
            raise self.fail(path or "top level", f"unknown keys {', '.join(unknown)}")

    def text(self, raw: Mapping[str, Any], key: str, default: str, path: str) -> str:
        """Return a stripped string."""
        value = raw.get(key)
        if value is None:
            return default
        if not isinstance(value, str):
            raise self.fail(path + key, "must be a string")
        return value.strip()

    def choice(
        self,
        raw: Mapping[str, Any],
        key: str,
        default: str,
        path: str,
        choices: tuple[str, ...],
    ) -> str:
        """Return a string that is one of *choices*."""
        value = self.text(raw, key, default, path)
        if value not in choices:
            raise self.fail(path + key, f"must be one of {', '.join(choices)}")
        return value

    def number(
        self, raw: Mapping[str, Any], key: str, default: float, path: str
    ) -> float:
        """Return a number that is not negative, as a float."""
        value = raw.get(key)
        if value is None:
            return default
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise self.fail(path + key, "must be a number")
        if value < 0:
            raise self.fail(path + key, "must not be negative")
        return float(value)

    def integer(self, raw: Mapping[str, Any], key: str, default: int, path: str) -> int:
        """Return an integer that is not negative."""
        value = raw.get(key)
        if value is None:
            return default
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise self.fail(path + key, "must be a non-negative integer")
        return value

    def positive(
        self, raw: Mapping[str, Any], key: str, default: int, path: str
    ) -> int:
        """Return an integer of at least 1."""
        value = self.integer(raw, key, default, path)
        if value < 1:
            raise self.fail(path + key, "must be at least 1")
        return value

    def flag(self, raw: Mapping[str, Any], key: str, default: bool, path: str) -> bool:
        """Return true or false."""
        value = raw.get(key)
        if value is None:
            return default
        if not isinstance(value, bool):
            raise self.fail(path + key, "must be true or false")
        return value

    def path(self, raw: Mapping[str, Any], key: str, path: str) -> Path | None:
        """Return a path resolved against the base, or None for ''."""
        text = self.text(raw, key, "", path)
        if not text:
            return None
        value = Path(text).expanduser()
        return value if value.is_absolute() else self.base / value

    def command(self, raw: Mapping[str, Any], key: str, path: str) -> str | None:
        """Return an executable: a bare name as given, a path resolved."""
        text = self.text(raw, key, "", path)
        if not text:
            return None
        value = Path(text).expanduser()
        if value.is_absolute() or len(value.parts) == 1:
            return str(value)
        return str(self.base / value)

    def flat(
        self,
        raw: Mapping[str, Any],
        keys: Sequence[tuple[str, ValueKind]],
        base: Any,
        path: str,
    ) -> dict[str, Any]:
        """Return the values of *keys* in order, each read with its kind.

        A missing key keeps the attribute of *base* of the same name.
        """
        return {
            key: getattr(self, kind)(raw, key, getattr(base, key), path)
            for key, kind in keys
        }


def check_blocks(builtin: frozenset[str], blocks: Mapping[str, BlockParser]) -> None:
    """Raise when an extra block reuses a built-in key.

    Raises:
        ValueError: A block name is a built-in key.
    """
    clashes = sorted(builtin.intersection(blocks))
    if clashes:
        raise ValueError(f"built-in settings keys: {', '.join(clashes)}")


def read_extensions(
    reader: Reader, data: Mapping[str, Any], blocks: Mapping[str, BlockParser]
) -> Mapping[str, Any]:
    """Parse each extra block present in *data* with its parser."""
    parsed: dict[str, Any] = {}
    for name, parser in blocks.items():
        if name not in data:
            continue
        try:
            parsed[name] = parser(data[name])
        except Exception as exc:
            raise reader.fail(name, str(exc) or type(exc).__name__) from exc
    return MappingProxyType(parsed)


def read_api_keys(reader: Reader, raw: Any) -> Mapping[str, str]:
    """Return the key variable of each provider; an empty value keeps the default."""
    section = reader.section(raw, "api_keys", frozenset(DEFAULT_API_KEYS))
    return MappingProxyType(
        {
            provider: reader.text(section, provider, default, "api_keys.") or default
            for provider, default in DEFAULT_API_KEYS.items()
        }
    )


@dataclass(frozen=True)
class Endpoint:
    """A named OpenAI-compatible endpoint."""

    name: str
    base_url: str
    api_key_env: str
    supports_vision: bool = True
    supports_schema: bool = False


_ENDPOINT_KEYS = frozenset(
    {"base_url", "api_key_env", "supports_vision", "supports_schema"}
)


def read_endpoints(reader: Reader, raw: Any) -> Mapping[str, Endpoint]:
    """Return the named endpoints; each needs ``base_url`` and ``api_key_env``."""
    section = reader.section(raw, "endpoints")
    endpoints: dict[str, Endpoint] = {}
    for name, value in section.items():
        path = f"endpoints.{name}"
        entry = reader.section(value, path, _ENDPOINT_KEYS)
        base_url = reader.text(entry, "base_url", "", path + ".")
        key_env = reader.text(entry, "api_key_env", "", path + ".")
        if not base_url or not key_env:
            raise reader.fail(path, "needs base_url and api_key_env")
        endpoints[str(name)] = Endpoint(
            name=str(name),
            base_url=base_url,
            api_key_env=key_env,
            supports_vision=reader.flag(entry, "supports_vision", True, path + "."),
            supports_schema=reader.flag(entry, "supports_schema", False, path + "."),
        )
    return MappingProxyType(endpoints)


@dataclass(frozen=True)
class Timeouts:
    """HTTP timeouts in seconds; ``request`` is the read timeout."""

    request: float = 900.0
    connect: float = 10.0
    write: float = 30.0
    pool: float = 30.0


_TIMEOUT_KEYS: tuple[tuple[str, ValueKind], ...] = tuple(
    (name, "number") for name in ("request", "connect", "write", "pool")
)


def read_timeouts(reader: Reader, raw: Any) -> Timeouts:
    """Return the HTTP timeouts."""
    names = frozenset(name for name, _ in _TIMEOUT_KEYS)
    section = reader.section(raw, "timeouts", names)
    base = Timeouts()
    return Timeouts(**reader.flat(section, _TIMEOUT_KEYS, base, "timeouts."))


def read_rate_limits(
    reader: Reader, raw: Any, default: tuple[tuple[int, int], ...]
) -> tuple[tuple[int, int], ...]:
    """Return the ``[requests, seconds]`` windows; *default* when absent."""
    if raw is None:
        return default
    if not isinstance(raw, list) or not raw:
        raise reader.fail("rate_limits", "must be a list of [requests, seconds]")
    limits: list[tuple[int, int]] = []
    for item in raw:
        if (
            not isinstance(item, list | tuple)
            or len(item) != 2
            or not all(isinstance(n, int) and not isinstance(n, bool) for n in item)
            or min(item) <= 0
        ):
            raise reader.fail("rate_limits", f"invalid entry {item!r}")
        limits.append((item[0], item[1]))
    return tuple(limits)


_MULTIPLIERS: Mapping[str, float] = MappingProxyType(
    dict.fromkeys(
        ("rate_limit", "timeout", "connection", "server_error", "other", "validation"),
        BACKOFF_MULTIPLIER,
    )
)


@dataclass(frozen=True)
class RetrySettings:
    """Attempt budgets of failed calls and the wall-clock limit per unit.

    ``max_attempts`` counts every call, the first included; timeouts and
    validation failures have their own smaller budgets. A response with
    fewer than ``min_input_tokens`` input tokens for an image request is
    retried (0 disables the check). ``page_timeout`` is ``auto`` (the request
    timeout times the timeout budget plus 300 s), 0 (off) or seconds. A
    tool with other budgets subclasses this block with its own defaults.
    """

    max_attempts: int = 8
    timeout_attempts: int = 3
    validation_attempts: int = 1
    min_input_tokens: int = 0
    page_timeout: float | Literal["auto"] = "auto"

    def policy(self) -> RetryPolicy:
        """Return the retry policy of the transient-error ladder."""
        return RetryPolicy(
            max_attempts=self.max_attempts,
            validation_attempts=self.validation_attempts,
            backoff_base=BACKOFF_BASE,
            backoff_cap=BACKOFF_CAP,
            multipliers=_MULTIPLIERS,
            jitter_min=JITTER[0],
            jitter_max=JITTER[1],
        )

    def page_deadline(self, request_timeout: float) -> float | None:
        """Return the wall-clock limit of one page in seconds, None when off."""
        if self.page_timeout == "auto":
            attempts = min(self.timeout_attempts, self.max_attempts)
            return request_timeout * attempts + PAGE_TIMEOUT_HEADROOM
        return self.page_timeout or None


_RETRY_KEYS: tuple[tuple[str, ValueKind], ...] = (
    ("max_attempts", "positive"),
    ("timeout_attempts", "positive"),
    ("validation_attempts", "positive"),
    ("min_input_tokens", "integer"),
)


def read_retry[R: RetrySettings](reader: Reader, raw: Any, base: R) -> R:
    """Return the retry block over *base*; a missing or null key keeps its value."""
    allowed = frozenset(key for key, _ in _RETRY_KEYS) | {"page_timeout"}
    section = reader.section(raw, "retry", allowed)
    values = reader.flat(section, _RETRY_KEYS, base, "retry.")
    raw_timeout = section.get("page_timeout")
    page_timeout = (
        base.page_timeout
        if raw_timeout is None
        else read_page_timeout(reader, raw_timeout)
    )
    return replace(base, **values, page_timeout=page_timeout)


def read_page_timeout(reader: Reader, value: Any) -> float | Literal["auto"]:
    """Return ``auto``, 0.0 for off (``off`` or false), or seconds."""
    if value is None or value == "auto":
        return "auto"
    if value is False or value == "off":
        return 0.0
    if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
        raise reader.fail("retry.page_timeout", "must be auto, off or seconds")
    return float(value)


@dataclass(frozen=True)
class Concurrency:
    """Parallelism outside the model requests and the delay before requests.

    ``image_processing`` is the number of workers of a tool's local image
    preprocessing, such as preparing pages for Tesseract; it does not bound
    the rendering of the page images sent to a model, and a tool without
    such preprocessing ignores it. ``delay_between_tasks`` is the pause in
    seconds each request worker takes before its next request; a tool that
    spaces the starts of all requests instead says so in its subclass. A
    tool with further workers subclasses this block.
    """

    image_processing: int = 4
    delay_between_tasks: float = 0.5


CONCURRENCY_KEYS: tuple[tuple[str, ValueKind], ...] = (
    ("image_processing", "positive"),
    ("delay_between_tasks", "number"),
)
"""The keys of the shared concurrency block, in reading order."""


def read_concurrency[C: Concurrency](
    reader: Reader,
    raw: Any,
    base: C,
    keys: Sequence[tuple[str, ValueKind]] = CONCURRENCY_KEYS,
) -> C:
    """Return the concurrency block over *base*.

    *keys* lists the accepted keys in reading order; a tool whose block has
    further fields passes them with the shared ones.
    """
    section = reader.section(raw, "concurrency", frozenset(key for key, _ in keys))
    return replace(base, **reader.flat(section, keys, base, "concurrency."))


@dataclass(frozen=True)
class Images:
    """Page image limits besides the run's image options.

    ``original_max_side_px`` limits the long side of OpenAI pages sent at
    ``original`` detail; None takes the model's default limit, if any.
    """

    original_max_side_px: int | None = None


def read_images(reader: Reader, raw: Any) -> Images:
    """Return the page image limits."""
    section = reader.section(raw, "images", frozenset({"original_max_side_px"}))
    if section.get("original_max_side_px") is None:
        return Images()
    side = reader.positive(section, "original_max_side_px", 1, "images.")
    return Images(original_max_side_px=side)


@dataclass(frozen=True)
class PromptCaching:
    """Provider prompt caching of the static prompt part.

    With ``enabled``, Anthropic system prompts, direct and through
    OpenRouter (``openrouter_cache_control``), carry a cache breakpoint with
    ``anthropic_ttl``. OpenAI caches without one; a set ``openai_retention``
    sends the prompt-cache parameter the model takes, None sends none. A
    tool with other defaults subclasses this block.
    """

    enabled: bool = True
    anthropic_ttl: str = "5m"
    openai_retention: str | None = None
    openrouter_cache_control: bool = True


@dataclass(frozen=True)
class Decisions:
    """Settings of a tool's decision-model pass.

    Pages are sent at most ``image_max_px`` on their longer side, at most
    ``pages_per_request`` to a request, and classified under ``profile`` (see
    ``page_types``). ``thresholds`` maps a request fingerprint
    (``decisions.request_fingerprint``) to the probability at which ``apply``
    acts; a request whose fingerprint is not listed runs ``shadow`` only.
    """

    image_max_px: int = 768
    pages_per_request: int = 20
    profile: str = DEFAULT_PROFILE
    timeout_s: float = 120.0
    thresholds: Mapping[str, float] = MappingProxyType({})


def read_decisions[D: Decisions](reader: Reader, raw: Any, base: D) -> D:
    """Return the ``decisions`` block over *base*."""
    keys = frozenset(
        {"image_max_px", "pages_per_request", "profile", "timeout_s", "thresholds"}
    )
    section = reader.section(raw, "decisions", keys)
    path = "decisions."
    raw_thresholds = reader.section(section.get("thresholds"), path + "thresholds")
    thresholds = dict(base.thresholds)
    for fingerprint in raw_thresholds:
        where = f"{path}thresholds.{fingerprint}"
        if raw_thresholds[fingerprint] is None:
            raise reader.fail(where, "needs a number; remove the entry to unset it")
        value = reader.number(raw_thresholds, fingerprint, 1.0, f"{path}thresholds.")
        if not 0.0 < value <= 1.0:
            raise reader.fail(where, "must be above 0 and at most 1")
        thresholds[str(fingerprint)] = value
    return replace(
        base,
        image_max_px=reader.positive(section, "image_max_px", base.image_max_px, path),
        pages_per_request=reader.positive(
            section, "pages_per_request", base.pages_per_request, path
        ),
        profile=reader.choice(section, "profile", base.profile, path, PROFILES),
        timeout_s=reader.number(section, "timeout_s", base.timeout_s, path),
        thresholds=MappingProxyType(thresholds),
    )


def read_prompt_caching[P: PromptCaching](reader: Reader, raw: Any, base: P) -> P:
    """Return the prompt caching block over *base*.

    An empty ``prompt_cache_retention`` sends no retention parameter.
    """
    keys = frozenset({"enabled", "anthropic", "openai", "openrouter"})
    section = reader.section(raw, "prompt_caching", keys)
    path = "prompt_caching"
    anthropic = reader.section(
        section.get("anthropic"), f"{path}.anthropic", frozenset({"ttl"})
    )
    openai = reader.section(
        section.get("openai"), f"{path}.openai", frozenset({"prompt_cache_retention"})
    )
    openrouter = reader.section(
        section.get("openrouter"),
        f"{path}.openrouter",
        frozenset({"anthropic_cache_control"}),
    )
    retention = base.openai_retention
    if "prompt_cache_retention" in openai:
        key = "prompt_cache_retention"
        retention = reader.text(openai, key, "", f"{path}.openai.") or None
    return replace(
        base,
        enabled=reader.flag(section, "enabled", base.enabled, f"{path}."),
        anthropic_ttl=reader.choice(
            anthropic,
            "ttl",
            base.anthropic_ttl,
            f"{path}.anthropic.",
            ANTHROPIC_CACHE_TTLS,
        ),
        openai_retention=retention,
        openrouter_cache_control=reader.flag(
            openrouter,
            "anthropic_cache_control",
            base.openrouter_cache_control,
            f"{path}.openrouter.",
        ),
    )


def read_state_dir(reader: Reader, data: Mapping[str, Any]) -> Path | None:
    """Return ``state_dir`` as written, or None for '' (the default folder)."""
    text = reader.text(data, "state_dir", "", "")
    return Path(text) if text else None


def state_directory(state_dir: Path | None, default_name: str) -> Path:
    """Return the state folder: *state_dir*, else ``~/<default_name>``."""
    if state_dir is not None:
        return state_dir.expanduser()
    return Path.home() / default_name


def read_logs_dir(reader: Reader, data: Mapping[str, Any]) -> Path | None:
    """Return ``logs_dir`` resolved against the reader's base, or None."""
    return reader.path(data, "logs_dir", "")


def log_file(logs_dir: Path | None, default_dir: Path) -> Path:
    """Return the log file: *logs_dir* names a ``.log`` file or its folder.

    Without *logs_dir* the file is ``application.log`` in *default_dir*.
    """
    if logs_dir is None:
        return default_dir / LOG_FILE_NAME
    if logs_dir.suffix == ".log":
        return logs_dir
    return logs_dir / LOG_FILE_NAME
