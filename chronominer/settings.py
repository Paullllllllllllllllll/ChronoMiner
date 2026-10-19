"""Typed settings: machine values and optional run defaults.

``load`` reads ``--settings FILE`` when given, else ``config/settings.yaml``
(gitignored), else the packaged ``resources/settings.example.yaml``; without
any file the code values apply. Every key is optional and falls back to the
code value; unknown keys are errors. Relative paths in the file resolve
against the current directory. The ``defaults`` block sets run options by
option name of the extract table (see :mod:`chronominer.spec`); the other
commands take the defaults of the options they share with it. A flag
overrides a default, and a default overrides the code value. Tuning knobs,
such as the line-range readjuster thresholds and the image byte caps, are
code constants. Further top-level blocks are accepted only when the caller
passes a parser for them; :attr:`Settings.extensions` holds their parsed
values.

Every block is one of the shared blocks of
:mod:`chronominer.common.settings_schema`, read with this tool's defaults.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from chronominer import spec as specs
from chronominer.common import settings_schema as schema
from chronominer.common.rate_limit import DEFAULT_RATE_LIMITS
from chronominer.common.settings_loader import SettingsError, load_settings
from chronominer.common.settings_schema import (
    DEFAULT_API_KEYS,
    BlockParser,
    Endpoint,
    Images,
    PromptCaching,
    Reader,
    RetrySettings,
    Timeouts,
    check_blocks,
    log_file,
    read_api_keys,
    read_concurrency,
    read_endpoints,
    read_extensions,
    read_images,
    read_logs_dir,
    read_prompt_caching,
    read_rate_limits,
    read_retry,
    read_state_dir,
    read_timeouts,
    state_directory,
)
from chronominer.spec import (
    OPTIONS,
    BatchCancelSpec,
    BatchCheckSpec,
    RangesAdjustSpec,
    RangesGenerateSpec,
    RenderSpec,
    RepairSpec,
    Resolution,
    RunSpec,
    SpecError,
)

__all__ = [
    "DEFAULT_API_KEYS",
    "DEFAULT_LOGS_DIR",
    "DEFAULT_SETTINGS_PATH",
    "EXAMPLE_SETTINGS_PATH",
    "BlockParser",
    "Concurrency",
    "Endpoint",
    "Images",
    "PromptCaching",
    "RetrySettings",
    "STATE_DIRNAME",
    "Settings",
    "SettingsError",
    "SpecError",
    "Timeouts",
    "load",
    "resolve",
    "resolve_batch_cancel",
    "resolve_batch_check",
    "resolve_ranges_adjust",
    "resolve_ranges_generate",
    "resolve_render",
    "resolve_repair",
]

_PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_SETTINGS_PATH = _PACKAGE_DIR.parent / "config" / "settings.yaml"
EXAMPLE_SETTINGS_PATH = _PACKAGE_DIR / "resources" / "settings.example.yaml"
DEFAULT_LOGS_DIR = _PACKAGE_DIR.parent / "logs"
STATE_DIRNAME = ".chronominer"
"""The state folder below the home folder when ``state_dir`` is ''."""

RATE_LIMITS: tuple[tuple[int, int], ...] = DEFAULT_RATE_LIMITS
"""The code default of ``rate_limits``: ``[requests, seconds]`` windows."""


@dataclass(frozen=True)
class Concurrency(schema.Concurrency):
    """The pause before each chunk request; none by default.

    ``delay_between_tasks`` is the pause in seconds each ``--concurrency``
    worker takes before its next request. ChronoMiner has no local image
    preprocessing, so ``image_processing`` is not read.
    """

    delay_between_tasks: float = 0.0


@dataclass(frozen=True)
class Settings:
    """Machine values, the optional run defaults and any extra blocks.

    *extensions* maps the name of each extra top-level block in the file to
    the value its parser returned.
    """

    api_keys: Mapping[str, str] = field(default_factory=lambda: DEFAULT_API_KEYS)
    endpoints: Mapping[str, Endpoint] = field(
        default_factory=lambda: MappingProxyType({})
    )
    timeouts: Timeouts = field(default_factory=Timeouts)
    rate_limits: tuple[tuple[int, int], ...] = RATE_LIMITS
    retry: RetrySettings = field(default_factory=RetrySettings)
    concurrency: Concurrency = field(default_factory=Concurrency)
    images: Images = field(default_factory=Images)
    prompt_caching: PromptCaching = field(default_factory=PromptCaching)
    state_dir: Path | None = None
    logs_dir: Path | None = None
    defaults: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    source: Path | None = None
    extensions: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def state_folder(self) -> Path:
        """Return the folder of the extensions' state files.

        ``state_dir`` names it; without it the folder is ``~/.chronominer``.
        """
        return state_directory(self.state_dir, STATE_DIRNAME)

    def log_file(self) -> Path:
        """Return the log file: ``logs_dir`` names a ``.log`` file or its folder.

        Without ``logs_dir`` the file is ``application.log`` in the
        repository's ``logs`` folder.
        """
        return log_file(self.logs_dir, DEFAULT_LOGS_DIR)

    def api_key_env(self, provider: str, endpoint: str | None = None) -> str:
        """Return the environment variable that holds the key of a provider.

        Raises:
            SettingsError: An unknown endpoint, or a provider without one.
        """
        if endpoint is not None:
            return self.endpoint(endpoint).api_key_env
        try:
            return self.api_keys[provider]
        except KeyError:
            raise SettingsError(
                f"no API key variable for provider {provider!r}"
            ) from None

    def endpoint(self, name: str) -> Endpoint:
        """Return a named endpoint.

        Raises:
            SettingsError: No endpoint of that name.
        """
        try:
            return self.endpoints[name]
        except KeyError:
            known = ", ".join(sorted(self.endpoints)) or "none"
            raise SettingsError(
                f"unknown endpoint {name!r}; the settings define: {known}"
            ) from None

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        *,
        defaults: Mapping[str, Any] | None = None,
        source: Path | None = None,
        blocks: Mapping[str, BlockParser] | None = None,
        base: Path | None = None,
    ) -> Settings:
        """Build settings from parsed YAML; missing keys keep code values.

        *blocks* names the extra top-level blocks to accept, each with its
        parser; a block present in *data* is parsed into :attr:`extensions`.
        Relative paths resolve against *base*, by default the current
        directory.

        Raises:
            SettingsError: An unknown key, a value of the wrong type, or a
                block its parser rejects.
            ValueError: A block name is a built-in key.
        """
        blocks = dict(blocks or {})
        check_blocks(_TOP_KEYS, blocks)
        where = str(source) if source is not None else "settings"
        reader = Reader(where, base)
        reader.keys(data, _TOP_KEYS | frozenset(blocks), "")
        return cls(
            api_keys=read_api_keys(reader, data.get("api_keys")),
            endpoints=read_endpoints(reader, data.get("endpoints")),
            timeouts=read_timeouts(reader, data.get("timeouts")),
            rate_limits=read_rate_limits(reader, data.get("rate_limits"), RATE_LIMITS),
            retry=read_retry(reader, data.get("retry"), RetrySettings()),
            concurrency=read_concurrency(
                reader, data.get("concurrency"), Concurrency()
            ),
            images=read_images(reader, data.get("images")),
            prompt_caching=read_prompt_caching(
                reader, data.get("prompt_caching"), PromptCaching()
            ),
            state_dir=read_state_dir(reader, data),
            logs_dir=read_logs_dir(reader, data),
            defaults=MappingProxyType(dict(defaults or {})),
            source=source,
            extensions=read_extensions(reader, data, blocks),
        )


_TOP_KEYS = frozenset(
    {
        "api_keys",
        "endpoints",
        "timeouts",
        "rate_limits",
        "retry",
        "concurrency",
        "images",
        "prompt_caching",
        "state_dir",
        "logs_dir",
    }
)


def load(
    path: Path | None = None, *, blocks: Mapping[str, BlockParser] | None = None
) -> Settings:
    """Load the settings; an explicit ``path`` (``--settings``) wins.

    Without ``path`` the repository's ``config/settings.yaml`` is read, else
    the packaged example; when neither exists the code values apply.
    *blocks* maps extra top-level block names to their parsers (see
    :meth:`Settings.from_mapping`).

    Raises:
        SettingsError: The file is missing or invalid.
    """
    if path is None and not (
        DEFAULT_SETTINGS_PATH.is_file() or EXAMPLE_SETTINGS_PATH.is_file()
    ):
        return Settings()
    loaded = load_settings(
        DEFAULT_SETTINGS_PATH,
        example=EXAMPLE_SETTINGS_PATH,
        explicit=path,
        options=OPTIONS,
    )
    return Settings.from_mapping(
        loaded.data, defaults=loaded.defaults, source=loaded.path, blocks=blocks
    )


def resolve(settings: Settings, flags: Mapping[str, Any]) -> Resolution[RunSpec]:
    """Return the effective extract spec and the source of each value.

    Precedence: flag, then settings default, then code default. A named
    endpoint must exist in the settings.

    Raises:
        SpecError: The options do not form a valid spec.
        SettingsError: A named endpoint is not defined.
    """
    resolution = specs.resolve_spec(flags, settings.defaults)
    _check_endpoint(settings, resolution.spec.model.endpoint)
    return resolution


def resolve_ranges_generate(
    settings: Settings, flags: Mapping[str, Any]
) -> Resolution[RangesGenerateSpec]:
    """Return the effective ranges generate spec; see :func:`resolve`."""
    return specs.resolve_ranges_generate(flags, settings.defaults)


def resolve_ranges_adjust(
    settings: Settings, flags: Mapping[str, Any]
) -> Resolution[RangesAdjustSpec]:
    """Return the effective ranges adjust spec; see :func:`resolve`."""
    resolution = specs.resolve_ranges_adjust(flags, settings.defaults)
    _check_endpoint(settings, resolution.spec.model.endpoint)
    return resolution


def resolve_repair(
    settings: Settings, flags: Mapping[str, Any]
) -> Resolution[RepairSpec]:
    """Return the effective repair spec; see :func:`resolve`."""
    resolution = specs.resolve_repair(flags, settings.defaults)
    _check_endpoint(settings, resolution.spec.model.endpoint)
    return resolution


def resolve_batch_check(
    settings: Settings, flags: Mapping[str, Any]
) -> Resolution[BatchCheckSpec]:
    """Return the effective batch check spec; see :func:`resolve`."""
    return specs.resolve_batch_check(flags, settings.defaults)


def resolve_batch_cancel(
    settings: Settings, flags: Mapping[str, Any]
) -> Resolution[BatchCancelSpec]:
    """Return the effective batch cancel spec; no option takes a settings default."""
    return specs.resolve_batch_cancel(flags)


def resolve_render(
    settings: Settings, flags: Mapping[str, Any]
) -> Resolution[RenderSpec]:
    """Return the effective render spec; no option takes a settings default."""
    return specs.resolve_render(flags)


def _check_endpoint(settings: Settings, endpoint: str | None) -> None:
    if endpoint is not None:
        settings.endpoint(endpoint)
