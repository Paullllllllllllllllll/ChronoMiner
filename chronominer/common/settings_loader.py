"""Settings file loading: the real file, else the tracked example.

Resolution: an explicit path wins and must exist. Otherwise the real file is
read silently; when it is missing, the example is read and one informational
line asks the user to copy it; when both are missing, the error names the
real file. An optional ``defaults`` block holds run defaults keyed by option
names; with an option table, unknown keys are an error and values are
converted. Nothing is read or logged at import.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .options import OptionError, OptionTable

__all__ = [
    "LoadedSettings",
    "SettingsError",
    "load_settings",
    "locate_settings",
    "read_yaml_mapping",
]

logger = logging.getLogger(__name__)


class SettingsError(ValueError):
    """A settings file is missing or invalid; the message names the file."""


@dataclass(frozen=True)
class LoadedSettings:
    """The content of one settings file.

    ``data`` holds every top-level key except the defaults block;
    ``defaults`` holds the converted run defaults.
    """

    path: Path
    from_example: bool
    data: Mapping[str, Any] = field(default_factory=dict)
    defaults: Mapping[str, Any] = field(default_factory=dict)


def locate_settings(
    real: Path, example: Path | None = None, explicit: Path | None = None
) -> tuple[Path, bool]:
    """Return the file to read and whether it is the example.

    Raises:
        SettingsError: The explicit file, or both the real file and the
            example, do not exist.
    """
    if explicit is not None:
        if not explicit.is_file():
            raise SettingsError(f"Settings file not found: {explicit}")
        return explicit, False
    if real.is_file():
        return real, False
    if example is not None and example.is_file():
        return example, True
    hint = f" (example {example} is missing too)" if example is not None else ""
    raise SettingsError(f"Settings file not found: {real}{hint}")


def read_yaml_mapping(path: Path) -> dict[str, Any]:
    """Read a YAML file whose top level is a mapping; empty files give {}.

    Raises:
        SettingsError: The file cannot be read or parsed, or is not a mapping.
    """
    try:
        text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
    except OSError as exc:
        raise SettingsError(f"Cannot read settings file {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise SettingsError(f"Invalid YAML in settings file {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise SettingsError(f"Settings file {path} must hold a mapping at the top")
    return {str(key): value for key, value in data.items()}


def load_settings(
    real: Path,
    *,
    example: Path | None = None,
    explicit: Path | None = None,
    options: OptionTable | None = None,
    defaults_key: str = "defaults",
) -> LoadedSettings:
    """Locate, read and split one settings file.

    Raises:
        SettingsError: The file is missing or invalid, or its defaults block
            holds an unknown option or an invalid value.
    """
    path, from_example = locate_settings(real, example, explicit)
    if from_example:
        logger.info(
            "Settings file %s not found; using %s. Copy it to %s and edit it.",
            real,
            path,
            real.name,
        )
    data = read_yaml_mapping(path)
    raw_defaults = data.pop(defaults_key, None)
    if raw_defaults is None:
        raw_defaults = {}
    if not isinstance(raw_defaults, dict):
        raise SettingsError(f"{path}: '{defaults_key}' must be a mapping")
    defaults: Mapping[str, Any] = raw_defaults
    if options is not None:
        try:
            defaults = options.check_defaults(raw_defaults)
        except OptionError as exc:
            raise SettingsError(f"{path}: {defaults_key}: {exc}") from exc
    return LoadedSettings(
        path=path, from_example=from_example, data=data, defaults=dict(defaults)
    )
