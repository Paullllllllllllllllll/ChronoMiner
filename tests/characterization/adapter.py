"""The single entry point through which characterization tests run the tool.

``run_tool(command, *, llm, root, config=None, env=None, batch_service=None,
**keywords)`` runs one ChronoMiner command in process, as a fresh CLI call
would, and returns a ``RunResult``. The case vocabulary names the commands
``extract``, ``generate_line_ranges``, ``adjust_line_ranges``,
``check_batches``, ``cancel_batches`` and ``repair``, and the keywords
``schema``, ``input``, ``chunking``,
``first_n_chunks``, ``tokens``, ``context_window``, ``files`` ...; see
``RUN_TOOL_KEYWORDS``). :func:`build_argv` translates them, and the config
overrides a case names (keyed by ``CONFIG_FILES``), into the argv of
``chronominer`` and the data of a temporary settings file, which the run
reads through ``--settings``. Only ``run_tool`` and ``build_argv`` know how
the tool is invoked.

The settings file carries the per-run configuration: the run
defaults (model ``gpt-4o``, 4,096 output tokens, temperature 0, 160 tokens
per chunk, one request at a time, 72 DPI, JPEG quality 90, grayscale, the
image detail of each provider), a 60-second request timeout, three attempts
per call, no delay between requests and the logs folder inside the run's
config folder. Schemas resolve against the fixture schemas in
``SCHEMAS_DIR``, never the repository's ``schemas/``.

The keywords map to CLI options: ``output_mode`` to ``--layout``;
``first_n_chunks``, ``last_n_chunks`` and ``page_range`` to ``--pages
first:N``, ``last:N`` and ``A-B``; ``chunk_size`` and ``tokens`` to
``--chunk-tokens``; ``concurrency_limit`` to ``--concurrency``; the
``context_image`` switch to ``--context-image auto`` (dropped with
``--context none``, which also suppresses the context image); ``input_type``
to ``--type`` (``image`` to ``images``); a folder input to ``--recursive``
and ``--all``, so runs always search subfolders and process every item;
the schema's ``csv_output`` to ``--formats csv``; a custom provider to a
settings endpoint and ``--endpoint``; ``inject_schema_into_prompt: false``
to ``--response-format schema``. ``check_batches``, ``cancel_batches`` and
``repair`` work on the output folder (``--input``); a repair's ``files``
name ``<stem>_temp.jsonl`` becomes the document's ``<stem>_output.json``.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import io
import json as jsonlib
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pytest
import yaml

from tests.characterization.fakes import CUSTOM_KEY_ENV, DUMMY_KEYS, FakeLLM

if TYPE_CHECKING:
    from tests.characterization.fakes_batch import FakeBatchService

Command = Literal[
    "extract",
    "generate_line_ranges",
    "adjust_line_ranges",
    "check_batches",
    "cancel_batches",
    "repair",
]

SCHEMAS_DIR = Path(__file__).resolve().parent / "schemas"
"""Fixture schemas; ``--schema`` of every run resolves against this folder."""

DATABASE_NAME = "chronominer.sqlite"

DEFAULT_SCHEMA = "FlatEntries"
DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-5",
    "google": "gemini-2.5-flash",
    "openrouter": "openai/gpt-4o",
    "custom": "org/fake-model",
}
"""Model of each provider when a run names a provider but no model."""

CUSTOM_MODEL = DEFAULT_MODELS["custom"]
CUSTOM_BASE_URL = "https://fake-endpoint.invalid/v1/"
CUSTOM_ENDPOINT = "fake"
"""Name of the settings endpoint a custom-endpoint override becomes."""

CONFIG_FILES = (
    "paths_config",
    "model_config",
    "concurrency_config",
    "chunking_and_context",
    "image_processing_config",
    "api_keys_config",
)
"""Override groups ``run_tool`` translates; ``config`` keys name them."""

RUN_DEFAULTS: Mapping[str, Any] = {
    "model": DEFAULT_MODELS["openai"],
    "max_output_tokens": 4096,
    "temperature": 0.0,
    "top_p": 1.0,
    "chunk_tokens": 160,
    "context_window": 6,
    "concurrency": 1,
    "dpi": 72,
    "jpeg_quality": 90,
    "grayscale": True,
}
"""The ``defaults:`` block of every run's settings file.

The image values are those every run sets;
``image_detail`` follows the provider (:data:`IMAGE_DETAILS`).
"""

IMAGE_DETAILS: Mapping[str, str] = {"anthropic": "low"}
"""Image detail per provider; others high."""

_SUBCOMMANDS: dict[str, tuple[str, ...]] = {
    "extract": ("extract",),
    "generate_line_ranges": ("ranges", "generate"),
    "adjust_line_ranges": ("ranges", "adjust"),
    "check_batches": ("batch", "check"),
    "cancel_batches": ("batch", "cancel"),
    "repair": ("repair",),
}

_TYPES = {"image": "images"}
"""Case ``input_type`` values and the ``--type`` ones; others pass unchanged."""

_CHUNKING = {
    "auto": "auto",
    "line_ranges": "line-ranges",
    "adjust-line-ranges": "adjust",
}
"""Case ``chunking`` values and the CLI ones; others pass unchanged."""

_MODEL_FLAGS = (
    "reasoning_effort",
    "verbosity",
    "max_output_tokens",
    "temperature",
    "top_p",
    "service_tier",
)

# Keywords each command accepts. resume (the default), quiet and verbose
# have no effect on the CLI, nor has the cancel command's force.
_KEYWORDS: dict[str, frozenset[str]] = {
    "extract": frozenset(
        {
            "schema",
            "provider",
            "model",
            "input",
            "output",
            "output_mode",
            "input_type",
            "chunking",
            "chunk_size",
            "first_n_chunks",
            "last_n_chunks",
            "page_range",
            "batch",
            "resume",
            "force",
            "dry_run",
            "json",
            "quiet",
            *_MODEL_FLAGS,
            "context",
            "fallback_context",
            "context_image",
            "concurrency_limit",
            "delay",
            "image_detail",
        }
    ),
    "generate_line_ranges": frozenset(
        {"schema", "provider", "model", "input", "tokens", "json", "verbose"}
    ),
    "adjust_line_ranges": frozenset(
        {
            "schema",
            "provider",
            "model",
            "input",
            "context_window",
            "resume",
            "force",
            "json",
            "first_n_chunks",
            "last_n_chunks",
            *(name for name in _MODEL_FLAGS if name != "verbosity"),
        }
    ),
    "check_batches": frozenset(
        {"schema", "provider", "model", "input", "json", "verbose"}
    ),
    "cancel_batches": frozenset({"schema", "provider", "model", "json", "force"}),
    "repair": frozenset(
        {"schema", "provider", "model", "files", "batch", "json", "force", "verbose"}
    ),
}

RUN_TOOL_KEYWORDS: Mapping[str, frozenset[str]] = _KEYWORDS
"""Keywords of ``run_tool`` that each command accepts."""

_ASYNC_SLEEP = asyncio.sleep
HEARTBEAT_TASK = "lease-heartbeat"


@dataclass
class RunResult:
    """Outcome of one ``run_tool`` call.

    ``summary`` is the JSON line the command printed (None without one);
    ``created`` lists the files the run created in ``scan_dirs``; ``calls``
    and ``models`` copy the LLM traffic of this run and ``batch`` the batch
    API records (``FakeBatchService.request_records``) of this run when a
    batch service was given; ``replacements`` then maps every batch and file
    id of the service to a placeholder for the goldens. Console messages are
    on ``stderr``; ``stdout`` holds only the JSON line.
    """

    command: str
    argv: list[str]
    exit_code: int
    summary: dict[str, Any] | None
    stdout: str
    stderr: str
    root: Path
    config_dir: Path
    output_dir: Path
    input_path: Path | None
    scan_dirs: tuple[Path, ...]
    created: list[Path]
    llm: FakeLLM
    schema_name: str
    settings: dict[str, Any] = field(default_factory=dict)
    calls: list[dict[str, Any]] = field(default_factory=list)
    models: list[dict[str, Any]] = field(default_factory=list)
    batch: dict[str, Any] | None = None
    replacements: dict[str, str] = field(default_factory=dict)

    @property
    def console(self) -> str:
        """Return everything the run wrote, stdout and stderr."""
        return self.stdout + self.stderr

    @property
    def database(self) -> Path:
        """Return the store of the run's output folder."""
        return self.output_dir / DATABASE_NAME

    def output_json(self, item: Path) -> Path:
        """Return the ``_output.json`` path of input *item* in the output folder."""
        return self.output_dir / f"{Path(item).stem}_output.json"


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Return *base* with *override* merged in; mappings merge key by key."""
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_schema(name: str) -> dict[str, Any] | None:
    """Return the fixture schema file whose ``name`` is *name*, or None."""
    for path in sorted(SCHEMAS_DIR.glob("*.json")):
        data = jsonlib.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("name") == name:
            return data
    return None


def custom_endpoint(
    model: str = CUSTOM_MODEL, base_url: str = CUSTOM_BASE_URL
) -> dict[str, Any]:
    """Return a ``model_config`` override that routes to the fake custom endpoint.

    Without a ``capabilities`` block the endpoint takes images and enforces
    the schema.
    """
    return {
        "extraction_model": {
            "provider": "custom",
            "name": model,
            "custom_endpoint": {
                "base_url": base_url,
                "api_key_env_var": CUSTOM_KEY_ENV,
            },
        }
    }


class _Invocation:
    """Option values and settings collected from the keywords and overrides."""

    def __init__(self) -> None:
        self.flags: dict[str, Any] = {}
        self.settings: dict[str, Any] = {}

    def flag(self, name: str, value: Any) -> None:
        if name in self.flags and self.flags[name] != value:
            raise ValueError(
                f"conflicting values for --{name.replace('_', '-')}: "
                f"{self.flags[name]!r} and {value!r}"
            )
        self.flags[name] = value

    def setting(self, *path_and_value: Any) -> None:
        *path, value = path_and_value
        node = self.settings
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = value


def _unknown(group: str, keys: Sequence[str]) -> None:
    if keys:
        raise ValueError(f"run_tool cannot translate {group} override(s) {keys}")


def _endpoint_setting(run: _Invocation, endpoint: Mapping[str, Any]) -> None:
    values = dict(endpoint)
    capabilities = dict(values.pop("capabilities", None) or {})
    run.setting(
        "endpoints",
        CUSTOM_ENDPOINT,
        {
            "base_url": values.pop("base_url", CUSTOM_BASE_URL),
            "api_key_env": values.pop("api_key_env_var", CUSTOM_KEY_ENV),
            "supports_vision": bool(capabilities.pop("supports_vision", True)),
            "supports_schema": bool(
                capabilities.pop("supports_structured_output", True)
            ),
        },
    )
    _unknown("custom_endpoint", sorted({*values, *capabilities}))


def _model_override(run: _Invocation, data: Mapping[str, Any]) -> None:
    model = dict(data.get("extraction_model") or {})
    _unknown("model_config", sorted(set(data) - {"extraction_model"}))
    provider = model.pop("provider", None)
    endpoint = model.pop("custom_endpoint", None)
    if provider == "custom":
        _endpoint_setting(run, endpoint or {})
        run.flag("endpoint", CUSTOM_ENDPOINT)
    elif provider is not None:
        run.flag("provider", provider)
    if "name" in model:
        run.flag("model", model.pop("name"))
    if model.pop("inject_schema_into_prompt", True) is False:
        run.flag("response_format", "schema")
    for name in _MODEL_FLAGS:
        if name in model:
            run.flag(name, model.pop(name))
    _unknown("extraction_model", sorted(model))


def _paths_override(
    run: _Invocation, data: Mapping[str, Any], schema_name: str
) -> dict[str, Any]:
    """Translate ``paths_config``; return the schema's path entry."""
    _unknown("paths_config", sorted(set(data) - {"general", "schemas_paths"}))
    general = dict(data.get("general") or {})
    if general.pop("input_paths_is_output_path", False):
        run.flag("beside_input", True)
    _unknown("paths_config.general", sorted(general))
    entries = dict(data.get("schemas_paths") or {})
    entry = dict(entries.pop(schema_name, None) or {})
    _unknown("paths_config.schemas_paths", sorted(entries))
    _unknown("schemas_paths entry", sorted(set(entry) - {"csv_output"}))
    return entry


def _concurrency_override(run: _Invocation, data: Mapping[str, Any]) -> None:
    concurrency = dict(data.get("concurrency") or {})
    _unknown("concurrency_config", sorted(set(data) - {"concurrency"}))
    extraction = dict(concurrency.pop("extraction", None) or {})
    _unknown("concurrency_config.concurrency", sorted(concurrency))
    if "concurrency_limit" in extraction:
        run.flag("concurrency", extraction.pop("concurrency_limit"))
    if "delay_between_tasks" in extraction:
        delay = extraction.pop("delay_between_tasks")
        run.setting("concurrency", "delay_between_tasks", delay)
    retry = dict(extraction.pop("retry", None) or {})
    names = {"attempts": "max_attempts", "timeout_attempts": "timeout_attempts"}
    for key in [key for key in retry if key in names]:
        run.setting("retry", names[key], retry.pop(key))
    _unknown("concurrency retry", sorted(retry))
    _unknown("concurrency extraction", sorted(extraction))


def _translate_config(
    run: _Invocation, config: Mapping[str, Mapping[str, Any]], schema_name: str
) -> bool:
    """Translate config overrides; return whether the CSV view is written."""
    _unknown("config", sorted(set(config) - set(CONFIG_FILES)))
    csv_output = True
    for name, data in config.items():
        if name == "model_config":
            _model_override(run, data)
        elif name == "paths_config":
            entry = _paths_override(run, data, schema_name)
            csv_output = bool(entry.get("csv_output", True))
        elif name == "concurrency_config":
            _concurrency_override(run, data)
        elif name == "api_keys_config":
            for provider, variable in data.items():
                run.setting("api_keys", provider, variable)
        else:
            _unknown(name, sorted(data))
    return csv_output


def _provider_flags(
    run: _Invocation, provider: str | None, model: str | None, *, with_model: bool
) -> None:
    """Pass the provider and its model; a custom provider names the endpoint."""
    if provider == "custom":
        if "endpoint" not in run.flags:
            _endpoint_setting(run, {})
            run.flag("endpoint", CUSTOM_ENDPOINT)
    elif provider is not None:
        run.flag("provider", provider)
    chosen = model
    if chosen is None and provider is not None and "model" not in run.flags:
        chosen = DEFAULT_MODELS[provider]
    if chosen is not None and with_model:
        run.flag("model", chosen)


def _pages(keywords: Mapping[str, Any]) -> str | None:
    given = [
        f"first:{keywords['first_n_chunks']}"
        if keywords.get("first_n_chunks")
        else None,
        f"last:{keywords['last_n_chunks']}" if keywords.get("last_n_chunks") else None,
        keywords.get("page_range"),
    ]
    chosen = [value for value in given if value]
    if len(chosen) > 1:
        raise ValueError(f"one chunk selection per run, not {chosen}")
    return str(chosen[0]) if chosen else None


def _copy_given(
    run: _Invocation, keywords: Mapping[str, Any], names: Sequence[str]
) -> None:
    for name in names:
        if keywords.get(name) is not None:
            run.flag(name, keywords[name])


@dataclass(frozen=True)
class _Places:
    """The resolved paths of a run: input, output folder and recursion."""

    input: Path | None
    output: Path
    recursive: bool


def _extract_source_flags(
    run: _Invocation, keywords: Mapping[str, Any], places: _Places
) -> None:
    if places.input is not None:
        run.flag("input", keywords["input"])
    if places.recursive:
        run.flag("recursive", True)
        run.flag("all", True)
    if "beside_input" not in run.flags:
        run.flag("output", keywords.get("output") or places.output)
    if keywords.get("output_mode") is not None:
        run.flag("layout", keywords["output_mode"])
    if keywords.get("input_type") is not None:
        kind = str(keywords["input_type"])
        run.flag("type", _TYPES.get(kind, kind))


def _extract_chunk_flags(run: _Invocation, keywords: Mapping[str, Any]) -> None:
    if keywords.get("chunking") is not None:
        chunking = str(keywords["chunking"])
        run.flag("chunking", _CHUNKING.get(chunking, chunking))
    if keywords.get("chunk_size") is not None:
        run.flag("chunk_tokens", keywords["chunk_size"])
    pages = _pages(keywords)
    if pages is not None:
        run.flag("pages", pages)


def _extract_flags(
    run: _Invocation, keywords: Mapping[str, Any], places: _Places, csv: bool
) -> None:
    _extract_source_flags(run, keywords, places)
    _extract_chunk_flags(run, keywords)
    if keywords.get("context") is not None:
        run.flag("context", keywords["context"])
    if keywords.get("fallback_context") is not None:
        run.flag("fallback_context", keywords["fallback_context"])
    if keywords.get("context_image") and keywords.get("context") != "none":
        run.flag("context_image", "auto")
    if keywords.get("concurrency_limit") is not None:
        run.flag("concurrency", keywords["concurrency_limit"])
    if keywords.get("delay") is not None:
        run.setting("concurrency", "delay_between_tasks", keywords["delay"])
    if keywords.get("image_detail") is not None:
        run.flag("image_detail", keywords["image_detail"])
    if csv:
        run.flag("formats", ("csv",))
    _copy_given(run, keywords, _MODEL_FLAGS)
    for name in ("batch", "force", "dry_run"):
        if keywords.get(name):
            run.flag(name, True)


def _adjust_flags(
    run: _Invocation, keywords: Mapping[str, Any], places: _Places
) -> None:
    run.flag("input", keywords["input"])
    if places.recursive:
        run.flag("recursive", True)
    pages = _pages(keywords)
    if pages is not None:
        run.flag("pages", pages)
    _copy_given(
        run,
        keywords,
        ("context_window", *(n for n in _MODEL_FLAGS if n != "verbosity")),
    )
    if keywords.get("force"):
        run.flag("force", True)


def _repair_inputs(keywords: Mapping[str, Any], places: _Places) -> Path:
    """Return the ``--input`` of a repair: the output folder or one output file.

    A ``files`` name ``<stem>_temp.jsonl`` names the document whose
    ``<stem>_output.json`` lies in the output folder.
    """
    files = list(keywords.get("files") or ())
    if not files:
        return places.output
    if len(files) > 1:
        raise ValueError("run_tool repairs one named file at a time")
    stem = Path(str(files[0])).name.removesuffix(".jsonl").removesuffix("_temp")
    return places.output / f"{stem}_output.json"


def _generate_flags(
    run: _Invocation, keywords: Mapping[str, Any], places: _Places
) -> None:
    run.flag("input", keywords["input"])
    if places.recursive:
        run.flag("recursive", True)
    if keywords.get("model") is not None:
        run.flag("model", keywords["model"])
    if keywords.get("tokens") is not None:
        run.flag("chunk_tokens", keywords["tokens"])


def _command_flags(
    command: str,
    run: _Invocation,
    keywords: Mapping[str, Any],
    places: _Places,
    csv: bool,
) -> None:
    provider = keywords.get("provider")
    model = keywords.get("model")
    if command == "extract":
        _provider_flags(run, provider, model, with_model=True)
        _extract_flags(run, keywords, places, csv)
    elif command == "generate_line_ranges":
        _generate_flags(run, keywords, places)
    elif command == "adjust_line_ranges":
        _provider_flags(run, provider, model, with_model=True)
        _adjust_flags(run, keywords, places)
    elif command == "check_batches":
        run.flag("input", keywords.get("input") or places.output)
        if csv:
            run.flag("formats", ("csv",))
    elif command == "cancel_batches":
        run.flag("input", places.output)
    else:
        _provider_flags(run, provider, model, with_model=True)
        run.flag("input", _repair_inputs(keywords, places))
        if keywords.get("batch"):
            run.flag("batch", True)
    if keywords.get("schema") is not None and command in {
        "extract",
        "adjust_line_ranges",
        "check_batches",
        "repair",
    }:
        run.flag("schema", keywords["schema"])
    if keywords.get("json"):
        run.flag("json", True)


def _settings_data(config_dir: Path, overrides: Mapping[str, Any]) -> dict[str, Any]:
    base: dict[str, Any] = {
        "timeouts": {"request": 60.0},
        "retry": {"max_attempts": 3, "timeout_attempts": 2},
        "concurrency": {"delay_between_tasks": 0.0},
        "logs_dir": str(config_dir / "logs"),
        "defaults": dict(RUN_DEFAULTS),
    }
    return deep_merge(base, overrides)


def _places(
    command: str, keywords: Mapping[str, Any], root: Path
) -> tuple[_Places, Path]:
    """Return the run's places and the folder holding its inputs."""
    input_path = _absolute(keywords.get("input"), root)
    output = _absolute(keywords.get("output"), root) or (root / "out").resolve()
    if input_path is None:
        input_dir = (root / "input").resolve()
    else:
        input_dir = input_path if input_path.is_dir() else input_path.parent
    recursive = (
        command in {"extract", "generate_line_ranges", "adjust_line_ranges"}
        and input_path is not None
        and input_path.is_dir()
    )
    return _Places(input_path, output, recursive), input_dir


def build_argv(
    command: Command,
    keywords: Mapping[str, Any],
    config: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    root: Path | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Return the argv and the settings data that run *command* with *keywords*.

    *keywords* are ``run_tool`` keywords; *config* the overrides a case
    names. The argv lacks ``--settings``. Relative paths resolve against
    *root* (the current directory by default); the output folder defaults to
    ``<root>/out``.

    Raises:
        TypeError: A keyword the command does not take.
        ValueError: An override with no flag or settings value.
    """
    if command not in _SUBCOMMANDS:
        raise ValueError(f"unknown command {command!r}")
    given = {
        name: value
        for name, value in keywords.items()
        if value is not None and value is not False
    }
    unknown = sorted(set(given) - _KEYWORDS[command])
    if unknown:
        raise TypeError(f"{command} has no flag for {unknown}")
    if command in {"extract", "adjust_line_ranges"}:
        given.setdefault("schema", DEFAULT_SCHEMA)
    run = _Invocation()
    schema_name = str(given.get("schema") or DEFAULT_SCHEMA)
    csv = _translate_config(run, config or {}, schema_name)
    places, _ = _places(command, given, root or Path.cwd())
    _command_flags(command, run, given, places, csv)
    provider = run.flags.get("provider") or (
        "custom" if "endpoint" in run.flags else "openai"
    )
    run.setting("defaults", "image_detail", IMAGE_DETAILS.get(provider, "high"))
    from chronominer.spec import (
        BATCH_CANCEL_OPTIONS,
        BATCH_CHECK_OPTIONS,
        OPTIONS,
        RANGES_ADJUST_OPTIONS,
        RANGES_GENERATE_OPTIONS,
        REPAIR_OPTIONS,
    )

    table = {
        "extract": OPTIONS,
        "generate_line_ranges": RANGES_GENERATE_OPTIONS,
        "adjust_line_ranges": RANGES_ADJUST_OPTIONS,
        "check_batches": BATCH_CHECK_OPTIONS,
        "cancel_batches": BATCH_CANCEL_OPTIONS,
        "repair": REPAIR_OPTIONS,
    }[command]
    return [*_SUBCOMMANDS[command], *table.to_argv(run.flags)], run.settings


def write_settings(folder: Path, data: Mapping[str, Any]) -> Path:
    """Write *data* as the settings file of a run into *folder*."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "settings.yaml"
    path.write_text(
        yaml.safe_dump(dict(data), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
        newline="\n",
    )
    return path


def _files_under(dirs: Sequence[Path]) -> set[Path]:
    found: set[Path] = set()
    for directory in dirs:
        if directory.is_dir():
            found.update(path for path in directory.rglob("*") if path.is_file())
    return found


def _last_json_line(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            parsed = jsonlib.loads(stripped)
        except jsonlib.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _no_sleep(_seconds: float) -> None:
    """Stand-in for ``time.sleep`` that returns at once."""


async def _yield_only(delay: float, result: Any = None) -> Any:
    """Stand-in for ``asyncio.sleep`` that only yields to the event loop.

    The store's lease heartbeat keeps its real interval, so it does not spin.
    """
    task = asyncio.current_task()
    if task is not None and task.get_name() == HEARTBEAT_TASK:
        return await _ASYNC_SLEEP(delay, result)
    await _ASYNC_SLEEP(0)
    return result


def _absolute(value: Any, root: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def run_tool(
    command: Command,
    *,
    llm: FakeLLM,
    root: Path,
    config: Mapping[str, Mapping[str, Any]] | None = None,
    env: Mapping[str, str | None] | None = None,
    batch_service: FakeBatchService | None = None,
    schema: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    input: Path | str | None = None,
    output: Path | str | None = None,
    output_mode: str | None = None,
    input_type: str | None = None,
    chunking: str | None = None,
    chunk_size: int | None = None,
    first_n_chunks: int | None = None,
    last_n_chunks: int | None = None,
    page_range: str | None = None,
    batch: bool = False,
    resume: bool = False,
    force: bool = False,
    dry_run: bool = False,
    json: bool = True,
    quiet: bool = False,
    reasoning_effort: str | None = None,
    verbosity: str | None = None,
    max_output_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    context: Path | str | None = None,
    fallback_context: str | None = None,
    context_image: bool = False,
    concurrency_limit: int | None = None,
    delay: float | None = None,
    image_detail: str | None = None,
    service_tier: str | None = None,
    tokens: int | None = None,
    context_window: int | None = None,
    files: Sequence[str] | None = None,
    verbose: bool = False,
) -> RunResult:
    """Run one ChronoMiner command in process, as a fresh CLI call would.

    *root* is the case root and the working directory, so relative paths
    resolve against it. *input* names the input; *output* the output folder
    (``<root>/out`` by default), which ``check_batches``, ``cancel_batches``
    and ``repair`` search. ``--json`` is passed unless ``json=False``, and
    ``--settings`` names the run's settings file in ``<root>-config``.

    The dummy provider keys in ``DUMMY_KEYS`` are set, then *env* (a None
    value removes a variable). During the run ``time.sleep`` returns at
    once and ``asyncio.sleep`` only yields. The fake LLM must already be
    installed (see ``conftest.fake_llm``); its records are reset first and
    its ``schema`` set to the run's fixture schema. With *batch_service*,
    ``RunResult.batch`` holds the batch API records of this run.
    """
    keywords = {
        name: value
        for name, value in locals().items()
        if name not in {"command", "llm", "root", "config", "env", "batch_service"}
    }
    command_argv, overrides = build_argv(command, keywords, config, root=root)
    config_dir = root.parent / f"{root.name}-config"
    settings = _settings_data(config_dir, overrides)
    settings_path = write_settings(config_dir, settings)
    argv = [*command_argv, "--settings", str(settings_path)]

    places, input_dir = _places(command, keywords, root)
    schema_name = schema or DEFAULT_SCHEMA
    fixture_schema = load_schema(schema_name)
    if fixture_schema is not None:
        llm.schema = fixture_schema

    scan_dirs = tuple(dict.fromkeys([input_dir, places.output]))
    before = _files_under(scan_dirs)
    batch_mark = batch_service.mark() if batch_service is not None else 0
    if batch_service is not None:
        batch_service.reset_records()

    llm.reset_records()
    stdout, stderr = io.StringIO(), io.StringIO()
    with pytest.MonkeyPatch.context() as mp:
        for name, value in {**DUMMY_KEYS, **dict(env or {})}.items():
            if value is None:
                mp.delenv(name, raising=False)
            else:
                mp.setenv(name, value)
        mp.chdir(root)
        mp.setattr(time, "sleep", _no_sleep)
        mp.setattr(asyncio, "sleep", _yield_only)
        import chronominer.spec as spec_module
        from chronominer import __main__ as entry

        mp.setattr(spec_module, "SCHEMAS_DIR", SCHEMAS_DIR)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                code: Any = entry.main(argv)
            except SystemExit as exc:
                code = exc.code
    exit_code = code if isinstance(code, int) else (0 if code is None else 1)

    created = sorted(_files_under(scan_dirs) - before)
    return RunResult(
        command=command,
        argv=argv,
        exit_code=exit_code,
        summary=_last_json_line(stdout.getvalue()),
        stdout=stdout.getvalue(),
        stderr=stderr.getvalue(),
        root=root,
        config_dir=config_dir,
        output_dir=places.output,
        input_path=places.input,
        scan_dirs=scan_dirs,
        created=created,
        llm=llm,
        schema_name=schema_name,
        settings=settings,
        calls=llm.sorted_calls(),
        models=llm.models,
        batch=(
            batch_service.request_records(since=batch_mark)
            if batch_service is not None
            else None
        ),
        replacements=(
            batch_service.id_replacements() if batch_service is not None else {}
        ),
    )


__all__ = [
    "CONFIG_FILES",
    "CUSTOM_BASE_URL",
    "CUSTOM_ENDPOINT",
    "CUSTOM_MODEL",
    "DATABASE_NAME",
    "DEFAULT_MODELS",
    "DEFAULT_SCHEMA",
    "IMAGE_DETAILS",
    "RUN_DEFAULTS",
    "RUN_TOOL_KEYWORDS",
    "SCHEMAS_DIR",
    "Command",
    "RunResult",
    "build_argv",
    "custom_endpoint",
    "deep_merge",
    "load_schema",
    "run_tool",
    "write_settings",
]
