"""Hand-written command lines parsed into the run spec, as the CLI parses them.

Each row parses one subcommand's arguments with the CLI parser, resolves them
with the spec module and compares the spec and the source of every field.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, is_dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from chronominer import __main__ as entry
from chronominer.cli.parser import UsageError, build_parser
from chronominer.common.options import OptionTable, Source
from chronominer.spec import (
    BATCH_CANCEL_OPTIONS,
    BATCH_CHECK_OPTIONS,
    CONTEXT_NONE,
    OPTIONS,
    RANGES_ADJUST_OPTIONS,
    RANGES_GENERATE_OPTIONS,
    RENDER_OPTIONS,
    REPAIR_OPTIONS,
    BatchCancelSpec,
    BatchCheckSpec,
    ChunkingSpec,
    CommandSpec,
    ImageSpec,
    InputSpec,
    ModelSpec,
    OutputSpec,
    PromptSpec,
    RangesAdjustSpec,
    RangesControl,
    RangesGenerateSpec,
    RenderSpec,
    RepairControl,
    RepairSpec,
    Resolution,
    RunControl,
    RunSpec,
    SpecError,
    resolve_batch_cancel,
    resolve_batch_check,
    resolve_ranges_adjust,
    resolve_ranges_generate,
    resolve_render,
    resolve_repair,
    resolve_spec,
)

FLAG = Source.FLAG
SETTINGS = Source.SETTINGS
DEFAULT = Source.DEFAULT

SCHEMA = "price_list_entries"

CODE_MODEL = ModelSpec(
    provider="openai",
    model="gpt-6-luna",
    endpoint=None,
    reasoning_effort="xhigh",
    verbosity="medium",
    max_output_tokens=None,
    temperature=1.0,
    top_p=1.0,
    service_tier="default",
    response_format=None,
)
CODE_DEFAULTS = RunSpec(
    input=InputSpec(
        path=Path("in.txt"), type="auto", files=(), recursive=False, pages=None
    ),
    output=OutputSpec(directory=None, layout="flat", formats=()),
    prompt=PromptSpec(schema=SCHEMA, context=None, context_image=CONTEXT_NONE),
    chunking=ChunkingSpec(mode="auto", chunk_tokens=3500, context_window=6),
    model=CODE_MODEL,
    images=ImageSpec(),
    run=RunControl(batch=False, force=False, concurrency=16, dry_run=False, json=False),
)
GENERATE_DEFAULTS = RangesGenerateSpec(
    input=Path("t.txt"),
    recursive=False,
    chunk_tokens=3500,
    model="gpt-6-luna",
    dry_run=False,
    json=False,
)
ADJUST_DEFAULTS = RangesAdjustSpec(
    input=Path("t.txt"),
    recursive=False,
    schema=SCHEMA,
    context_window=6,
    context=None,
    pages=None,
    model=CODE_MODEL,
    run=RangesControl(force=False, concurrency=16, dry_run=False, json=False),
)
REPAIR_DEFAULTS = RepairSpec(
    input=Path("."),
    schema=None,
    indices=None,
    model=CODE_MODEL,
    images=ImageSpec(),
    run=RepairControl(batch=False, dry_run=False, json=False),
)
CHECK_DEFAULTS = BatchCheckSpec(
    directory=Path("."), batch_ids=(), formats=None, json=False
)
CANCEL_DEFAULTS = BatchCancelSpec(batch_ids=(), all=False, json=False)
RENDER_DEFAULTS = RenderSpec(input=Path("."))

type Resolver = Callable[[Mapping[str, Any], Mapping[str, Any]], Resolution[Any]]


@dataclass(frozen=True)
class Command:
    """A subcommand: its words, option table, resolver and base command line."""

    words: tuple[str, ...]
    table: OptionTable
    resolve: Resolver
    base_argv: tuple[str, ...]
    base: CommandSpec
    base_fields: tuple[str, ...]


COMMANDS: dict[str, Command] = {
    "extract": Command(
        ("extract",),
        OPTIONS,
        resolve_spec,
        ("--input", "in.txt", "--schema", SCHEMA),
        CODE_DEFAULTS,
        ("input.path", "prompt.schema"),
    ),
    "ranges generate": Command(
        ("ranges", "generate"),
        RANGES_GENERATE_OPTIONS,
        resolve_ranges_generate,
        ("--input", "t.txt"),
        GENERATE_DEFAULTS,
        ("input",),
    ),
    "ranges adjust": Command(
        ("ranges", "adjust"),
        RANGES_ADJUST_OPTIONS,
        resolve_ranges_adjust,
        ("--input", "t.txt", "--schema", SCHEMA),
        ADJUST_DEFAULTS,
        ("input", "schema"),
    ),
    "repair": Command(
        ("repair",),
        REPAIR_OPTIONS,
        resolve_repair,
        (),
        REPAIR_DEFAULTS,
        (),
    ),
    "batch check": Command(
        ("batch", "check"),
        BATCH_CHECK_OPTIONS,
        resolve_batch_check,
        (),
        CHECK_DEFAULTS,
        (),
    ),
    "batch cancel": Command(
        ("batch", "cancel"),
        BATCH_CANCEL_OPTIONS,
        lambda flags, _: resolve_batch_cancel(flags),
        (),
        CANCEL_DEFAULTS,
        (),
    ),
    "render": Command(
        ("render",),
        RENDER_OPTIONS,
        lambda flags, _: resolve_render(flags),
        (),
        RENDER_DEFAULTS,
        (),
    ),
}


def _flags(command: Command, argv: tuple[str, ...]) -> dict[str, Any]:
    """Parse *argv* with the CLI parser and return the given options by name."""
    namespace = build_parser().parse_args([*command.words, *argv])
    return command.table.values(namespace)


def _resolve(
    command: Command, argv: tuple[str, ...], defaults: Mapping[str, Any]
) -> Resolution[Any]:
    checked = OPTIONS.check_defaults(defaults)
    return command.resolve(_flags(command, argv), checked)


def _with(spec: Any, changes: Mapping[str, Any]) -> Any:
    """Return *spec* with the dotted-path fields of *changes* replaced."""
    for path, value in changes.items():
        spec = _set(spec, path.split("."), value)
    return spec


def _set(obj: Any, names: list[str], value: Any) -> Any:
    head, *rest = names
    if rest:
        part = getattr(obj, head)
        assert is_dataclass(part)
        value = _set(part, rest, value)
    return replace(obj, **{head: value})


@dataclass(frozen=True)
class Row:
    """One command line and the spec fields it sets.

    ``sets`` take the source FLAG unless ``sources`` names another one; a
    given option resets its alternatives, which take its source.
    """

    command: str
    option: str
    argv: tuple[str, ...]
    sets: Mapping[str, Any]
    defaults: Mapping[str, Any] = field(default_factory=dict)
    sources: Mapping[str, Source] = field(default_factory=dict)


def _row(
    option: str,
    *words: str,
    sets: Mapping[str, Any],
    defaults: Mapping[str, Any] | None = None,
    sources: Mapping[str, Source] | None = None,
    command: str = "extract",
) -> Row:
    argv = (*COMMANDS[command].base_argv, *words)
    return Row(command, option, argv, sets, defaults or {}, sources or {})


def _model(**changes: Any) -> dict[str, Any]:
    return {f"model.{name}": value for name, value in changes.items()}


def _selection(**given: Any) -> dict[str, Any]:
    """Return the fields of the selection: the given one and its reset others."""
    fields = {"select": None, "all": False, "files": (), **given}
    return {f"input.{name}": value for name, value in fields.items()}


ROWS = [
    Row(
        "extract",
        "input",
        ("--input", "a b/c.txt", "--schema", SCHEMA),
        {"input.path": Path("a b/c.txt")},
    ),
    _row("type", "--type", "pdf", sets={"input.type": "pdf"}),
    _row("type", "--type", "images", sets={"input.type": "images"}),
    _row("select", "--select", "1,3-4", sets=_selection(select="1,3-4")),
    _row("select", "--select", "census", sets=_selection(select="census")),
    _row("all", "--all", sets=_selection(all=True)),
    _row(
        "files",
        "--type",
        "text",
        "--files",
        "a.txt",
        "--files",
        "b c.txt,a.txt,d.md",
        sets={"input.type": "text", **_selection(files=("a.txt", "b c.txt", "d.md"))},
    ),
    _row("recursive", "--recursive", sets={"input.recursive": True}),
    _row(
        "recursive",
        "--no-recursive",
        sets={"input.recursive": False},
        defaults={"recursive": True},
    ),
    _row("pages", "--pages", "first:5", sets={"input.pages": "first:5"}),
    _row("pages", "--pages", "1,3,5-8", sets={"input.pages": "1,3,5-8"}),
    _row("pages", "--pages=-7", sets={"input.pages": "-7"}),
    _row("output", "--output", "x y", sets={"output.directory": Path("x y")}),
    _row(
        "beside_input",
        "--beside-input",
        sets={"output.directory": None},
        defaults={"output": "out"},
    ),
    _row("layout", "--layout", "mirror", sets={"output.layout": "mirror"}),
    _row("formats", "--formats", "csv", sets={"output.formats": ("csv",)}),
    _row(
        "formats",
        "--formats",
        "csv",
        "--formats",
        "csv",
        sets={"output.formats": ("csv",)},
    ),
    _row(
        "schema",
        "--schema",
        "WageRecords",
        sets={"prompt.schema": "WageRecords"},
    ),
    _row("schema", "--schema=-s.json", sets={"prompt.schema": "-s.json"}),
    _row("context", "--context", "ctx.txt", sets={"prompt.context": "ctx.txt"}),
    _row(
        "context",
        "--context",
        "prices; wages",
        sets={"prompt.context": "prices; wages"},
    ),
    _row("context", "--context", "none", sets={"prompt.context": CONTEXT_NONE}),
    _row(
        "fallback_context",
        "--fallback-context",
        "notes.txt",
        sets={"prompt.fallback_context": "notes.txt"},
    ),
    _row(
        "fallback_context",
        "--fallback-context",
        "none",
        sets={"prompt.fallback_context": None},
        defaults={"fallback_context": "notes.txt"},
    ),
    _row(
        "context",
        "--context",
        "auto",
        sets={"prompt.context": None},
        defaults={"context": "ctx.txt"},
    ),
    _row(
        "context_image",
        "--context-image",
        "page.png",
        sets={"prompt.context_image": Path("page.png").resolve()},
    ),
    _row(
        "context_image",
        "--context-image",
        "auto",
        sets={"prompt.context_image": None},
    ),
    _row(
        "chunking",
        "--chunking",
        "line-ranges",
        sets={"chunking.mode": "line-ranges"},
    ),
    _row(
        "chunk_tokens",
        "--chunk-tokens",
        "500",
        sets={"chunking.chunk_tokens": 500},
    ),
    _row(
        "context_window",
        "--chunking",
        "adjust",
        "--context-window",
        "9",
        sets={"chunking.mode": "adjust", "chunking.context_window": 9},
    ),
    _row(
        "provider",
        "--provider",
        "openrouter",
        "--model",
        "openai/gpt-5.6-luna",
        sets=_model(provider="openrouter", model="openai/gpt-5.6-luna"),
    ),
    _row(
        "model",
        "--model",
        "gpt-5.6-terra",
        sets=_model(model="gpt-5.6-terra", provider="openai"),
    ),
    _row(
        "model",
        "--model",
        "claude-sonnet-5.5",
        sets=_model(model="claude-sonnet-5-5", provider="anthropic"),
    ),
    _row(
        "model",
        "--model",
        "openrouter:anthropic/claude-sonnet-5-5",
        sets=_model(model="anthropic/claude-sonnet-5-5", provider="openrouter"),
    ),
    _row(
        "model",
        "--provider",
        "google",
        "--model",
        "gemini-3.1-pro",
        sets=_model(model="gemini-3.1-pro", provider="google"),
    ),
    _row(
        "endpoint",
        "--endpoint",
        "local",
        "--model",
        "qwen-vl",
        sets=_model(endpoint="local", provider="custom", model="qwen-vl"),
    ),
    _row(
        "reasoning_effort",
        "--reasoning-effort",
        "low",
        sets=_model(reasoning_effort="low"),
    ),
    _row("verbosity", "--verbosity", "high", sets=_model(verbosity="high")),
    _row(
        "max_output_tokens",
        "--max-output-tokens",
        "8000",
        sets=_model(max_output_tokens=8000),
    ),
    _row("temperature", "--temperature", "0.4", sets=_model(temperature=0.4)),
    _row("top_p", "--top-p", "0.9", sets=_model(top_p=0.9)),
    _row(
        "service_tier",
        "--service-tier",
        "priority",
        sets=_model(service_tier="priority"),
    ),
    _row(
        "response_format",
        "--response-format",
        "json",
        sets=_model(response_format="json"),
    ),
    _row("dpi", "--dpi", "native", sets={"images.dpi": "native"}),
    _row("dpi", "--dpi", "300", sets={"images.dpi": 300}),
    _row("image_detail", "--image-detail", "high", sets={"images.detail": "high"}),
    _row("image_format", "--image-format", "png", sets={"images.format": "png"}),
    _row("jpeg_quality", "--jpeg-quality", "80", sets={"images.jpeg_quality": 80}),
    _row("grayscale", "--grayscale", sets={"images.grayscale": True}),
    _row("grayscale", "--no-grayscale", sets={"images.grayscale": False}),
    _row("batch", "--batch", sets={"run.batch": True}),
    _row(
        "batch",
        "--no-batch",
        sets={"run.batch": False},
        defaults={"batch": True},
    ),
    _row("force", "--force", sets={"run.force": True}),
    _row("concurrency", "--concurrency", "4", sets={"run.concurrency": 4}),
    _row("dry_run", "--dry-run", sets={"run.dry_run": True}),
    _row("json", "--json", sets={"run.json": True}),
    _row("settings", "--settings", "other.yaml", sets={}),
    _row(
        "recursive",
        "--recursive",
        "--chunk-tokens",
        "800",
        "--model",
        "gpt-5.6-terra",
        sets={"recursive": True, "chunk_tokens": 800, "model": "gpt-5.6-terra"},
        command="ranges generate",
    ),
    _row(
        "dry_run",
        "--dry-run",
        "--json",
        sets={"dry_run": True, "json": True},
        command="ranges generate",
    ),
    _row(
        "context_window",
        "--context-window",
        "9",
        "--context",
        "none",
        "--pages",
        "first:3",
        sets={"context_window": 9, "context": CONTEXT_NONE, "pages": "first:3"},
        command="ranges adjust",
    ),
    _row(
        "fallback_context",
        "--fallback-context",
        "entries begin with a date",
        sets={"fallback_context": "entries begin with a date"},
        command="ranges adjust",
    ),
    _row(
        "model",
        "--model",
        "claude-sonnet-5-5",
        "--force",
        "--concurrency",
        "2",
        sets={
            **_model(model="claude-sonnet-5-5", provider="anthropic"),
            "run.force": True,
            "run.concurrency": 2,
        },
        command="ranges adjust",
    ),
    _row(
        "schema",
        "--input",
        "out",
        "--schema",
        "wage_records",
        "--indices",
        "12,0,3-5",
        sets={
            "input": Path("out"),
            "schema": "wage_records",
            "indices": (0, 3, 4, 5, 12),
        },
        command="repair",
    ),
    _row(
        "model",
        "--model",
        "claude-sonnet-5-5",
        "--dpi",
        "200",
        sets={
            **_model(model="claude-sonnet-5-5", provider="anthropic"),
            "images.dpi": 200,
        },
        command="repair",
    ),
    _row(
        "batch",
        "--batch",
        "--dry-run",
        "--json",
        sets={"run.batch": True, "run.dry_run": True, "run.json": True},
        defaults={"concurrency": 4},
        command="repair",
    ),
    _row(
        "batch",
        "--no-batch",
        sets={"run.batch": False},
        defaults={"batch": True},
        command="repair",
    ),
    _row(
        "input",
        "--input",
        "out dir",
        "--schema",
        "wage_records",
        sets={"directory": Path("out dir"), "schema": "wage_records"},
        command="batch check",
    ),
    _row(
        "batch_ids",
        "--batch-ids",
        "b1,b2",
        "--batch-ids",
        "b3",
        sets={"batch_ids": ("b1", "b2", "b3")},
        command="batch check",
    ),
    _row(
        "formats",
        "--formats",
        "csv",
        sets={"formats": ("csv",)},
        command="batch check",
    ),
    _row(
        "adopt",
        "--adopt",
        "msgbatch_1",
        "--json",
        sets={"adopt": "msgbatch_1", "abandon": None, "json": True},
        command="batch check",
    ),
    _row(
        "abandon",
        "--abandon=-s1",
        sets={"abandon": "-s1", "adopt": None},
        command="batch check",
    ),
    _row(
        "batch_ids",
        "--batch-ids=-b1,b2",
        sets={"batch_ids": ("-b1", "b2"), "all": False, "directory": None},
        command="batch cancel",
    ),
    _row(
        "all",
        "--all",
        "--json",
        sets={"all": True, "batch_ids": (), "directory": None, "json": True},
        command="batch cancel",
    ),
    _row(
        "input",
        "--input",
        "jobs",
        sets={"directory": Path("jobs"), "batch_ids": (), "all": False},
        command="batch cancel",
    ),
    _row(
        "input",
        "--input",
        "out/chronominer.sqlite",
        sets={"input": Path("out/chronominer.sqlite")},
        command="render",
    ),
    _row(
        "formats",
        "--formats",
        "csv",
        "--force",
        "--dry-run",
        "--json",
        sets={
            "formats": ("csv",),
            "force": True,
            "dry_run": True,
            "json": True,
        },
        defaults={"formats": []},
        command="render",
    ),
]


def test_every_extract_option_has_a_row() -> None:
    options = {row.option for row in ROWS if row.command == "extract"}
    assert options == set(OPTIONS.names)


@pytest.mark.parametrize("command", list(COMMANDS))
def test_every_subcommand_has_a_row(command: str) -> None:
    assert any(row.command == command for row in ROWS)


@pytest.mark.parametrize(
    "row",
    ROWS,
    ids=[f"{row.command}:{row.option}:{' '.join(row.argv)}" for row in ROWS],
)
def test_a_command_line_sets_its_fields(row: Row) -> None:
    command = COMMANDS[row.command]
    assert row.option in _flags(command, row.argv)

    resolution = _resolve(command, row.argv, row.defaults)

    assert resolution.spec == _with(command.base, row.sets)
    flagged = {*command.base_fields, *row.sets}
    expected = {
        path: row.sources.get(path, FLAG if path in flagged else DEFAULT)
        for path in resolution.sources
    }
    assert dict(resolution.sources) == expected


def test_extract_with_the_code_defaults() -> None:
    resolution = _resolve(COMMANDS["extract"], COMMANDS["extract"].base_argv, {})
    assert resolution.spec == CODE_DEFAULTS
    assert {path for path, s in resolution.sources.items() if s is FLAG} == {
        "input.path",
        "prompt.schema",
    }


def test_the_settings_option_sets_the_settings_path() -> None:
    argv = (*COMMANDS["extract"].base_argv, "--settings", "o.yaml")
    flags = _flags(COMMANDS["extract"], argv)
    assert flags["settings"] == Path("o.yaml")


COMMAND_LINES: list[tuple[tuple[str, ...], dict[str, Any]]] = [
    (
        (
            "--input",
            "scans",
            "--schema",
            "wage_records",
            "--type",
            "images",
            "--files",
            "a,b",
            "--recursive",
            "--model",
            "claude-sonnet-5-5",
            "--formats",
            "csv",
            "--layout",
            "mirror",
            "--context-image",
            "auto",
            "--batch",
            "--dry-run",
            "--json",
        ),
        {
            "input.path": Path("scans"),
            "prompt.schema": "wage_records",
            "input.type": "images",
            "input.files": ("a", "b"),
            "input.recursive": True,
            **_model(model="claude-sonnet-5-5", provider="anthropic"),
            "output.formats": ("csv",),
            "output.layout": "mirror",
            "prompt.context_image": None,
            "run.batch": True,
            "run.dry_run": True,
            "run.json": True,
        },
    ),
    (
        (
            "--input",
            "ledger.txt",
            "--schema",
            "trade_records",
            "--type",
            "text",
            "--chunking",
            "adjust",
            "--context-window",
            "10",
            "--chunk-tokens",
            "2000",
            "--pages",
            "last:2",
            "--output",
            "out",
            "--concurrency",
            "2",
            "--force",
        ),
        {
            "input.path": Path("ledger.txt"),
            "prompt.schema": "trade_records",
            "input.type": "text",
            "input.pages": "last:2",
            "chunking.mode": "adjust",
            "chunking.context_window": 10,
            "chunking.chunk_tokens": 2000,
            "output.directory": Path("out"),
            "run.concurrency": 2,
            "run.force": True,
        },
    ),
]


@pytest.mark.parametrize(
    ("argv", "changes"), COMMAND_LINES, ids=[" ".join(c[0]) for c in COMMAND_LINES]
)
def test_a_full_command_line_gives_the_written_spec(
    argv: tuple[str, ...], changes: dict[str, Any]
) -> None:
    resolution = _resolve(COMMANDS["extract"], argv, {})
    assert resolution.spec == _with(CODE_DEFAULTS, changes)


PRECEDENCE: list[tuple[tuple[str, ...], dict[str, Any], str, Any, Source]] = [
    ((), {}, "run.concurrency", 16, DEFAULT),
    ((), {"concurrency": 4}, "run.concurrency", 4, SETTINGS),
    (("--concurrency", "8"), {"concurrency": 4}, "run.concurrency", 8, FLAG),
    ((), {"formats": ["csv"]}, "output.formats", ("csv",), SETTINGS),
    ((), {"output": "out"}, "output.directory", Path("out"), SETTINGS),
    (("--beside-input",), {"output": "out"}, "output.directory", None, FLAG),
    ((), {"model": "claude-sonnet-5-5"}, "model.provider", "anthropic", SETTINGS),
    (
        ("--model", "gpt-5.6-terra"),
        {"provider": "anthropic"},
        "model.provider",
        "openai",
        FLAG,
    ),
    ((), {"endpoint": "local", "model": "q"}, "model.provider", "custom", SETTINGS),
    (
        ("--provider", "openai"),
        {"endpoint": "local"},
        "model.endpoint",
        None,
        FLAG,
    ),
    ((), {"chunking": "line-ranges"}, "chunking.mode", "line-ranges", SETTINGS),
    (("--chunking", "auto"), {"chunking": "adjust"}, "chunking.mode", "auto", FLAG),
    ((), {"chunk_tokens": 900}, "chunking.chunk_tokens", 900, SETTINGS),
    ((), {"batch": True}, "run.batch", True, SETTINGS),
    (("--no-batch",), {"batch": True}, "run.batch", False, FLAG),
    ((), {"dpi": 300}, "images.dpi", 300, SETTINGS),
    (("--dpi", "native"), {"dpi": 300}, "images.dpi", "native", FLAG),
    ((), {"context": "ctx.txt"}, "prompt.context", "ctx.txt", SETTINGS),
    ((), {"schema": "wage_records"}, "prompt.schema", SCHEMA, FLAG),
    ((), {"reasoning_effort": "low"}, "model.reasoning_effort", "low", SETTINGS),
    (
        ("--reasoning-effort", "xhigh"),
        {"reasoning_effort": "low"},
        "model.reasoning_effort",
        "xhigh",
        FLAG,
    ),
]


@pytest.mark.parametrize(("argv", "defaults", "path", "value", "source"), PRECEDENCE)
def test_precedence_of_flag_settings_default_and_code_default(
    argv: tuple[str, ...],
    defaults: dict[str, Any],
    path: str,
    value: Any,
    source: Source,
) -> None:
    resolution = _resolve(
        COMMANDS["extract"], (*COMMANDS["extract"].base_argv, *argv), defaults
    )
    obj: Any = resolution.spec
    for name in path.split("."):
        obj = getattr(obj, name)
    assert obj == value
    assert resolution.sources[path] is source


def test_extract_takes_the_schema_from_the_settings() -> None:
    resolution = _resolve(
        COMMANDS["extract"], ("--input", "in.txt"), {"schema": "wage_records"}
    )
    assert resolution.spec.prompt.schema == "wage_records"
    assert resolution.sources["prompt.schema"] is SETTINGS


def test_repair_keeps_the_shared_settings_defaults_and_ignores_the_others() -> None:
    defaults = {"model": "gpt-5.6-terra", "batch": True, "concurrency": 4}
    resolution = _resolve(COMMANDS["repair"], (), defaults)
    assert resolution.spec.model.model == "gpt-5.6-terra"
    assert resolution.sources["model.model"] is SETTINGS
    assert resolution.spec.run.batch is True
    assert resolution.sources["run.batch"] is SETTINGS
    assert "run.concurrency" not in resolution.sources


def test_ranges_take_the_shared_settings_defaults() -> None:
    defaults = {"chunk_tokens": 900, "model": "gpt-5.6-terra", "context_window": 9}
    generate = _resolve(COMMANDS["ranges generate"], ("--input", "t.txt"), defaults)
    assert (generate.spec.chunk_tokens, generate.spec.model) == (900, "gpt-5.6-terra")
    assert generate.sources["chunk_tokens"] is SETTINGS
    adjust = _resolve(
        COMMANDS["ranges adjust"], COMMANDS["ranges adjust"].base_argv, defaults
    )
    assert adjust.spec.context_window == 9
    assert adjust.sources["context_window"] is SETTINGS
    assert adjust.spec.model.model == "gpt-5.6-terra"


def test_batch_check_renders_the_run_formats_unless_given() -> None:
    resolution = _resolve(COMMANDS["batch check"], (), {"formats": ["csv"]})
    assert resolution.spec.formats is None
    assert resolution.sources["formats"] is DEFAULT
    given = _resolve(COMMANDS["batch check"], ("--formats", "csv"), {})
    assert given.spec.formats == ("csv",)


SPEC_ERRORS: list[tuple[tuple[str, ...], dict[str, Any], str]] = [
    (("--provider", "anthropic"), {}, "cannot run the model 'gpt-6-luna'"),
    (
        ("--provider", "google"),
        {"model": "claude-sonnet-5-5"},
        "cannot run the model 'claude-sonnet-5-5'",
    ),
    (("--endpoint", "local"), {}, "needs a model of that endpoint"),
    (("--provider", "custom", "--model", "m"), {}, "needs --endpoint NAME"),
]


@pytest.mark.parametrize(("argv", "defaults", "message"), SPEC_ERRORS)
def test_an_invalid_combination_is_a_spec_error(
    argv: tuple[str, ...], defaults: dict[str, Any], message: str
) -> None:
    with pytest.raises(SpecError, match=message):
        _resolve(COMMANDS["extract"], (*COMMANDS["extract"].base_argv, *argv), defaults)


def test_extract_without_a_schema_is_a_spec_error() -> None:
    with pytest.raises(SpecError, match="a schema is required"):
        _resolve(COMMANDS["extract"], ("--input", "in.txt"), {})


@pytest.mark.parametrize(
    "argv",
    [
        ("--provider", "anthropic"),
        ("--provider", "google", "--model", "claude-sonnet-5-5"),
        ("--endpoint", "local"),
    ],
)
def test_an_invalid_combination_exits_2(
    argv: tuple[str, ...], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = str(tmp_path / "in.txt")
    code = entry.main(
        ["extract", "--input", source, "--schema", SCHEMA, *argv, "--dry-run", "--json"]
    )
    assert code == 2
    assert "Invalid arguments" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ("extract", "--input", "x", "--type", "pdfs"),
        ("extract", "--input", "x", "--chunking", "per-file"),
        ("extract", "--input", "x", "--output", "o", "--beside-input"),
        ("extract", "--input", "x", "--formats", "docx"),
        ("extract", "--input", "x", "--response-format", "text"),
        ("ranges", "generate", "--input", "x", "--chunk-tokens", "0"),
        ("ranges", "adjust", "--input", "x", "--context-window", "-1"),
        ("repair", "--indices", "5-2"),
        ("batch", "check", "--adopt", "b1", "--abandon", "s1"),
        ("batch", "cancel", "--all", "--input", "jobs"),
        ("render", "--formats", "json"),
    ],
)
def test_a_parse_error_exits_2(argv: tuple[str, ...]) -> None:
    with pytest.raises(UsageError):
        build_parser().parse_args(list(argv))
    assert entry.main(list(argv)) == 2
