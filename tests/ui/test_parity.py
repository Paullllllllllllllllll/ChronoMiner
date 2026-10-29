"""Parity between the CLI and the wizard for the nine tasks.

Every option of a task's table must be an argument of its subcommand and must
be set by some scripted wizard session, through a step that lists it, apart
from the exemptions below. Each session's printed command, parsed by the real
parser and resolved with the settings file it names, must resolve to the spec
the session's answers resolve to; it must also do so with an empty defaults
block. The service tier of a provider other than OpenAI is left out of the
command and of the comparison. Seeded random sessions keep that property. The
round trip ``parse(to_argv(spec)) == spec`` lives in
``tests/core/test_spec_roundtrip.py``.
"""

from __future__ import annotations

import argparse
import dataclasses
import random
import shutil
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal

import pytest

from chronominer import ext
from chronominer import run as core
from chronominer.cli.parser import build_parser
from chronominer.common.options import Option, OptionTable
from chronominer.common.testing import KEEP, ScriptedPrompter, ScriptError
from chronominer.common.wizard import (
    BACK,
    NO_DEFAULT,
    Choice,
    Decision,
    Nav,
    Outcome,
    State,
    Step,
    StepContext,
    Validator,
    command_display,
    default_index,
)
from chronominer.output.store import DATABASE_NAME
from chronominer.settings import Settings, load
from chronominer.spec import (
    CHUNKING_MODES,
    DEFAULT_SERVICE_TIER,
    INPUT_TYPES,
    OPTIONS,
    PROVIDER_TABLES,
    RunSpec,
)
from chronominer.ui import batch, extract, ranges, repair, store
from chronominer.ui.review import FORCE_ACTION
from chronominer.ui.session import (
    TASKS,
    Session,
    TaskFlow,
    absolute,
    build_wizard,
    new_state,
)
from tests.characterization.inputs import (
    make_context_image,
    make_image_folder,
    make_mixed_folder,
    make_scanned_pdf,
    make_text_file,
    write_ranges,
)
from tests.output.test_store import build_store, chunk
from tests.ui.conftest import DUMMY_KEYS, SettingsWriter

EXEMPT: Final[Mapping[str, str]] = MappingProxyType(
    {
        "json": "agent-contract output (one JSON line on stdout); no use in the UI",
        "settings": "given to the ui itself (ui --settings) and added to the command",
        "skipped": "repairs the decision pre-pass's skips; set by flag",
        "wait": "batch check in the ui makes one pass",
        "poll_minutes": "only with --wait, which the ui does not offer",
        "wait_hours": "only with --wait, which the ui does not offer",
    }
)
WIDTH: Final = 100
SCHEMA: Final = "WageRecords"
FLAT_SCHEMA: Final = (
    Path(__file__).resolve().parents[1] / "characterization" / "schemas"
) / "flat_entries.json"
_MISSING: Final = object()

FLOWS: Final[Mapping[str, Callable[[], TaskFlow]]] = MappingProxyType(
    {
        "extract": extract.flow,
        "ranges-generate": ranges.generate_flow,
        "ranges-adjust": ranges.adjust_flow,
        "batch-check": batch.check_flow,
        "batch-cancel": batch.cancel_flow,
        "repair": repair.flow,
        "render": store.render_flow,
        "drop": store.drop_flow,
    }
)


def unreached(names: Collection[str], reached: Collection[str]) -> set[str]:
    """Return the options of *names* that no wizard answer reached."""
    return {name for name in names if name not in EXEMPT and name not in reached}


class Workspace:
    """The inputs, files and settings of one test."""

    def __init__(
        self,
        tmp_path: Path,
        write_settings: SettingsWriter,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self.tmp = tmp_path
        self.write_settings = write_settings
        self.monkeypatch = monkeypatch
        self.text = make_text_file(tmp_path / "inputs" / "ledger.txt")
        self.ranged = make_text_file(tmp_path / "ranged" / "ledger.txt")
        write_ranges(self.ranged, [(1, 29), (31, 59)])
        self.stored = make_text_file(tmp_path / "stored" / "ledger.txt")
        write_ranges(self.stored, [(1, 29), (31, 59)])
        (self.stored.parent / DATABASE_NAME).write_bytes(b"")
        self.pdf = make_scanned_pdf(tmp_path / "scans" / "book.pdf", pages=2)
        self.pages = make_image_folder(tmp_path / "pages", pages=2)
        self.archive = tmp_path / "my archive"
        make_image_folder(self.archive / "volume_1", pages=1)
        make_image_folder(self.archive / "volume_2", pages=1)
        self.mixed = make_mixed_folder(tmp_path / "mixed")
        files = tmp_path / "files"
        files.mkdir()
        self.context = self._file(files / "context.txt", "Wage ledgers.\n")
        self.context_image = make_context_image(files / "context image.png")
        self.schema = files / "schema.json"
        shutil.copyfile(FLAT_SCHEMA, self.schema)
        self.out = tmp_path / "out dir"
        self.jobs = tmp_path / "jobs"
        self.jobs.mkdir()
        self.store = tmp_path / "store"
        for name in ("ledger", "prices"):
            build_store(self.store, [chunk(0)], name=name)
        monkeypatch.chdir(tmp_path)
        for name, value in DUMMY_KEYS.items():
            monkeypatch.setenv(name, value)

    @staticmethod
    def _file(path: Path, text: str) -> Path:
        path.write_text(text, encoding="utf-8")
        return path

    def images(self, folder: Path) -> list[str]:
        """Return the ``--files`` names of the images below *folder*."""
        return sorted(
            path.relative_to(folder).as_posix()
            for path in folder.rglob("*")
            if path.is_file()
        )

    def session(
        self, defaults: Mapping[str, Any] | None = None, *, named: bool = True
    ) -> Session:
        """Return a session over a settings file with *defaults*.

        Without *named* the session has the file's settings but no path, as
        the ui started without ``--settings`` has.
        """
        path = self.write_settings(defaults)
        settings = load(path, blocks=ext.settings_blocks())
        return Session(settings, path if named else None, {})


@dataclasses.dataclass(frozen=True)
class Script:
    """A wizard session: task, settings and the answers after the task step."""

    task: str
    session: Session
    answers: Sequence[Any]


@dataclasses.dataclass
class Played:
    """A finished session: outcome, transcript and what each step set."""

    flow: TaskFlow
    session: Session
    outcome: Outcome
    prompter: ScriptedPrompter
    changed: dict[str, set[str]]
    reached: dict[str, set[str]]

    @property
    def command(self) -> tuple[str, ...]:
        command = self.outcome.command
        assert command is not None, self.prompter.transcript
        return command

    def spec(self) -> Any:
        """Resolve the outcome's answers as the task does."""
        defaults = _defaults(self.session, self.flow)
        state = State(self.flow.table, defaults, self.outcome.answers)
        return self.flow.resolve(state, self.session).spec


def _defaults(session: Session, flow: TaskFlow) -> dict[str, Any]:
    return new_state(session, flow.table).defaults


def _recording(
    steps: Iterable[Step],
    table: OptionTable,
    changed: dict[str, set[str]],
    reached: dict[str, set[str]],
) -> list[Step]:
    """Wrap *steps* to record the answers each one changes and reaches."""

    def wrap(step: Step) -> Step:
        def ask(ctx: StepContext) -> Nav:
            before = ctx.state.answers
            result = step.ask(ctx)
            after = ctx.state.answers
            names = {
                name
                for name in before.keys() | after.keys()
                if before.get(name, _MISSING) != after.get(name, _MISSING)
            }
            changed.setdefault(step.name, set()).update(names)
            reached.setdefault(step.name, set()).update(
                name
                for name in names
                if name in after and after[name] != table[name].default
            )
            return result

        return dataclasses.replace(step, ask=ask)

    return [wrap(step) for step in steps]


def play(
    task: str, session: Session, prompter: ScriptedPrompter, *, check: bool = True
) -> Played:
    """Run the wizard of *task* with *prompter* and return what it produced."""
    flow = FLOWS[task]()
    changed: dict[str, set[str]] = {}
    reached: dict[str, set[str]] = {}
    recorded = dataclasses.replace(
        flow,
        steps=lambda s: _recording(flow.steps(s), flow.table, changed, reached),
    )
    wizard = build_wizard(recorded, session, prompter, width=WIDTH)
    try:
        outcome = wizard.run()
        if check:
            prompter.finish()
    except ScriptError as exc:
        raise ScriptError(f"{exc}\n{prompter.transcript}") from exc
    return Played(flow, session, outcome, prompter, changed, reached)


def play_script(script: Script) -> Played:
    """Play *script*, its task step answered with the preselected task."""
    prompter = ScriptedPrompter([script.task, *script.answers])
    return play(script.task, script.session, prompter)


MODEL_KEEPS: Final = [KEEP] * 4
"""Provider, model, reasoning effort and the recommended tuning."""
CONTEXT_KEEPS: Final = [KEEP] * 3
"""The extraction context (automatic), its fallback and the context image."""


def pdf_customized(ws: Workspace) -> Script:
    """A PDF with every model, response, image and run value set."""
    answers = [
        str(ws.pdf),
        "pdf",
        "range",
        "1-2",
        "folder",
        str(ws.out),
        ["csv"],
        "openai",
        "<other>",
        "gpt-4o",
        "customize",
        "4000",
        "0.3",
        "0.5",
        "priority",
        "json",
        SCHEMA,
        "file",
        str(ws.context),
        "file",
        str(ws.context_image),
        "customize",
        "200",
        "high",
        "jpeg",
        "80",
        False,
        "shadow",  # decision pre-pass
        "openai/gpt-6-luna",
        "openrouter",
        "https://example.org/decide",
        True,
        [],
        "copy",
    ]
    return Script("extract", ws.session(), answers)


def text_reasoning_and_schema_file(ws: Workspace) -> Script:
    """A text file in automatic chunks with reasoning, verbosity, a schema file
    and four parallel requests."""
    answers = [
        str(ws.text),
        "text",
        "auto",
        "500",
        KEEP,
        "beside",
        KEEP,
        "openai",
        "gpt-5.6-terra",
        "low",
        "customize",
        "low",
        "",
        KEEP,
        "schema",
        "<other>",
        str(ws.schema),
        "none",
        "auto",
        KEEP,
        KEEP,  # decision pre-pass
        ["force", "dry_run"],
        "4",
        "copy",
    ]
    return Script("extract", ws.session(), answers)


def text_line_ranges(ws: Workspace) -> Script:
    """A text file by its line ranges, a range of them, a fallback context."""
    answers = [
        str(ws.ranged),
        KEEP,
        "line-ranges",
        "range",
        "1",
        "beside",
        KEEP,
        *MODEL_KEEPS,
        KEEP,
        SCHEMA,
        KEEP,  # context: automatic
        "text",  # fallback context
        "wage ledgers",
        KEEP,  # context image
        KEEP,  # decision pre-pass
        KEEP,
        [],
        KEEP,
        "copy",
    ]
    return Script("extract", ws.session(), answers)


def text_adjusted_ranges_with_back(ws: Workspace) -> Script:
    """Adjusted line ranges with a context window, Back within the step."""
    answers = [
        str(ws.ranged),
        "text",
        "adjust",
        BACK,
        "adjust",
        "10",
        KEEP,
        "beside",
        KEEP,
        *MODEL_KEEPS,
        KEEP,
        SCHEMA,
        *CONTEXT_KEEPS,
        KEEP,  # decision pre-pass
        KEEP,
        [],
        KEEP,
        "copy",
    ]
    return Script("extract", ws.session(), answers)


def images_at_named_endpoint(ws: Workspace) -> Script:
    """The settings' named endpoint on chosen images of the subfolders."""
    answers = [
        str(ws.archive),
        True,
        "images",
        "list",
        ws.images(ws.archive)[-1:],
        "folder",
        str(ws.out),
        "mirror",
        KEEP,
        "endpoint:local",
        "qwen-vl",
        KEEP,
        KEEP,
        SCHEMA,
        *CONTEXT_KEEPS,
        "recommended",
        KEEP,  # decision pre-pass
        [],
        KEEP,
        "copy",
    ]
    return Script("extract", ws.session(), answers)


def mixed_folder_auto(ws: Workspace) -> Script:
    """Every type of a mixed folder in auto mode."""
    answers = [
        str(ws.mixed),
        False,
        "auto",
        "all",
        KEEP,
        KEEP,
        KEEP,
        "beside",
        KEEP,
        *MODEL_KEEPS,
        KEEP,
        SCHEMA,
        *CONTEXT_KEEPS,
        KEEP,
        KEEP,  # decision pre-pass
        KEEP,
        [],
        KEEP,
        "copy",
    ]
    return Script("extract", ws.session(), answers)


def provider_without_implied_model(ws: Workspace) -> Script:
    """OpenRouter with a typed model name that implies no provider, on the
    first image chosen by number."""
    answers = [
        str(ws.pages),
        False,
        KEEP,
        "numbers",
        "1",
        "beside",
        KEEP,
        "openrouter",
        "<other>",
        "qwen-vl",
        KEEP,
        KEEP,
        SCHEMA,
        *CONTEXT_KEEPS,
        KEEP,
        KEEP,  # decision pre-pass
        [],
        KEEP,
        "copy",
    ]
    return Script("extract", ws.session(), answers)


EXTRACT_DEFAULTS: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "formats": ["csv"],
        "reasoning_effort": "medium",
        "concurrency": 4,
        "service_tier": "priority",
        "layout": "mirror",
        "schema": SCHEMA,
    }
)


def extract_settings_defaults(ws: Workspace) -> Script:
    """Settings defaults kept; the output folder of the settings replaced."""
    defaults = {**EXTRACT_DEFAULTS, "output": str(ws.tmp / "default out")}
    answers = [
        str(ws.pdf),
        KEEP,
        KEEP,
        "beside",
        KEEP,
        *MODEL_KEEPS,
        KEEP,
        KEEP,  # schema of the settings
        *CONTEXT_KEEPS,
        KEEP,
        KEEP,  # decision pre-pass
        KEEP,
        KEEP,
        KEEP,
        "copy",
    ]
    return Script("extract", ws.session(defaults), answers)


def extract_defaults_unnamed(ws: Workspace) -> Script:
    """Settings defaults of a ui started without ``--settings``."""
    defaults = {**EXTRACT_DEFAULTS, "output": str(ws.tmp / "default out")}
    answers = [
        str(ws.text),
        KEEP,
        KEEP,
        KEEP,
        KEEP,
        "folder",
        KEEP,
        KEEP,
        *MODEL_KEEPS,
        KEEP,
        KEEP,  # schema of the settings
        *CONTEXT_KEEPS,
        KEEP,  # decision pre-pass
        KEEP,
        KEEP,
        KEEP,
        "copy",
    ]
    return Script("extract", ws.session(defaults, named=False), answers)


def _complete_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every planned item complete unless the run forces it."""
    real = core.plan

    def plan(spec: RunSpec, settings: Settings, **kwargs: Any) -> core.RunPlan:
        result = real(spec, settings, **kwargs)
        if spec.run.force:
            return result
        return dataclasses.replace(result, to_process=(), skipped=result.to_process)

    monkeypatch.setattr(core, "plan", plan)


def extract_force_action(ws: Workspace) -> Script:
    """The review's force action for complete items."""
    _complete_plan(ws.monkeypatch)
    answers = [
        str(ws.text),
        KEEP,
        KEEP,
        KEEP,
        KEEP,
        "beside",
        KEEP,
        *MODEL_KEEPS,
        KEEP,
        SCHEMA,
        *CONTEXT_KEEPS,
        KEEP,  # decision pre-pass
        KEEP,
        [],
        KEEP,
        FORCE_ACTION,
        "copy",
    ]
    return Script("extract", ws.session(), answers)


def generate_customized(ws: Workspace) -> Script:
    """Subfolders, tokens per chunk, a tokenizer model and a dry run, with Back."""
    answers = [
        str(ws.tmp / "ranged"),
        True,
        "500",
        BACK,
        "600",
        "gpt-4o",
        True,
        "copy",
    ]
    return Script("ranges-generate", ws.session(), answers)


def generate_defaults(ws: Workspace) -> Script:
    """One text file over a chunk tokens default."""
    answers = [str(ws.text), KEEP, KEEP, KEEP, "copy"]
    return Script("ranges-generate", ws.session({"chunk_tokens": 900}), answers)


def adjust_customized(ws: Workspace) -> Script:
    """Every adjust value: selection, schema file, window, context, model, run."""
    answers = [
        str(ws.tmp / "stored"),
        True,
        "first:1",
        "<other>",
        str(ws.schema),
        "10",
        "file",
        str(ws.context),
        "openai",
        "<other>",
        "gpt-4o",
        "customize",
        "4000",
        "0.3",
        "0.5",
        "priority",
        "json",
        "4",
        True,
        True,
        "copy",
    ]
    return Script("ranges-adjust", ws.session(), answers)


def adjust_reasoning_with_back(ws: Workspace) -> Script:
    """Reasoning and verbosity, no context, with Back from the model."""
    answers = [
        str(ws.ranged),
        "",
        SCHEMA,
        KEEP,
        "none",
        "openai",
        BACK,
        BACK,
        KEEP,
        "none",
        "openai",
        "gpt-5.6-terra",
        "low",
        "customize",
        "low",
        "",
        KEEP,
        "schema",
        KEEP,
        KEEP,
        "copy",
    ]
    return Script("ranges-adjust", ws.session(), answers)


def adjust_endpoint(ws: Workspace) -> Script:
    """The named endpoint, over settings defaults, with a fallback context."""
    answers = [
        str(ws.ranged),
        KEEP,
        SCHEMA,
        KEEP,
        KEEP,
        "text",  # fallback context
        "entries begin with a date",
        "endpoint:local",
        "llava",
        KEEP,
        KEEP,
        KEEP,
        KEEP,
        "copy",
    ]
    defaults = {"reasoning_effort": "low", "concurrency": 4}
    return Script("ranges-adjust", ws.session(defaults), answers)


def repair_customized(ws: Workspace) -> Script:
    """A schema, indices, every model and image value, a batch."""
    answers = [
        str(ws.text),
        SCHEMA,
        "0,2-3",
        "openai",
        "<other>",
        "gpt-4o",
        "customize",
        "4000",
        "0.3",
        "0.5",
        "priority",
        "json",
        "customize",
        "200",
        "high",
        "jpeg",
        "80",
        False,
        True,
        True,
        "copy",
    ]
    return Script("repair", ws.session(), answers)


def repair_endpoint_with_back(ws: Workspace) -> Script:
    """A typed schema at the named endpoint, with Back."""
    answers = [
        str(ws.text),
        "<other>",
        BACK,
        "<other>",
        str(ws.schema),
        "",
        "endpoint:local",
        "qwen-vl",
        KEEP,
        KEEP,
        "recommended",
        KEEP,
        "copy",
    ]
    return Script("repair", ws.session(), answers)


def repair_reasoning_defaults(ws: Workspace) -> Script:
    """Reasoning and verbosity over settings defaults."""
    defaults = {"reasoning_effort": "low", "service_tier": "priority"}
    answers = [
        str(ws.text),
        KEEP,
        KEEP,
        "openai",
        "gpt-5.6-terra",
        "medium",
        "customize",
        "high",
        "",
        KEEP,
        "schema",
        KEEP,
        KEEP,
        KEEP,
        "copy",
    ]
    return Script("repair", ws.session(defaults), answers)


def repair_provider(ws: Workspace) -> Script:
    """OpenRouter with a typed model name, edited from the review."""
    answers = [
        str(ws.text),
        KEEP,
        KEEP,
        *MODEL_KEEPS,
        KEEP,
        KEEP,
        KEEP,
        KEEP,
        "edit",
        "model",
        "openrouter",
        "<other>",
        "qwen-vl",
        KEEP,
        "copy",
    ]
    return Script("repair", ws.session(named=False), answers)


def check_adopt(ws: Workspace) -> Script:
    """A schema, named batches, an adopted batch and a format, with Back."""
    answers = [
        str(ws.jobs),
        SCHEMA,
        "b1, b2",
        BACK,
        "b1,b2",
        "adopt",
        "batch_7",
        "choose",
        ["csv"],
        "copy",
    ]
    return Script("batch-check", ws.session(), answers)


def check_abandon_defaults(ws: Workspace) -> Script:
    """An abandoned submission, edited from the review; the check keeps the run
    formats over a formats default of the settings."""
    answers = [
        str(ws.jobs),
        "<other>",
        "Custom Records",
        "",
        "adopt",
        "batch_7",
        KEEP,
        "edit",
        "unresolved",
        "abandon",
        "sub-1",
        "copy",
    ]
    return Script("batch-check", ws.session({"formats": ["csv"]}), answers)


def cancel_ids_with_back(ws: Workspace) -> Script:
    """Named batch IDs after Back from the folder."""
    answers = ["dir", BACK, "ids", "batch_1, -batch_2", "copy"]
    return Script("batch-cancel", ws.session(), answers)


def cancel_all(ws: Workspace) -> Script:
    """Every unfinished batch."""
    return Script("batch-cancel", ws.session(named=False), ["all", "copy"])


def cancel_folder(ws: Workspace) -> Script:
    """The open submissions below a folder, over settings defaults."""
    answers = ["dir", str(ws.jobs), "copy"]
    return Script("batch-cancel", ws.session(EXTRACT_DEFAULTS), answers)


def render_chosen(ws: Workspace) -> Script:
    """Chosen formats, force and a dry run, with Back."""
    answers = [
        str(ws.out),
        "choose",
        BACK,
        "choose",
        ["csv"],
        True,
        True,
        "copy",
    ]
    return Script("render", ws.session(), answers)


def render_kept(ws: Workspace) -> Script:
    """The formats rendered before, over settings defaults."""
    answers = [str(ws.out), KEEP, KEEP, KEEP, "copy"]
    return Script("render", ws.session(EXTRACT_DEFAULTS), answers)


def drop_numbers_dry_run(ws: Workspace) -> Script:
    """Documents by number, sent as their names, as a dry run."""
    answers = [str(ws.store), "numbers", "1", True, "copy"]
    return Script("drop", ws.session(), answers)


def drop_search_with_back(ws: Workspace) -> Script:
    """A name search after Back from the list, over settings defaults."""
    answers = [str(ws.store), "list", BACK, "search", "pri", KEEP, "copy"]
    return Script("drop", ws.session(EXTRACT_DEFAULTS), answers)


def drop_all(ws: Workspace) -> Script:
    """Every stored document."""
    return Script("drop", ws.session(), [str(ws.store), "all", KEEP, "copy"])


SESSIONS: Final[Mapping[str, Callable[[Workspace], Script]]] = MappingProxyType(
    {
        builder.__name__: builder
        for builder in (
            pdf_customized,
            text_reasoning_and_schema_file,
            text_line_ranges,
            text_adjusted_ranges_with_back,
            images_at_named_endpoint,
            mixed_folder_auto,
            provider_without_implied_model,
            extract_settings_defaults,
            extract_defaults_unnamed,
            extract_force_action,
            generate_customized,
            generate_defaults,
            adjust_customized,
            adjust_reasoning_with_back,
            adjust_endpoint,
            repair_customized,
            repair_endpoint_with_back,
            repair_reasoning_defaults,
            repair_provider,
            check_adopt,
            check_abandon_defaults,
            cancel_ids_with_back,
            cancel_all,
            cancel_folder,
            render_chosen,
            render_kept,
            drop_numbers_dry_run,
            drop_search_with_back,
            drop_all,
        )
    }
)


@pytest.fixture
def workspace(
    tmp_path: Path, write_settings: SettingsWriter, monkeypatch: pytest.MonkeyPatch
) -> Workspace:
    return Workspace(tmp_path, write_settings, monkeypatch)


def _subparser(words: Sequence[str]) -> argparse.ArgumentParser:
    """Return the parser of the subcommand *words*, such as batch check."""
    parser: argparse.ArgumentParser = build_parser()
    for word in words:
        (commands,) = [
            action
            for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        ]
        parser = commands.choices[word]
    return parser


def _arguments(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    return {
        action.dest: action
        for action in parser._actions
        if not isinstance(action, argparse._HelpAction | argparse._SubParsersAction)
    }


def _extension_dests() -> set[str]:
    parser = argparse.ArgumentParser(add_help=False)
    ext.register_args(parser)
    return {action.dest for action in parser._actions}


@pytest.mark.parametrize("task", list(FLOWS))
def test_every_option_is_an_argument_of_the_subcommand(task: str) -> None:
    flow = FLOWS[task]()
    arguments = _arguments(_subparser(flow.program[1:]))
    calls = any(flow.table is table for table in PROVIDER_TABLES)
    extension = _extension_dests() if calls else set()
    assert set(arguments) == {*flow.table.names, *extension}
    for option in flow.table:
        strings = set(arguments[option.name].option_strings)
        expected = {option.flag}
        if option.kind.value == "switch":
            expected.add(option.negative_flag)
        assert strings == expected, option.name


def test_the_ui_parser_accepts_the_settings_option(tmp_path: Path) -> None:
    arguments = _arguments(_subparser(("ui",)))
    assert set(arguments) == {"settings", *_extension_dests()}
    path = tmp_path / "other.yaml"
    args = build_parser().parse_args(["ui", "--settings", str(path)])
    assert args.settings == path


def test_the_tasks_are_the_flows() -> None:
    assert list(FLOWS) == list(TASKS)
    for task, make in FLOWS.items():
        flow = make()
        assert flow.name == task
        assert flow.program[0] == "chronominer"


def test_the_exemptions_are_options_no_step_sets(workspace: Workspace) -> None:
    session = workspace.session()
    assert all(reason.strip() for reason in EXEMPT.values())
    for make in FLOWS.values():
        flow = make()
        assert "json" in flow.table
        declared = {name for step in flow.steps(session) for name in step.sets}
        assert not declared & set(EXEMPT), flow.name


def test_an_option_missing_from_every_step_is_reported() -> None:
    names = set(OPTIONS.names)
    extra = Option("extra", "--extra", "an option no step sets")
    table = OptionTable([*OPTIONS, extra])
    assert unreached(names, names) == set()
    assert unreached(table.names, names) == {"extra"}
    assert unreached(names, names - {"top_p", "json"}) == {"top_p"}


@pytest.mark.parametrize("task", list(FLOWS))
def test_the_sessions_reach_every_option_from_their_steps(
    workspace: Workspace, task: str
) -> None:
    changed: dict[str, set[str]] = {}
    reached: dict[str, set[str]] = {}
    for build in SESSIONS.values():
        script = build(workspace)
        if script.task != task:
            continue
        played = play_script(script)
        for name, names in played.changed.items():
            changed.setdefault(name, set()).update(names)
        for name, names in played.reached.items():
            reached.setdefault(name, set()).update(names)
    flow = FLOWS[task]()
    every = {name for names in reached.values() for name in names}
    assert unreached(flow.table.names, every) == set()
    for step in flow.steps(workspace.session()):
        assert changed.get(step.name, set()) <= set(step.sets), step.name
        assert set(step.sets) <= reached.get(step.name, set()), step.name


def test_the_extract_sessions_cover_every_type_and_chunking_mode(
    workspace: Workspace,
) -> None:
    types: set[str] = set()
    modes: set[str] = set()
    for build in SESSIONS.values():
        script = build(workspace)
        if script.task == "extract":
            spec = play_script(script).spec()
            types.add(spec.input.type)
            modes.add(spec.chunking.mode)
    assert types == set(INPUT_TYPES)
    assert modes == set(CHUNKING_MODES)


def command_spec(played: Played, *, empty: bool = False) -> Any:
    """Parse the printed command with the real parser and resolve it.

    The settings are those of the file the command names, else the
    session's; *empty* drops their defaults block.
    """
    flow, words = played.flow, played.command
    assert tuple(words[: len(flow.program)]) == flow.program
    args = build_parser().parse_args(list(words[1:]))
    flags = flow.table.values(args)
    path = flags.pop("settings", None)
    settings = played.session.settings
    if path is not None:
        settings = load(path, blocks=ext.settings_blocks())
    if empty:
        settings = dataclasses.replace(settings, defaults=MappingProxyType({}))
    session = Session(settings)
    state = State(flow.table, _defaults(session, flow), flags)
    return flow.resolve(state, session).spec


def _tier_neutral(spec: Any) -> Any:
    """Return *spec* without the service tier of a provider other than OpenAI.

    The printed command leaves that tier out, since only OpenAI calls send it.
    """
    model = getattr(spec, "model", None)
    if not dataclasses.is_dataclass(model) or model.provider == "openai":
        return spec
    return dataclasses.replace(
        spec, model=dataclasses.replace(model, service_tier=DEFAULT_SERVICE_TIER)
    )


def check_command(played: Played) -> None:
    """The printed command yields the session's spec, with and without defaults."""
    spec = _tier_neutral(played.spec())
    words = played.command
    assert "--json" not in words
    path = played.session.settings_path
    if path is not None and "settings" in played.flow.table:
        assert words[-2:] == ("--settings", str(absolute(path)))
    else:
        assert "--settings" not in words
    assert _tier_neutral(command_spec(played)) == spec, played.prompter.transcript
    assert _tier_neutral(command_spec(played, empty=True)) == spec, (
        played.prompter.transcript
    )
    shown = command_display(list(words), width=WIDTH)
    lines = played.prompter.lines
    start = len(lines) - lines[::-1].index("Equivalent command:")
    assert lines[start : start + len(shown)] == shown


@pytest.mark.parametrize("name", list(SESSIONS))
def test_the_printed_command_yields_the_wizard_spec(
    workspace: Workspace, name: str
) -> None:
    played = play_script(SESSIONS[name](workspace))
    assert played.outcome.action is Decision.COPY
    check_command(played)


def _flag_value(words: Sequence[str], flag: str) -> str:
    return words[list(words).index(flag) + 1]


def test_settings_defaults_are_kept_and_spelled_out(workspace: Workspace) -> None:
    played = play_script(extract_settings_defaults(workspace))
    spec = played.spec()
    assert spec.output.directory is None
    assert spec.output.formats == ("csv",)
    assert spec.model.reasoning_effort == "medium"
    words = played.command
    assert "--beside-input" in words and "--output" not in words
    assert _flag_value(words, "--formats") == "csv"
    assert _flag_value(words, "--reasoning-effort") == "medium"
    assert _flag_value(words, "--concurrency") == "4"
    assert _flag_value(words, "--service-tier") == "priority"
    assert _flag_value(words, "--schema") == SCHEMA
    assert "(s)" in played.prompter.transcript


def test_the_force_action_and_the_dry_run_reach_the_command(
    workspace: Workspace,
) -> None:
    forced = play_script(extract_force_action(workspace))
    assert forced.outcome.answers["force"] is True
    assert "--force" in forced.command
    dry = play_script(text_reasoning_and_schema_file(workspace))
    run = dry.spec().run
    assert (run.dry_run, run.force) == (True, True)
    assert {"--dry-run", "--force"} <= set(dry.command)


RANDOM_SESSIONS: Final = 300
PATIENCE: Final = 120
"""Questions a random session answers before it stops going back."""
_BACK_RATE: Final = 0.08


def _texts(ws: Workspace) -> list[tuple[str, list[str]]]:
    """Candidate texts by question prefix; the first matching prefix wins."""
    inputs = [ws.text, ws.ranged, ws.pdf, ws.pages, ws.archive, ws.mixed]
    ranged = [ws.tmp / "ranged", ws.tmp / "stored", ws.text, ws.mixed, ws.tmp]
    return [
        ("Input (file or folder):", [str(path) for path in inputs]),
        ("Output folder:", [str(ws.out), str(ws.tmp / "other")]),
        ("Tokens per chunk:", ["500", "3500"]),
        ("Lines around each boundary:", ["6", "10"]),
        ("Model name:", ["gpt-4o", "claude-sonnet-5-5", "gemini-3.1-pro", "qwen-vl"]),
        ("Model at endpoint", ["qwen-vl", "llava"]),
        ("Max output tokens", ["", "4000"]),
        ("Temperature", ["0.3", "1.5"]),
        ("Top p", ["0.5", "1"]),
        ("Schema file:", [str(ws.schema)]),
        ("Context file:", [str(ws.context)]),
        ("Context image file:", [str(ws.context_image)]),
        ("Topics:", ["wage ledgers", "prices; wages"]),
        ("Items (numbers or ranges):", ["1", "1-2", "1,2"]),
        ("Name search:", ["ledger", "page", "book", "."]),
        ("DPI", ["", "native", "200"]),
        ("JPEG quality", ["", "70"]),
        ("Range:", ["1", "first:1", "2-", "1,2"]),
        ("Parallel requests:", ["1", "16", "4"]),
        ("Text file or folder", [str(path) for path in ranged]),
        ("Line ranges to adjust", ["", "1", "first:1"]),
        ("Tokenizer model", ["gpt-4o", "gpt-5.6-luna"]),
        ("Decision model id:", ["openai/gpt-6-luna", "gpt-5.6-luna"]),
        ("Decision endpoint URL", ["", "https://example.org/decide"]),
        ("Context window", ["6", "12"]),
        ("Folder, chronominer.sqlite or file", [str(ws.text), str(ws.jobs)]),
        ("Schema name or file:", [SCHEMA, "Custom Records", str(ws.schema)]),
        ("Chunk indices", ["", "0", "0,2-3"]),
        ("Folder with", [str(ws.jobs), str(ws.tmp)]),
        ("Batch IDs to finalize", ["", "b1", "b1, -b2"]),
        ("Batch IDs (comma-separated):", ["batch_1", "b1, -b2"]),
        ("Provider batch ID:", ["batch_7"]),
        ("Submission ID:", ["sub-1", "-sub"]),
        ("Input (folder or", [str(ws.out), str(ws.jobs), str(ws.tmp / "missing")]),
        ("Store (output folder", [str(ws.store), str(ws.tmp / "missing")]),
        ("Documents (numbers or ranges):", ["1", "1-2", "2"]),
    ]


class RandomPrompter(ScriptedPrompter):
    """Answer from the offered choices and from valid candidate texts.

    The task step keeps its task and the quit question stays. After
    :data:`PATIENCE` questions no answer is Back and the review copies the
    command; a review that offers no command quits after four times as
    many. Each answer is recorded as a scripted answer, so the transcript
    and the checks of :class:`ScriptedPrompter` apply.
    """

    def __init__(self, rng: random.Random, texts: list[tuple[str, list[str]]]):
        super().__init__([])
        self.rng = rng
        self.texts = texts
        self.asked = 0

    def _back(self) -> bool:
        return self.asked < PATIENCE and self.rng.random() < _BACK_RATE

    def _answer(self, answer: Any) -> None:
        self.asked += 1
        self._answers.append(answer)

    def select[T](
        self,
        message: str,
        choices: Sequence[Choice[T]],
        *,
        default: Any = NO_DEFAULT,
        marker: str | None = None,
    ) -> T | Literal[Nav.BACK]:
        values = [choice.value for choice in choices]
        answer: Any
        if message == "Task:":
            answer = default
        elif message == "Quit without running?":
            answer = "stay"
        elif message == "Start:":
            answer = self._start(values)
        elif self._back():
            answer = BACK
        elif self.rng.random() < 0.3 and default_index(choices, default) is not None:
            answer = KEEP
        else:
            answer = self.rng.choice(values)
        self._answer(answer)
        return super().select(message, choices, default=default, marker=marker)

    def _start(self, values: list[Any]) -> Any:
        if self.asked >= PATIENCE and Decision.COPY.value in values:
            return Decision.COPY.value
        roll = self.rng.random()
        if roll < 0.15 and self.asked < PATIENCE:
            return BACK
        if roll < 0.35 and self.asked < PATIENCE:
            return "edit"
        if FORCE_ACTION in values and roll < 0.5:
            return FORCE_ACTION
        if Decision.COPY.value in values:
            return Decision.COPY.value
        return "edit" if self.asked < 4 * PATIENCE else Decision.QUIT.value

    def checkbox[T](
        self,
        message: str,
        choices: Sequence[Choice[T]],
        *,
        checked: Collection[T] = (),
        marker: str | None = None,
    ) -> list[T] | Literal[Nav.BACK]:
        answer: Any
        if self._back():
            answer = BACK
        else:
            answer = [choice.value for choice in choices if self.rng.random() < 0.5]
            if not answer and choices and self.asked >= PATIENCE:
                answer = [choices[0].value]
        self._answer(answer)
        return super().checkbox(message, choices, checked=checked, marker=marker)

    def _typed(
        self,
        message: str,
        default: str,
        validate: Validator | None,
        marker: str | None,
    ) -> str | Literal[Nav.BACK]:
        if self._back():
            self._answer(BACK)
        else:
            self._answer(self._text(message, default, validate))
        return super()._typed(message, default, validate, marker)

    def _text(self, message: str, default: str, validate: Validator | None) -> str:
        pool = next(
            (texts for prefix, texts in self.texts if message.startswith(prefix)), []
        )
        candidates = [*pool, default] if default else list(pool)
        valid = [
            text for text in candidates if validate is None or validate(text) is None
        ]
        if not valid:
            raise ScriptError(f"no valid candidate text for {message!r}")
        return self.rng.choice(valid)


_DEFAULT_POOL: Final[Mapping[str, Sequence[Any]]] = MappingProxyType(
    {
        "type": ["auto", "text", "images", "pdf"],
        "fallback_context": ["wage ledgers", "none"],
        "recursive": [True, False],
        "layout": ["mirror", "flat"],
        "formats": [["csv"], []],
        "schema": [SCHEMA, "TradeRecords"],
        "context": ["none", "auto"],
        "context_image": ["none", "auto"],
        "chunking": ["auto", "line-ranges", "adjust"],
        "chunk_tokens": [900],
        "context_window": [12],
        "provider": ["anthropic", "google", "openai"],
        "model": ["gpt-5.6-terra", "claude-sonnet-5-5", "gemini-3.1-pro", "gpt-4o"],
        "endpoint": ["local"],
        "reasoning_effort": ["low", "medium"],
        "verbosity": ["low"],
        "max_output_tokens": [3000],
        "temperature": [0.4],
        "top_p": [0.7],
        "service_tier": ["priority", "auto"],
        "response_format": ["json", "schema"],
        "dpi": [200, "native"],
        "image_detail": ["low", "auto"],
        "image_format": ["png", "jpeg"],
        "jpeg_quality": [70],
        "grayscale": [True, False],
        "batch": [True, False],
        "concurrency": [4],
    }
)


def _random_defaults(rng: random.Random, ws: Workspace) -> dict[str, Any] | None:
    if rng.random() < 0.3:
        return None
    pool = {
        **_DEFAULT_POOL,
        "output": [str(ws.tmp / "default out")],
        "context": [*_DEFAULT_POOL["context"], str(ws.context)],
    }
    keys = rng.sample(sorted(pool), rng.randint(1, 10))
    return {key: rng.choice(pool[key]) for key in keys}


@pytest.mark.parametrize("task", list(FLOWS))
def test_random_sessions_print_the_command_of_their_spec(
    workspace: Workspace, task: str
) -> None:
    texts = _texts(workspace)
    for seed in range(RANDOM_SESSIONS // len(FLOWS)):
        rng = random.Random(f"{task}-{seed}")
        defaults = _random_defaults(rng, workspace)
        session = workspace.session(defaults, named=rng.random() < 0.5)
        prompter = RandomPrompter(rng, texts)
        played = play(task, session, prompter, check=False)
        assert played.outcome.action is Decision.COPY, (seed, prompter.transcript)
        check_command(played)
