"""Extension seams of the ranges commands, which ChronoTranscriber lacks.

``ranges adjust`` checks its keys through ``ext.key_variables`` with a chat
client and the parsed arguments, registers the extension's arguments, hands
them to ``ext.startup``, and opens every boundary call as an attempt of role
``adjust``; the commands the ranges tasks of the ui print end with the words of
``ext.extra_argv``. Every test replaces the hooks; no call leaves the process.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

import chronominer.run as core
from chronominer import __main__ as entry
from chronominer import ext
from chronominer.cli.parser import build_parser
from chronominer.common.wizard import State
from chronominer.ext import Identity
from chronominer.settings import Settings
from chronominer.spec import (
    RANGES_ADJUST_OPTIONS,
    RANGES_GENERATE_OPTIONS,
    CommandSpec,
    parse_ranges_adjust,
)
from chronominer.ui import ranges as ranges_ui
from chronominer.ui.session import RunRequest, Session
from tests.characterization.fakes import PROVIDER_CLASSES, FakeLLM
from tests.characterization.inputs import make_text_file, write_ranges
from tests.characterization.seams import replace

FLAG = "--scratch-access"
NAME = "alpha"
ALTERNATIVE = "ALTERNATIVE_TEST_KEY"
SCHEMA = "PriceListEntries"


class Variables:
    """Replace ``ext.key_variables``; record each call and return *names*."""

    def __init__(
        self, monkeypatch: pytest.MonkeyPatch, names: tuple[str, ...] = ()
    ) -> None:
        self.calls: list[tuple[str, argparse.Namespace | None]] = []

        def key_variables(
            provider: str,
            endpoint: str | None = None,
            *,
            kind: str = "chat",
            settings: Settings,
            args: argparse.Namespace | None = None,
        ) -> tuple[str, ...]:
            self.calls.append((kind, args))
            return names

        monkeypatch.setattr(ext, "key_variables", key_variables)


class Startups:
    """Replace ``ext.startup``; record the extension value it receives."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.values: list[str | None] = []

        def startup(
            spec: CommandSpec, *, settings: Settings, args: argparse.Namespace
        ) -> None:
            self.values.append(getattr(args, "scratch_access", None))

        monkeypatch.setattr(ext, "startup", startup)


@pytest.fixture
def scratch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a plugin that registers ``--scratch-access`` and copies it."""

    def register_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(FLAG, dest="scratch_access", metavar="NAME")

    def extra_argv(args: argparse.Namespace | None) -> tuple[str, ...]:
        value = getattr(args, "scratch_access", None)
        return (FLAG, value) if value else ()

    monkeypatch.setattr(ext, "register_args", register_args)
    monkeypatch.setattr(ext, "extra_argv", extra_argv)


@pytest.fixture
def ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A ledger with line ranges, and no key variable set."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv(ALTERNATIVE, raising=False)
    path = make_text_file(tmp_path / "ranged" / "ledger.txt")
    write_ranges(path, [(1, 29), (31, 59)])
    return path


def _adjust_spec(path: Path) -> Any:
    return parse_ranges_adjust(["--input", str(path), "--schema", SCHEMA])


# Plan-time key checks


def test_the_adjust_plan_asks_for_chat_keys_with_the_args(
    monkeypatch: pytest.MonkeyPatch, ledger: Path
) -> None:
    variables = Variables(monkeypatch)
    args = argparse.Namespace(extra=True)

    ranges_plan = core.plan_ranges(_adjust_spec(ledger), Settings(), args=args)

    assert ranges_plan.files == (ledger,)
    assert variables.calls == [("chat", args)]


def test_the_adjust_plan_requires_the_alternative_variable(
    monkeypatch: pytest.MonkeyPatch, ledger: Path
) -> None:
    Variables(monkeypatch, (ALTERNATIVE,))
    monkeypatch.setenv("OPENAI_API_KEY", "settings-key")

    with pytest.raises(core.ConfigurationError, match=ALTERNATIVE):
        core.plan_ranges(_adjust_spec(ledger), Settings())

    monkeypatch.setenv(ALTERNATIVE, "k")
    assert core.plan_ranges(_adjust_spec(ledger), Settings()).files


# Arguments and startup


@pytest.mark.usefixtures("scratch")
def test_the_adjust_command_hands_the_extension_flag_to_startup(
    monkeypatch: pytest.MonkeyPatch, ledger: Path
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    startups = Startups(monkeypatch)
    argv = ["ranges", "adjust", "--input", str(ledger), "--schema", SCHEMA]

    assert entry.main([*argv, "--dry-run", "--json", FLAG, NAME]) == 0

    assert startups.values == [NAME]


# Copied commands of the ranges tasks


@pytest.mark.usefixtures("scratch")
def test_only_the_adjust_command_ends_with_the_extension_flag(ledger: Path) -> None:
    session = Session(
        Settings(), None, {}, build_parser().parse_args(["ui", FLAG, NAME])
    )
    adjust = State(RANGES_ADJUST_OPTIONS, {}, {"input": ledger, "schema": SCHEMA})
    generate = State(RANGES_GENERATE_OPTIONS, {}, {"input": ledger})

    words = ranges_ui.adjust_flow().command(adjust, session)
    plain = ranges_ui.generate_flow().command(generate, session)

    assert words[-2:] == [FLAG, NAME]
    assert build_parser().parse_args(words[1:]).scratch_access == NAME
    assert FLAG not in plain


# Attempt roles of the boundary calls


@pytest.mark.usefixtures("scratch")
def test_boundary_calls_open_attempts_of_role_adjust(
    monkeypatch: pytest.MonkeyPatch, ledger: Path
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    controller = FakeLLM()
    for module_name, name in PROVIDER_CLASSES:
        replace(monkeypatch, module_name, name, controller.constructor(name))
    identities: list[Identity] = []
    real = ext.attempt

    def attempt(*args: Any, **kwargs: Any) -> ext.Attempt[Any]:
        opened: ext.Attempt[Any] = real(*args, **kwargs)
        identities.append(opened.identity)
        return opened

    monkeypatch.setattr(ext, "attempt", attempt)
    startups = Startups(monkeypatch)
    args = build_parser().parse_args(["ui", FLAG, NAME])
    session = Session(Settings(), None, {}, args)
    answers = {"input": ledger, "schema": SCHEMA, "context": "none"}
    console = Console(file=io.StringIO(), width=200)

    code = ranges_ui.adjust_flow().run(RunRequest(session, answers, console, args))

    assert code == 0, console.file.getvalue()  # type: ignore[attr-defined]
    assert controller.calls
    assert identities
    assert {identity.role for identity in identities} == {"adjust"}
    assert {identity.kind for identity in identities} == {"chat"}
    assert startups.values == [NAME]
