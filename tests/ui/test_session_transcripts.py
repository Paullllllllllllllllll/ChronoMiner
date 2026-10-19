"""Recorded guided sessions, run to the end: text by line ranges, images in batch.

Each scripted session goes through the real wizard and app, goes Back at
least once and runs from the review screen. The settings file holds a
defaults block, so the review marks values taken from it. The providers and
their batch APIs are the characterization fakes; the image preprocessing
runs for real. The golden ``golden/session_<name>.txt`` holds the prompter
transcript followed by the run screen and the summary, with temp paths,
store and batch ids and durations replaced by placeholders.
``CM_UPDATE_GOLDEN=1`` rewrites it.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import difflib
import io
import os
import re
import shlex
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml
from rich.console import Console

from chronominer import ext, spec
from chronominer.cli.parser import build_parser
from chronominer.common.store import HEARTBEAT_INTERVAL
from chronominer.common.testing import KEEP, ScriptedPrompter, WriteGuard
from chronominer.common.wizard import BACK, command_display
from chronominer.llm import prompts
from chronominer.settings import Settings, load, resolve
from chronominer.spec import OPTIONS, RunSpec
from chronominer.ui import app
from tests.characterization.fakes import DUMMY_KEYS, PROVIDER_CLASSES, FakeLLM
from tests.characterization.fakes_batch import FakeBatchService
from tests.characterization.golden import Normalizer, store_ids
from tests.characterization.inputs import (
    make_image_folder,
    make_text_file,
    write_ranges,
)
from tests.characterization.seams import replace

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
UPDATE_ENV = "CM_UPDATE_GOLDEN"
WIDTH = 100
PUBLIC_SCHEMAS = (
    "price_list_entries.json",
    "trade_records.json",
    "wage_records.json",
)

_COMMAND_HEADER = "Equivalent command:"
_DURATION_RE = re.compile(r"(Run time: |Waiting )\d+(?:\.\d+)?s\b")
_ASYNC_SLEEP = asyncio.sleep


async def _short_sleeps_yield(delay: float, result: Any = None) -> Any:
    """Stand-in for ``asyncio.sleep`` that only yields for waits and retries.

    The lease heartbeat keeps its real interval, so it does not renew a
    lease in the middle of the run's release of it.
    """
    await _ASYNC_SLEEP(delay if delay >= HEARTBEAT_INTERVAL else 0)
    return result


@dataclasses.dataclass(frozen=True)
class Case:
    """One recorded session: input, settings defaults, answers, batch or sync."""

    name: str
    defaults: Mapping[str, Any]
    make_input: Callable[[Path], Path]
    answers: Callable[[Path, Path], list[Any]]
    batch: bool = False


def _ranged_text(root: Path) -> Path:
    path = make_text_file(root / "inputs" / "ledger.txt")
    write_ranges(path, [(1, 29), (30, 59), (60, 71)])
    return path


def _text_answers(source: Path, out: Path) -> list[Any]:
    return [
        KEEP,  # task
        str(source),
        "text",
        "line-ranges",
        KEEP,  # line ranges: all
        "folder",
        str(out),
        ["csv"],
        BACK,  # from the provider back to the output step
        KEEP,  # output choice
        KEEP,  # output folder
        KEEP,  # formats
        *[KEEP] * 4,  # provider, model, reasoning, model parameters
        KEEP,  # response format
        "PriceListEntries",
        *[KEEP] * 3,  # context, fallback context, context image
        KEEP,  # submission
        [],  # run options
        KEEP,  # concurrency
        "run",
    ]


def _image_answers(source: Path, out: Path) -> list[Any]:
    return [
        KEEP,  # task
        str(source),
        KEEP,  # subfolders
        "images",
        "all",
        "folder",
        str(out),
        KEEP,  # layout
        KEEP,  # formats
        BACK,  # from the provider back to the output step
        KEEP,  # output choice
        KEEP,  # output folder
        KEEP,  # layout
        ["csv"],
        *[KEEP] * 4,  # provider, model, reasoning, model parameters
        KEEP,  # response format
        "WageRecords",
        *[KEEP] * 3,  # context, fallback context, context image
        KEEP,  # page images
        True,  # submission: batch API
        [],  # run options; no parallel requests for a batch run
        "run",
    ]


CASES = (
    Case(
        "text-line-ranges",
        {"model": "gpt-5.6-terra", "service_tier": "priority", "concurrency": 4},
        _ranged_text,
        _text_answers,
    ),
    Case(
        "image-batch",
        {"reasoning_effort": "medium", "image_detail": "high", "layout": "mirror"},
        lambda root: make_image_folder(root / "inputs" / "scans", pages=3),
        _image_answers,
        batch=True,
    ),
)


def _public_schemas(folder: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Show the wizard only some public schemas, whatever ``schemas/`` holds."""
    folder.mkdir(parents=True)
    for name in PUBLIC_SCHEMAS:
        shutil.copyfile(spec.SCHEMAS_DIR / name, folder / name)
    monkeypatch.setattr(spec, "SCHEMAS_DIR", folder)


def _settings_file(folder: Path, defaults: Mapping[str, Any]) -> Path:
    """Write a settings file with a defaults block."""
    path = folder / "settings.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"defaults": dict(defaults)}
    path.write_text(yaml.safe_dump(data), encoding="utf-8", newline="\n")
    return path


def _command_block(lines: Sequence[str]) -> tuple[int, int]:
    """Return the slice of the last printed equivalent command."""
    start = len(lines) - 1 - list(lines)[::-1].index(_COMMAND_HEADER) + 1
    end = start
    while end < len(lines) and lines[end].startswith("  "):
        end += 1
    return start, end


_POWERSHELL_WORD = re.compile(r"'((?:[^']|'')*)'|(\S+)")


def _split(line: str) -> list[str]:
    """Split a command line printed for the current platform into words.

    On Windows the words are quoted for PowerShell: bare, or in single quotes
    with each inner single quote doubled.
    """
    if os.name != "nt":
        return shlex.split(line)
    return [
        bare if quoted is None else quoted.replace("''", "'")
        for quoted, bare in (
            (match.group(1), match.group(2))
            for match in _POWERSHELL_WORD.finditer(line)
        )
    ]


def _spec_of(words: Sequence[str]) -> RunSpec:
    """Parse and resolve a printed command as ``chronominer`` would."""
    assert list(words[:2]) == ["chronominer", "extract"]
    flags = OPTIONS.values(build_parser().parse_args(list(words[1:])))
    settings = load(flags.get("settings"), blocks=ext.settings_blocks())
    return resolve(settings, flags).spec


def _normalize(
    lines: list[str], words: Sequence[str], console: str, norm: Normalizer
) -> str:
    """Replace temp paths, ids and durations; re-wrap the command.

    The command is wrapped again from its normalized words, quoted for
    Windows, so its line breaks do not depend on the temp path or the OS.
    """
    start, end = _command_block(lines)
    shown = command_display(
        [norm.text(word) for word in words], width=WIDTH, windows=True
    )
    lines = [*lines[:start], *shown, *lines[end:]]
    text = norm.text("\n".join(lines) + "\n" + console)
    text = _DURATION_RE.sub(r"\1<duration>", text)
    return "".join(line.rstrip() + "\n" for line in text.splitlines())


def _check_golden(name: str, actual: str, write_guard: WriteGuard) -> None:
    golden = GOLDEN_DIR / f"session_{name}.txt"
    if os.environ.get(UPDATE_ENV) == "1":
        golden.parent.mkdir(parents=True, exist_ok=True)
        golden.write_text(actual, encoding="utf-8", newline="\n")
        write_guard.consume(golden.parent)
        return
    if not golden.is_file():
        pytest.fail(f"No golden {golden.name}; run with {UPDATE_ENV}=1 to create.")
    expected = golden.read_bytes().decode("utf-8")
    if actual != expected:
        diff = "".join(
            difflib.unified_diff(
                expected.splitlines(keepends=True),
                actual.splitlines(keepends=True),
                fromfile=f"golden/{golden.name}",
                tofile="session",
            )
        )
        pytest.fail(
            f"Session transcript changed (set {UPDATE_ENV}=1 to accept):\n{diff}"
        )


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_recorded_session(
    case: Case,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    write_guard: WriteGuard,
) -> None:
    for name, value in DUMMY_KEYS.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(asyncio, "sleep", _short_sleeps_yield)
    llm = FakeLLM()
    for module_name, name in PROVIDER_CLASSES:
        replace(monkeypatch, module_name, name, llm.constructor(name))
    service = FakeBatchService(llm)
    service.install(monkeypatch)
    context = tmp_path / "context"
    context.mkdir()
    monkeypatch.setattr(prompts, "CONTEXT_DIR", context)
    _public_schemas(tmp_path / "schemas", monkeypatch)
    source = case.make_input(tmp_path)
    output = tmp_path / "out"
    settings = _settings_file(tmp_path / "config", case.defaults)
    started: list[RunSpec] = []
    real_startup = ext.startup

    def startup(spec: RunSpec, *, settings: Settings, args: argparse.Namespace) -> None:
        started.append(spec)
        real_startup(spec, settings=settings, args=args)

    monkeypatch.setattr(ext, "startup", startup)
    monkeypatch.chdir(tmp_path)

    console = Console(
        file=io.StringIO(),
        width=WIDTH,
        color_system=None,
        force_terminal=False,
        legacy_windows=False,
        soft_wrap=True,
    )
    prompter = ScriptedPrompter(case.answers(source, output))
    try:
        with prompter:
            code = app.run_app(
                settings, prompter=prompter, console=console, require_tty=False
            )
        batches = len(service.batches)
        ids = {**store_ids([output]), **service.id_replacements()}
    finally:
        service.reset()
    screen = console.file
    assert isinstance(screen, io.StringIO)
    assert code == 0, prompter.transcript + screen.getvalue()
    assert (batches > 0) is case.batch
    assert ("batch API; collect the results" in screen.getvalue()) is case.batch

    lines = prompter.transcript.splitlines()
    assert "  answer: Back" in lines
    assert any(line.endswith(" (s)") or " (s) " in line for line in lines)
    start, end = _command_block(lines)
    words = _split(" ".join(line.strip() for line in lines[start:end]))
    assert lines[start:end] == command_display(words, width=WIDTH)
    assert len(started) == 1
    assert _spec_of(words) == started[0]

    norm = Normalizer({tmp_path: "<tmp>"}, ids)
    actual = _normalize(lines, words, screen.getvalue(), norm)
    _check_golden(case.name, actual, write_guard)
