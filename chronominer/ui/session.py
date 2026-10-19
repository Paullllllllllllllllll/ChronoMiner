"""What a guided session shares: settings, tasks, the task-flow contract.

Every wizard task (a new extraction, ranges generate and adjust, batch check
and cancel, repair, render, drop) is a :class:`TaskFlow`.
:func:`build_wizard` puts the task step first, so the user can switch tasks,
then the task's own steps. :class:`Inventory` holds what an input path holds.
"""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

import fitz

from chronominer import ext
from chronominer.common.options import OptionTable
from chronominer.common.wizard import (
    BACK,
    DONE,
    Choice,
    Nav,
    Prompter,
    Review,
    State,
    Step,
    StepContext,
    Wizard,
)
from chronominer.discovery import (
    DiscoveryError,
    SourceItem,
    detect_input_type,
    discover_items,
    is_sidecar,
)
from chronominer.output.store import LINE_RANGES_SUFFIX
from chronominer.settings import Settings
from chronominer.spec import OPTIONS, PROVIDER_TABLES, Resolution

if TYPE_CHECKING:
    from rich.console import Console

__all__ = [
    "INVENTORY",
    "TASKS",
    "TASK_STEP",
    "TITLE",
    "Inventory",
    "RunRequest",
    "Session",
    "TaskFlow",
    "TaskSwitch",
    "absolute",
    "build_wizard",
    "can_reset",
    "command_words",
    "has_settings_default",
    "inventory_of",
    "new_state",
    "plural",
    "take_inventory",
    "task_step",
]

logger = logging.getLogger(__name__)

TITLE: Final = "ChronoMiner"
INVENTORY: Final = "inventory"
TASK_STEP: Final = "task"

TASKS: Mapping[str, str] = MappingProxyType(
    {
        "extract": "new extraction",
        "ranges-generate": "generate line ranges",
        "ranges-adjust": "adjust line ranges",
        "batch-check": "check batches",
        "batch-cancel": "cancel batches",
        "repair": "repair failed chunks",
        "render": "render files again",
        "drop": "drop stored documents",
    }
)
"""The wizard tasks in menu order, by name, with their titles."""


@dataclass(frozen=True)
class Session:
    """The settings of a guided session and where they came from.

    ``settings_path`` is the ``--settings`` file the ui was started with;
    ``environ`` is where the review looks up API key variables; ``args``
    holds the parsed ``ui`` arguments the wizard passes to the extension.
    """

    settings: Settings
    settings_path: Path | None = None
    environ: Mapping[str, str] = field(default_factory=lambda: os.environ)
    args: argparse.Namespace | None = None


def new_state(session: Session, table: OptionTable) -> State:
    """Return an empty answer state over *table* and its settings defaults."""
    defaults = {
        name: value
        for name, value in session.settings.defaults.items()
        if name in table and table[name].settable
    }
    return State(table, defaults)


_QUOTES = ('"', "'")


def absolute(path: Path | str) -> Path:
    """Make *path* absolute against the current directory.

    One matching pair of surrounding quotes is removed, as Explorer's
    "Copy as path" adds them, and ``~`` is expanded.
    """
    text = str(path).strip()
    if len(text) >= 2 and text[0] in _QUOTES and text[-1] == text[0]:
        text = text[1:-1]
    return Path(os.path.abspath(os.path.expanduser(text)))


def plural(count: int, word: str, many: str | None = None) -> str:
    """Return ``1 page`` or ``2 pages``."""
    return f"{count} {word if count == 1 else (many or word + 's')}"


def has_settings_default(state: State, name: str) -> bool:
    """Whether the settings defaults give option *name* a value."""
    current: str | None = name
    while current is not None:
        if current in state.defaults:
            return True
        current = state.table[current].fallback
    return False


def can_reset(state: State, name: str) -> bool:
    """Whether *name* may be reset to its unset code default.

    An unset value has no command-line spelling, so it cannot override a
    settings default; the wizard offers the reset only without one.
    """
    return not has_settings_default(state, name)


_OPENAI_ONLY: Final = "--service-tier"


def _without_flag(argv: Sequence[str], flag: str) -> list[str]:
    """Return *argv* without *flag*, in both its spellings, and its value."""
    words: list[str] = []
    skip = False
    for word in argv:
        if skip:
            skip = False
        elif word == flag:
            skip = True
        elif not word.startswith(f"{flag}="):
            words.append(word)
    return words


def command_words(
    program: Sequence[str],
    argv: Sequence[str],
    session: Session,
    table: OptionTable,
    *,
    provider: str | None = None,
) -> list[str]:
    """Return the full command: *program*, *argv*, settings and extension words.

    With a *provider* other than OpenAI, ``--service-tier`` is left out, as
    the review leaves out the tier. ``--settings`` is added when the ui was
    started with one and *table* has the option. The extension's own options
    given to the ui follow (:func:`chronominer.ext.extra_argv`) when the
    command registers them: the commands that call providers
    (:data:`~chronominer.spec.PROVIDER_TABLES`).
    """
    if provider is not None and provider != "openai":
        argv = _without_flag(argv, _OPENAI_ONLY)
    words = [*program, *argv]
    if session.settings_path is not None and "settings" in table:
        words += ["--settings", str(absolute(session.settings_path))]
    if any(table is provider_table for provider_table in PROVIDER_TABLES):
        words += ext.extra_argv(session.args)
    return words


class TaskSwitch(Exception):
    """The user chose another task on the task step."""

    def __init__(self, task: str) -> None:
        super().__init__(task)
        self.task = task


def task_step(current: str) -> Step:
    """Return the task step: DONE for *current*, TaskSwitch for another task."""

    def ask(ctx: StepContext) -> Nav:
        answer = ctx.prompter.select(
            "Task:",
            [Choice(name, title) for name, title in TASKS.items()],
            default=current,
        )
        if answer is BACK:
            return BACK
        if answer != current:
            raise TaskSwitch(answer)
        return DONE

    return Step(TASK_STEP, "Task", ask=ask)


class _TaskWizard(Wizard):
    """A wizard whose task step counts as answered on its first visit.

    It serves a task the user chose on an earlier wizard's task step: the
    first pass skips the task step without showing it, and Back from the
    task's first step returns to it.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._task_chosen = True

    def _ask(self, step: Step) -> Nav:
        if self._task_chosen and step.name == TASK_STEP:
            self._task_chosen = False
            self.visited.add(step.name)
            return DONE
        return super()._ask(step)


@dataclass(frozen=True)
class RunRequest:
    """What a task runs with after the wizard: answers, console and arguments.

    ``args`` holds the parsed ``ui`` arguments for the extension.
    """

    session: Session
    answers: Mapping[str, Any]
    console: Console
    args: argparse.Namespace


@dataclass(frozen=True)
class TaskFlow:
    """One wizard task: its steps, review, command, spec and run.

    ``program`` holds the leading command words, such as
    ``("chronominer", "batch", "check")``; ``steps`` returns the task's
    steps without the task step; ``command`` returns the full equivalent
    command, program first; ``run`` returns the exit code.
    """

    name: str
    title: str
    program: tuple[str, ...]
    table: OptionTable
    steps: Callable[[Session], Sequence[Step]]
    review: Callable[[State, Session], Review]
    command: Callable[[State, Session], list[str]]
    resolve: Callable[[State, Session], Resolution[Any]]
    run: Callable[[RunRequest], int]


def build_wizard(
    flow: TaskFlow,
    session: Session,
    prompter: Prompter,
    *,
    width: int = 80,
    chosen: bool = False,
) -> Wizard:
    """Return the wizard of *flow*: the task step, then the task's steps.

    The extension steps follow only for the extract options. *chosen*
    means the user chose the task on an earlier wizard's task step: the
    wizard then starts at the task's first step, and Back from it returns
    to the task step.
    """
    steps = [task_step(flow.name), *flow.steps(session)]
    if flow.table is OPTIONS:
        steps += ext.wizard_steps()
    wizard = _TaskWizard if chosen else Wizard
    return wizard(
        new_state(session, flow.table),
        prompter,
        steps,
        title=TITLE,
        task=flow.title,
        review=lambda state: flow.review(state, session),
        command=lambda state: flow.command(state, session),
        width=width,
    )


_ORDER: Final = ("text", "pdf", "image")
_LABELS: Final = {"text": "text file", "pdf": "PDF", "image": "image"}


def _page_count(path: Path) -> int | None:
    try:
        with fitz.open(path) as document:
            return int(document.page_count)
    except Exception as exc:  # noqa: BLE001 - an unreadable PDF has no count
        logger.debug("Could not count the pages of %s: %s", path, exc)
        return None


def _count(item: SourceItem) -> int | None:
    return _page_count(item.path) if item.type == "pdf" else None


def _has_ranges(item: SourceItem) -> bool:
    """Whether a text item has a ``_line_ranges.txt`` beside it."""
    if item.type != "text":
        return False
    path = item.path
    return path.with_name(f"{path.stem}{LINE_RANGES_SUFFIX}").is_file()


@dataclass(frozen=True)
class Inventory:
    """The items found at an input path, with page counts and line ranges.

    PDFs count pages; other items have no count. ``ranged`` marks the text
    items that have a ``_line_ranges.txt`` beside them.
    """

    path: Path
    items: tuple[SourceItem, ...]
    counts: tuple[int | None, ...]
    recursive: bool = False
    ranged: tuple[bool, ...] = ()

    @property
    def names(self) -> list[str]:
        """The item names."""
        return [item.name for item in self.items]

    @property
    def input_type(self) -> str | None:
        """The type the items share, ``mixed``, or None without items."""
        return detect_input_type(self.items)

    def has_ranges(self, index: int) -> bool:
        """Whether the item at 0-based *index* has line ranges."""
        return index < len(self.ranged) and self.ranged[index]

    def of_type(self, kind: str) -> list[int]:
        """Return the 0-based indices of the items of type *kind*."""
        return [index for index, item in enumerate(self.items) if item.type == kind]

    def describe(self) -> str:
        """Return ``2 text files (1 with line ranges), 1 PDF, 12 pages``.

        Without items the text is ``nothing to process``. Images found in
        more than one folder name the folder count.
        """
        parts: list[str] = []
        texts = self.of_type("text")
        if texts:
            text = plural(len(texts), "text file")
            ranged = sum(self.has_ranges(index) for index in texts)
            if ranged:
                text += f" ({ranged} with line ranges)"
            parts.append(text)
        pdfs = self.of_type("pdf")
        if pdfs:
            counts = [self.counts[index] for index in pdfs]
            total = sum(count or 0 for count in counts)
            unknown = " or more" if any(count is None for count in counts) else ""
            parts.append(
                f"{plural(len(pdfs), 'PDF')}, {plural(total, 'page')}{unknown}"
            )
        images = self.of_type("image")
        if images:
            text = plural(len(images), "image")
            folders = {self.items[index].path.parent for index in images}
            if len(folders) > 1:
                text += f" in {plural(len(folders), 'folder')}"
            parts.append(text)
        return ", ".join(parts) if parts else "nothing to process"

    def label(self, index: int) -> str:
        """Return one numbered list line for the item at 0-based *index*."""
        item, count = self.items[index], self.counts[index]
        detail = _LABELS.get(item.type, item.type)
        if count is not None:
            detail += f", {plural(count, 'page')}"
        if self.has_ranges(index):
            detail += ", line ranges"
        return f"{index + 1:>4}  {item.name}  ({detail})"


def _discover(path: Path, recursive: bool) -> list[SourceItem]:
    """Return what an auto-mode run would discover at *path*, by type."""
    if path.is_file() and is_sidecar(path):
        return []
    try:
        found = discover_items(path, "auto", recursive=recursive)
    except (DiscoveryError, OSError):
        return []
    return sorted(found, key=lambda item: _ORDER.index(item.type))


def take_inventory(path: Path, *, recursive: bool = False) -> Inventory:
    """Discover the items at *path*, count PDF pages and find line ranges.

    A folder is searched like auto mode (text, image and PDF files, no
    sidecars); *recursive* searches its subfolders as ``--recursive`` does.
    """
    items = tuple(_discover(path, recursive))
    return Inventory(
        path,
        items,
        tuple(_count(item) for item in items),
        recursive,
        tuple(_has_ranges(item) for item in items),
    )


def inventory_of(state: State, path: Path) -> Inventory:
    """Return the stored inventory of *path*, taking it when missing.

    The search is recursive when the state's ``recursive`` option is on.
    """
    recursive = bool("recursive" in state.table and state.value("recursive"))
    stored = state.data.get(INVENTORY)
    if (
        isinstance(stored, Inventory)
        and stored.path == path
        and stored.recursive == recursive
    ):
        return stored
    inventory = take_inventory(path, recursive=recursive)
    state.data[INVENTORY] = inventory
    return inventory
