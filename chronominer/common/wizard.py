"""Guided input over an option table: steps, Back navigation and a review.

A :class:`Wizard` runs hand-ordered :class:`Step` objects, each naming the
options it may set, against a :class:`State` that resolves its answers like
command-line flags over the settings defaults. The review screen shows
tool-supplied lines, marks values taken from settings defaults and prints the
equivalent command, quoted for PowerShell on Windows and for a POSIX shell
elsewhere. All screen input and output goes through a
:class:`Prompter`; this module imports no prompt library.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field
from enum import Enum, StrEnum
from typing import Any, Final, Literal, Protocol

from .options import Kind, Option, OptionError, OptionTable, Resolved, Source

__all__ = [
    "BACK",
    "BACK_HINT",
    "BACK_TEXT",
    "BACK_TITLE",
    "DONE",
    "NO_DEFAULT",
    "SETTINGS_LEGEND",
    "SETTINGS_MARKER",
    "Choice",
    "ClipboardRunner",
    "Decision",
    "Line",
    "Nav",
    "Outcome",
    "Part",
    "Prompter",
    "Review",
    "ReviewAction",
    "State",
    "Step",
    "StepContext",
    "Validator",
    "Wizard",
    "access_line",
    "choice_label",
    "clipboard_command",
    "command_display",
    "command_line",
    "copy_to_clipboard",
    "default_index",
    "option_validator",
    "quote_word",
    "render_review",
]

SETTINGS_MARKER: Final = "(s)"
SETTINGS_LEGEND: Final = "(s) = from settings defaults"
BACK_TITLE: Final = "Back"
BACK_TEXT: Final = "<"
BACK_HINT: Final = "('<' = Back)"


class Nav(Enum):
    """Navigation result of a question or a step."""

    BACK = "back"
    DONE = "done"


BACK: Final = Nav.BACK
DONE: Final = Nav.DONE


class _Unset(Enum):
    UNSET = "unset"


NO_DEFAULT: Final = _Unset.UNSET

Validator = Callable[[str], str | None]
"""Return an error message for invalid text, None to accept it."""


@dataclass(frozen=True)
class Choice[T]:
    """One entry of a select or checkbox list."""

    value: T
    title: str
    note: str | None = None


def choice_label(choice: Choice[Any], marker: str | None = None) -> str:
    """Return the displayed entry: title, optional marker, optional note."""
    label = choice.title
    if marker:
        label += f" {marker}"
    if choice.note:
        label += f"  {choice.note}"
    return label


def default_index(choices: Sequence[Choice[Any]], default: Any) -> int | None:
    """Return the index of the choice whose value equals ``default``."""
    if default is NO_DEFAULT:
        return None
    for index, choice in enumerate(choices):
        if choice.value == default:
            return index
    return None


class Prompter(Protocol):
    """Screen input and output of a wizard.

    Every select and checkbox list ends with a Back entry; text and path
    prompts return Back when the answer is ``<`` alone. ``marker`` is shown
    beside the preselected value. KeyboardInterrupt propagates.
    """

    def select[T](
        self,
        message: str,
        choices: Sequence[Choice[T]],
        *,
        default: T | _Unset = NO_DEFAULT,
        marker: str | None = None,
    ) -> T | Literal[Nav.BACK]:
        """Return the chosen value."""
        ...

    def checkbox[T](
        self,
        message: str,
        choices: Sequence[Choice[T]],
        *,
        checked: Collection[T] = (),
        marker: str | None = None,
    ) -> list[T] | Literal[Nav.BACK]:
        """Return the checked values in choice order."""
        ...

    def text(
        self,
        message: str,
        *,
        default: str = "",
        validate: Validator | None = None,
        marker: str | None = None,
    ) -> str | Literal[Nav.BACK]:
        """Return the entered text; a validator re-asks with its message."""
        ...

    def path(
        self,
        message: str,
        *,
        default: str = "",
        validate: Validator | None = None,
        only_directories: bool = False,
        marker: str | None = None,
    ) -> str | Literal[Nav.BACK]:
        """Return the entered path text, with path completion."""
        ...

    def show(self, text: str) -> None:
        """Write ``text`` and a line break to the screen."""
        ...


@dataclass(frozen=True)
class Part:
    """One part of a review line; flagged when it shows a settings default."""

    text: str
    from_settings: bool = False


class State:
    """Wizard answers over an option table and a settings defaults block.

    Answers act like command-line flags; reading a value resolves it with
    :meth:`OptionTable.resolve`. ``data`` holds tool values that are not
    options, such as what an input step discovered.
    """

    def __init__(
        self,
        table: OptionTable,
        defaults: Mapping[str, Any] | None = None,
        answers: Mapping[str, Any] | None = None,
        *,
        marker: str = SETTINGS_MARKER,
    ) -> None:
        self.table = table
        self.defaults: dict[str, Any] = dict(defaults or {})
        self.settings_marker = marker
        self.data: dict[str, Any] = {}
        self._answers: dict[str, Any] = {}
        for name, value in (answers or {}).items():
            self._option(name)
            self._answers[name] = value

    @property
    def answers(self) -> dict[str, Any]:
        """The stored answers, by option name."""
        return dict(self._answers)

    def resolved(self) -> dict[str, Resolved]:
        """Resolve every option: answer, then settings default, then default."""
        return self.table.resolve(self._answers, self.defaults)

    def value(self, name: str) -> Any:
        """Return the effective value of ``name``."""
        return self.resolved()[self._option(name).name].value

    def source(self, name: str) -> Source:
        """Return where the effective value of ``name`` comes from."""
        return self.resolved()[self._option(name).name].source

    def from_settings(self, name: str) -> bool:
        """Whether the effective value comes from the settings defaults."""
        return self.source(name) is Source.SETTINGS

    def marker(self, name: str) -> str | None:
        """Return the settings marker when the value is a settings default."""
        return self.settings_marker if self.from_settings(name) else None

    def is_answered(self, name: str) -> bool:
        """Whether an answer is stored for ``name``."""
        return self._option(name).name in self._answers

    def set(self, name: str, value: Any) -> None:
        """Answer ``name`` with ``value``.

        The answer is stored only when it differs from what resolution gives
        without it, so an unchanged settings default keeps its source.
        Setting an exclusive member to a value other than None or False
        clears the other members of its group.
        """
        option = self._option(name)
        if option.kind is Kind.LIST and isinstance(value, list):
            value = tuple(value)
        answers = {k: v for k, v in self._answers.items() if k != name}
        if option.exclusive is not None and value is not None and value is not False:
            answers = {
                k: v
                for k, v in answers.items()
                if self.table[k].exclusive != option.exclusive
            }
        if self.table.resolve(answers, self.defaults)[name].value != value:
            answers[name] = value
        self._answers = answers

    def clear(self, name: str) -> None:
        """Remove the answer for ``name``."""
        self._answers.pop(self._option(name).name, None)

    def part(self, name: str, text: str) -> Part:
        """Return a review part flagged when ``name`` is a settings default."""
        return Part(text, self.from_settings(name))

    def _option(self, name: str) -> Option:
        if name not in self.table:
            raise OptionError(f"unknown option {name!r}")
        return self.table[name]


def option_validator(option: Option, empty_ok: bool = False) -> Validator:
    """Return a validator that converts text with ``option``."""

    def validate(text: str) -> str | None:
        if empty_ok and not text.strip():
            return None
        try:
            option.convert(text.strip())
        except OptionError as exc:
            return str(exc)
        return None

    return validate


def _text_of(option: Option, value: Any) -> str:
    if value is None:
        return ""
    if option.kind is Kind.LIST:
        return ",".join(str(item) for item in value)
    return str(value)


Question = Callable[[], Nav | None]
"""One question of a step: DONE, BACK, or None when skipped."""


class StepContext:
    """What a step's ask function works with: the state and the prompter."""

    def __init__(self, state: State, prompter: Prompter) -> None:
        self.state = state
        self.prompter = prompter

    def select_option(
        self, name: str, message: str, choices: Sequence[Choice[Any]]
    ) -> Nav:
        """Ask for ``name`` from ``choices``, preselecting its value."""
        current = self.state.value(name)
        marker = None
        if default_index(choices, current) is not None:
            marker = self.state.marker(name)
        answer = self.prompter.select(message, choices, default=current, marker=marker)
        if answer is BACK:
            return BACK
        self.state.set(name, answer)
        return DONE

    def checkbox_option(
        self, name: str, message: str, choices: Sequence[Choice[Any]]
    ) -> Nav:
        """Ask for the list option ``name``, checking its current items."""
        option = self.state.table[name]
        current = self.state.value(name) or ()
        answer = self.prompter.checkbox(
            message, choices, checked=tuple(current), marker=self.state.marker(name)
        )
        if answer is BACK:
            return BACK
        value: Any = list(answer)
        if option.kind is Kind.LIST:
            value = option.convert(value)
        self.state.set(name, value)
        return DONE

    def text_option(
        self,
        name: str,
        message: str,
        *,
        validate: Validator | None = None,
        empty: Any = NO_DEFAULT,
    ) -> Nav:
        """Ask for ``name`` as text converted by its option.

        An empty answer sets ``empty`` when given; otherwise it is converted
        like any other text.
        """
        return self._typed(name, message, validate, empty, path=False)

    def path_option(
        self,
        name: str,
        message: str,
        *,
        validate: Validator | None = None,
        empty: Any = NO_DEFAULT,
        only_directories: bool = False,
    ) -> Nav:
        """Ask for ``name`` as a path with completion."""
        return self._typed(
            name, message, validate, empty, path=True, only_directories=only_directories
        )

    def _typed(
        self,
        name: str,
        message: str,
        validate: Validator | None,
        empty: Any,
        *,
        path: bool,
        only_directories: bool = False,
    ) -> Nav:
        option = self.state.table[name]
        if option.kind not in (Kind.VALUE, Kind.LIST):
            raise OptionError(f"{option.flag}: not a text option")
        convert = option_validator(option, empty_ok=empty is not NO_DEFAULT)

        def check(text: str) -> str | None:
            error = convert(text)
            if error is None and validate is not None:
                error = validate(text)
            return error

        default = _text_of(option, self.state.value(name))
        marker = self.state.marker(name)
        answer: str | Literal[Nav.BACK]
        if path:
            answer = self.prompter.path(
                message,
                default=default,
                validate=check,
                only_directories=only_directories,
                marker=marker,
            )
        else:
            answer = self.prompter.text(
                message, default=default, validate=check, marker=marker
            )
        if answer is BACK:
            return BACK
        text = answer.strip()
        if not text and empty is not NO_DEFAULT:
            self.state.set(name, empty)
        else:
            self.state.set(name, option.convert(text))
        return DONE

    @staticmethod
    def sequence(questions: Sequence[Question]) -> Nav:
        """Ask ``questions`` in order with Back between them.

        A question returning None was skipped and is passed over by Back.
        Back on the first asked question returns BACK.
        """
        history: list[int] = []
        index = 0
        while index < len(questions):
            result = questions[index]()
            if result is None:
                index += 1
            elif result is BACK:
                if not history:
                    return BACK
                index = history.pop()
            else:
                history.append(index)
                index += 1
        return DONE


def _always(state: State) -> bool:
    return True


@dataclass(frozen=True)
class Step:
    """One wizard step.

    ``sets`` names the options the step may set; ``applies`` decides from
    the state whether the step is asked; ``ask`` returns DONE or BACK.
    """

    name: str
    title: str
    _: KW_ONLY
    ask: Callable[[StepContext], Nav]
    sets: tuple[str, ...] = ()
    applies: Callable[[State], bool] = _always


@dataclass(frozen=True)
class Line:
    """One review line: a label and its parts."""

    label: str
    parts: Sequence[Part | str]


def access_line(routes: Iterable[tuple[str, Callable[[], str]]]) -> Line | None:
    """Return the review line that names how the run's model calls authenticate.

    Each route pairs a role, such as a phase or a client kind, with a
    function that describes its access. A route whose function raises
    ValueError is left out, since the plan names that error. Roles that share
    a description are named together, and a description shared by every
    route stands alone. None when no route is left.
    """
    roles: dict[str, list[str]] = {}
    skipped = False
    for role, describe in routes:
        try:
            text = describe()
        except ValueError:
            skipped = True
            continue
        roles.setdefault(text, []).append(role)
    if not roles:
        return None
    if len(roles) == 1 and not skipped:
        return Line("access", list(roles))
    return Line(
        "access", [f"{', '.join(names)}: {text}" for text, names in roles.items()]
    )


@dataclass(frozen=True)
class ReviewAction:
    """A tool action on the review screen; it changes the answers."""

    key: str
    title: str
    apply: Callable[[State], None]


@dataclass(frozen=True)
class Review:
    """What the review screen shows.

    ``notes`` are free lines below the summary, such as a plan preview;
    ``withheld`` is the reason Run is not offered.
    """

    lines: Sequence[Line]
    notes: Sequence[str] = ()
    actions: Sequence[ReviewAction] = ()
    withheld: str | None = None


class Decision(StrEnum):
    """How the wizard ended."""

    RUN = "run"
    COPY = "copy"
    QUIT = "quit"


@dataclass(frozen=True)
class Outcome:
    """The wizard's result: the decision, the answers and the command."""

    action: Decision
    answers: Mapping[str, Any] = field(default_factory=dict)
    command: tuple[str, ...] | None = None


_EDIT = "edit"
_RESERVED = frozenset(
    {Decision.RUN.value, Decision.COPY.value, Decision.QUIT.value, _EDIT}
)


def render_review(
    review: Review,
    *,
    command: Sequence[str] = (),
    command_error: str | None = None,
    width: int = 80,
    marker: str = SETTINGS_MARKER,
    legend: str = SETTINGS_LEGEND,
) -> str:
    """Return the review screen text.

    ``command`` holds the display lines of the equivalent command;
    ``command_error`` replaces it when the command is unavailable.
    """
    flagged = any(
        isinstance(part, Part) and part.from_settings
        for line in review.lines
        for part in line.parts
    )
    header = "Review"
    if flagged:
        header = header.ljust(max(width - len(legend), len(header) + 2)) + legend
    lines = [header]
    label_width = max((len(line.label) for line in review.lines), default=0) + 3
    for line in review.lines:
        texts = []
        for part in line.parts:
            if isinstance(part, str):
                texts.append(part)
            else:
                texts.append(
                    f"{part.text} {marker}" if part.from_settings else part.text
                )
        lines.append(
            ("  " + line.label.ljust(label_width) + " | ".join(texts)).rstrip()
        )
    if review.notes:
        lines.append("")
        lines.extend(review.notes)
    if review.withheld is not None:
        lines.extend(["", f"Run unavailable: {review.withheld}"])
    if command_error is not None:
        lines.extend(["", f"Equivalent command unavailable: {command_error}"])
    elif command:
        lines.extend(["", "Equivalent command:", *command])
    return "\n".join(lines)


class Wizard:
    """Run steps with Back navigation, then the review screen.

    ``command`` returns the words of the equivalent command; a ValueError
    from it is shown instead and withholds Copy. The answers of a step that
    stops applying are set aside, not lost: they return when the step
    applies again, unless the option was answered in the meantime.
    """

    def __init__(
        self,
        state: State,
        prompter: Prompter,
        steps: Sequence[Step],
        *,
        title: str,
        task: str | Callable[[State], str],
        review: Callable[[State], Review],
        command: Callable[[State], Sequence[str]] | None = None,
        width: int = 80,
    ) -> None:
        self.state = state
        self.prompter = prompter
        self.steps: tuple[Step, ...] = tuple(steps)
        self.title = title
        self.task = task
        self.review = review
        self.command = command
        self.width = width
        self.visited: set[str] = set()
        self._parked: dict[str, Any] = {}
        names = [step.name for step in self.steps]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate step names in {names}")
        for step in self.steps:
            for name in step.sets:
                if name not in state.table:
                    raise OptionError(f"step {step.name!r}: unknown option {name!r}")
        self._context = StepContext(state, prompter)

    def applicable(self) -> list[Step]:
        """Return the steps that apply to the current state."""
        return [step for step in self.steps if step.applies(self.state)]

    def header(self, step: Step) -> str:
        """Return the step header: title, task and step number.

        No total is shown: later answers change which steps apply.
        """
        applicable = self.applicable()
        task = self.task if isinstance(self.task, str) else self.task(self.state)
        position = applicable.index(step) + 1 if step in applicable else 0
        return f"{self.title} | {task}   step {position}"

    def run(self) -> Outcome:
        """Ask every applicable step, then show the review until a decision."""
        history: list[int] = []
        index = 0
        while True:
            if index >= len(self.steps):
                outcome = self._review_loop()
                if outcome is not None:
                    return outcome
                index = self._back(history)
                if index < 0:
                    if self._confirm_quit():
                        return Outcome(Decision.QUIT, self.state.answers)
                    index = len(self.steps)
                continue
            step = self.steps[index]
            if not step.applies(self.state):
                self._park(step)
                index += 1
                continue
            self._unpark(step)
            if self._ask(step) is DONE:
                history.append(index)
                index += 1
                continue
            previous = self._back(history)
            if previous >= 0:
                index = previous
            elif self._confirm_quit():
                return Outcome(Decision.QUIT, self.state.answers)

    def _ask(self, step: Step) -> Nav:
        self.prompter.show("\n" + self.header(step))
        result = step.ask(self._context)
        if result is DONE:
            self.visited.add(step.name)
        return result

    def _back(self, history: list[int]) -> int:
        while history:
            index = history.pop()
            if self.steps[index].applies(self.state):
                return index
        return -1

    def _confirm_quit(self) -> bool:
        answer = self.prompter.select(
            "Quit without running?",
            [Choice("stay", "No, stay"), Choice("quit", "Yes, quit")],
        )
        return answer == "quit"

    def _park(self, skipped: Step) -> None:
        """Set aside the answers of options only inapplicable steps set."""
        kept = {name for step in self.applicable() for name in step.sets}
        answers = self.state.answers
        for name in skipped.sets:
            if name not in kept and name in answers:
                self._parked[name] = answers[name]
                self.state.clear(name)

    def _unpark(self, step: Step) -> None:
        """Restore the set-aside answers of an applicable step.

        An answer is dropped instead when its option, or another member of
        its exclusive group, was answered after it was set aside.
        """
        for name in step.sets:
            if name not in self._parked:
                continue
            value = self._parked.pop(name)
            group = self.state.table[name].exclusive
            answered = self.state.answers
            if name in answered or (
                group is not None
                and any(
                    self.state.table[other].exclusive == group for other in answered
                )
            ):
                continue
            self.state.set(name, value)

    def _settle(self) -> None:
        """Park the answers of inapplicable steps, restore those applying again."""
        for step in self.steps:
            if step.applies(self.state):
                self._unpark(step)
            else:
                self._park(step)

    def _command(self) -> tuple[tuple[str, ...] | None, str | None]:
        if self.command is None:
            return None, None
        try:
            return tuple(self.command(self.state)), None
        except ValueError as exc:
            return None, str(exc)

    def _review_loop(self) -> Outcome | None:
        while True:
            self._settle()
            review = self.review(self.state)
            words, error = self._command()
            display = command_display(words, width=self.width) if words else []
            self.prompter.show(
                "\n"
                + render_review(
                    review,
                    command=display,
                    command_error=error,
                    width=self.width,
                    marker=self.state.settings_marker,
                )
            )
            choices = self._review_choices(review, words is not None)
            answer = self.prompter.select("Start:", choices)
            if answer is BACK:
                return None
            if answer == Decision.RUN.value:
                return Outcome(Decision.RUN, self.state.answers, words)
            if answer == Decision.COPY.value and words is not None:
                return Outcome(Decision.COPY, self.state.answers, words)
            if answer == Decision.QUIT.value:
                return Outcome(Decision.QUIT, self.state.answers, words)
            if answer == _EDIT:
                self._edit()
                continue
            for action in review.actions:
                if action.key == answer:
                    action.apply(self.state)

    def _review_choices(self, review: Review, has_command: bool) -> list[Choice[str]]:
        choices: list[Choice[str]] = []
        if review.withheld is None:
            choices.append(Choice(Decision.RUN.value, "Run"))
        for action in review.actions:
            if action.key in _RESERVED:
                raise ValueError(f"review action key {action.key!r} is reserved")
            choices.append(Choice(action.key, action.title))
        choices.append(Choice(_EDIT, "Edit a step"))
        if has_command:
            choices.append(Choice(Decision.COPY.value, "Copy command and quit"))
        choices.append(Choice(Decision.QUIT.value, "Quit"))
        return choices

    def _edit(self) -> None:
        applicable = self.applicable()
        answer = self.prompter.select(
            "Edit which step?",
            [Choice(step.name, step.title) for step in applicable],
        )
        if answer is BACK:
            return
        step = next(step for step in applicable if step.name == answer)
        if self._ask(step) is BACK:
            return
        for pending in self.steps:
            if not pending.applies(self.state):
                self._park(pending)
                continue
            self._unpark(pending)
            if pending.name not in self.visited and self._ask(pending) is BACK:
                return


def _is_windows(windows: bool | None) -> bool:
    return os.name == "nt" if windows is None else windows


_POWERSHELL_PLAIN = re.compile(r"[A-Za-z0-9_./\\:=+-]+(,[A-Za-z0-9_./\\:=+-]+)*")
_POWERSHELL_OPTION = re.compile(r"--?[A-Za-z][A-Za-z0-9-]*")
_POWERSHELL_QUOTES = re.compile("(['‘’‚‛])")


def _powershell_quote(word: str) -> str:
    r"""Quote ``word`` for PowerShell unless it reads literally as it is.

    Words left bare hold only letters, digits and ``_./\:=+-`` and inner
    commas, and do not begin with ``-`` unless they are a plain option name,
    since PowerShell splits ``-a.b`` at the dot. Any other word goes in single
    quotes, in which only a single quote is special; PowerShell also reads
    the typographic single quotes as one, so each of them is doubled.
    """
    if _POWERSHELL_PLAIN.fullmatch(word) and (
        not word.startswith("-") or _POWERSHELL_OPTION.fullmatch(word)
    ):
        return word
    return "'" + _POWERSHELL_QUOTES.sub(r"\1\1", word) + "'"


def quote_word(word: str, *, windows: bool | None = None) -> str:
    """Quote one word for PowerShell on Windows, for a POSIX shell elsewhere."""
    if _is_windows(windows):
        return _powershell_quote(word)
    return shlex.quote(word)


def command_line(words: Sequence[str], *, windows: bool | None = None) -> str:
    """Return ``words`` as one command line for the current platform.

    On Windows the line is quoted for PowerShell, elsewhere for a POSIX shell.
    """
    if _is_windows(windows):
        return " ".join(_powershell_quote(word) for word in words)
    return shlex.join(words)


def command_display(
    words: Sequence[str],
    *,
    width: int = 80,
    indent: int = 2,
    continuation: int = 4,
    windows: bool | None = None,
) -> list[str]:
    """Return the command wrapped for reading, continuation lines indented.

    A flag stays on one line with its value; the leading words before the
    first flag stay together.
    """
    units: list[str] = []
    seen_flag = awaiting_value = False
    for word in (quote_word(word, windows=windows) for word in words):
        is_flag = word.startswith("--")
        if units and not is_flag and (awaiting_value or not seen_flag):
            units[-1] = f"{units[-1]} {word}"
            awaiting_value = False
            continue
        units.append(word)
        seen_flag = seen_flag or is_flag
        awaiting_value = is_flag
    lines: list[str] = []
    current = " " * indent
    for unit in units:
        if not current.strip():
            current += unit
        elif len(current) + 1 + len(unit) > width:
            lines.append(current)
            current = " " * continuation + unit
        else:
            current = f"{current} {unit}"
    if current.strip():
        lines.append(current)
    return lines


ClipboardRunner = Callable[[Sequence[str], bytes], object]
"""Run a clipboard program with the given argv and stdin bytes."""


def clipboard_command(platform: str | None = None) -> tuple[list[str], str] | None:
    """Return the clipboard program and its input encoding, if known."""
    platform = sys.platform if platform is None else platform
    if platform == "win32":
        return ["clip"], "utf-16"
    if platform == "darwin":
        return ["pbcopy"], "utf-8"
    return None


def _run_clipboard(argv: Sequence[str], data: bytes) -> None:
    subprocess.run(list(argv), input=data, check=True, capture_output=True, timeout=10)


def copy_to_clipboard(
    text: str,
    *,
    runner: ClipboardRunner | None = None,
    platform: str | None = None,
) -> bool:
    """Copy ``text`` to the clipboard; return whether it worked. Never raises."""
    found = clipboard_command(platform)
    if found is None:
        return False
    argv, encoding = found
    try:
        (runner or _run_clipboard)(argv, text.encode(encoding))
    except Exception:  # noqa: BLE001 - a clipboard failure is reported as False
        return False
    return True
