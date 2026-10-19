"""Prompter on questionary: arrow-key lists, path completion, Back entries.

Input and output are prompt_toolkit objects and can be injected, so tests
drive the prompts through a pipe. KeyboardInterrupt propagates.
"""

from __future__ import annotations

import sys
from collections.abc import Collection, Sequence
from typing import Any, Literal, TextIO

import questionary
from prompt_toolkit.input import Input
from prompt_toolkit.output import Output

from .wizard import (
    BACK,
    BACK_HINT,
    BACK_TEXT,
    BACK_TITLE,
    NO_DEFAULT,
    Choice,
    Nav,
    Validator,
    choice_label,
    default_index,
)

__all__ = ["QuestionaryPrompter"]

_BACK_INDEX = -1


class QuestionaryPrompter:
    """A wizard prompter on questionary.

    ``input`` and ``output`` default to the terminal; ``stream`` receives
    :meth:`show` text and defaults to standard output.
    """

    def __init__(
        self,
        *,
        input: Input | None = None,
        output: Output | None = None,
        stream: TextIO | None = None,
    ) -> None:
        self._input = input
        self._output = output
        self._stream = stream

    def _io(self) -> dict[str, Any]:
        io: dict[str, Any] = {}
        if self._input is not None:
            io["input"] = self._input
        if self._output is not None:
            io["output"] = self._output
        return io

    def select[T](
        self,
        message: str,
        choices: Sequence[Choice[T]],
        *,
        default: Any = NO_DEFAULT,
        marker: str | None = None,
    ) -> T | Literal[Nav.BACK]:
        """Return the chosen value or BACK."""
        selected = default_index(choices, default)
        items = [
            questionary.Choice(
                choice_label(choice, marker if index == selected else None),
                value=index,
            )
            for index, choice in enumerate(choices)
        ]
        items.append(questionary.Choice(BACK_TITLE, value=_BACK_INDEX))
        preselected = items[selected] if selected is not None else None
        answer = questionary.select(
            message, items, default=preselected, **self._io()
        ).unsafe_ask()
        if answer is None or answer == _BACK_INDEX:
            return BACK
        return choices[int(answer)].value

    def checkbox[T](
        self,
        message: str,
        choices: Sequence[Choice[T]],
        *,
        checked: Collection[T] = (),
        marker: str | None = None,
    ) -> list[T] | Literal[Nav.BACK]:
        """Return the checked values, or BACK when Back is checked."""
        items = []
        for index, choice in enumerate(choices):
            on = choice.value in checked
            items.append(
                questionary.Choice(
                    choice_label(choice, marker if on else None),
                    value=index,
                    checked=on,
                )
            )
        items.append(questionary.Choice(BACK_TITLE, value=_BACK_INDEX))
        answer = questionary.checkbox(message, items, **self._io()).unsafe_ask()
        if answer is None or _BACK_INDEX in answer:
            return BACK
        return [choices[index].value for index in sorted(answer)]

    def text(
        self,
        message: str,
        *,
        default: str = "",
        validate: Validator | None = None,
        marker: str | None = None,
    ) -> str | Literal[Nav.BACK]:
        """Return the entered text, or BACK for ``<``."""
        answer = questionary.text(
            message,
            default=default,
            validate=_check(validate),
            instruction=_hint(marker),
            **self._io(),
        ).unsafe_ask()
        return _answer(answer)

    def path(
        self,
        message: str,
        *,
        default: str = "",
        validate: Validator | None = None,
        only_directories: bool = False,
        marker: str | None = None,
    ) -> str | Literal[Nav.BACK]:
        """Return the entered path text, or BACK for ``<``."""
        answer = questionary.path(
            f"{message} {_hint(marker)}",
            default=default,
            validate=_check(validate),
            only_directories=only_directories,
            **self._io(),
        ).unsafe_ask()
        return _answer(answer)

    def show(self, text: str) -> None:
        """Write ``text`` and a line break."""
        stream = self._stream if self._stream is not None else sys.stdout
        stream.write(text + "\n")
        stream.flush()


def _hint(marker: str | None) -> str:
    return f"{marker} {BACK_HINT}" if marker else BACK_HINT


def _check(validate: Validator | None) -> Any:
    def check(text: str) -> bool | str:
        if validate is None or text.strip() == BACK_TEXT:
            return True
        error = validate(text)
        return True if error is None else error

    return check


def _answer(answer: Any) -> str | Literal[Nav.BACK]:
    if answer is None or str(answer).strip() == BACK_TEXT:
        return BACK
    return str(answer)
