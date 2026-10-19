"""Styled console messages on stderr.

Stdout carries only the JSON summary line of ``--json``; every human-readable
message goes to stderr. Messages carry their ANSI style only when stderr is a
terminal and ``NO_COLOR`` is unset or empty, so logs hold plain text.
"""

from __future__ import annotations

import os
import sys
from typing import TextIO

import colorama

__all__ = [
    "print_error",
    "print_header",
    "print_info",
    "print_success",
    "print_warning",
    "ui_print",
]

colorama.just_fix_windows_console()

RULE = "="
HEADER = "\033[1;36m"
INFO = "\033[0;36m"
SUCCESS = "\033[1;32m"
WARNING = "\033[1;33m"
ERROR = "\033[1;31m"
RESET = "\033[0m"


def _color(stream: TextIO) -> bool:
    """Whether *stream* takes ANSI styles: a terminal, and no ``NO_COLOR``."""
    if os.environ.get("NO_COLOR"):
        return False
    try:
        return stream.isatty()
    except (AttributeError, ValueError):
        return False


def ui_print(message: str, style: str = "", end: str = "\n") -> None:
    """Write *message* to stderr, in the ANSI *style* on a terminal.

    Text the stream cannot encode is written with ``?`` in its place.
    """
    stream = sys.stderr
    if not _color(stream):
        style = ""
    try:
        text = f"{style}{message}{RESET}" if style else message
        stream.write(text + end)
    except UnicodeEncodeError:
        safe = message.encode("ascii", "replace").decode("ascii")
        stream.write((f"{style}{safe}{RESET}" if style else safe) + end)
    stream.flush()


def print_header(title: str, subtitle: str = "") -> None:
    """Write a section header framed by rules."""
    width = 80
    ui_print("\n" + RULE * width, HEADER)
    ui_print(f"  {title}", HEADER)
    if subtitle:
        ui_print(f"  {subtitle}", INFO)
    ui_print(RULE * width, HEADER)
    ui_print("")


def print_info(message: str, prefix: str = "[INFO]") -> None:
    """Write an informational message."""
    ui_print(f"{prefix} {message}", INFO)


def print_success(message: str, prefix: str = "[SUCCESS]") -> None:
    """Write a success message."""
    ui_print(f"{prefix} {message}", SUCCESS)


def print_warning(message: str, prefix: str = "[WARNING]") -> None:
    """Write a warning."""
    ui_print(f"{prefix} {message}", WARNING)


def print_error(message: str, prefix: str = "[ERROR]") -> None:
    """Write an error."""
    ui_print(f"{prefix} {message}", ERROR)
