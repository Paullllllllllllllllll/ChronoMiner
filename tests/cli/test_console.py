"""The styled console messages write to stderr and leave stdout empty."""

from __future__ import annotations

import pytest

from chronominer.cli.console import (
    INFO,
    RESET,
    WARNING,
    print_error,
    print_header,
    print_info,
    print_success,
    print_warning,
    ui_print,
)


def test_messages_go_to_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    print_info("information")
    print_success("done")
    print_warning("careful")
    print_error("broken")
    print_header("Title", "Subtitle")

    captured = capsys.readouterr()
    assert captured.out == ""
    for text in (
        "[INFO] information",
        "[SUCCESS] done",
        "[WARNING] careful",
        "[ERROR] broken",
        "Title",
        "Subtitle",
    ):
        assert text in captured.err


def test_empty_message_is_a_newline(capsys: pytest.CaptureFixture[str]) -> None:
    ui_print("")
    assert capsys.readouterr().err == "\n"


def test_unencodable_text_falls_back_to_ascii(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    written: list[str] = []

    class _Ascii:
        def write(self, text: str) -> None:
            text.encode("ascii")
            written.append(text)

        def flush(self) -> None:
            pass

    monkeypatch.setattr("sys.stderr", _Ascii())
    ui_print("café")
    assert written == ["caf?\n"]


class _Stream:
    """A stderr stand-in that is a terminal or not."""

    def __init__(self, tty: bool, ascii_only: bool = False) -> None:
        self.tty = tty
        self.ascii_only = ascii_only
        self.text = ""

    def isatty(self) -> bool:
        return self.tty

    def write(self, text: str) -> None:
        if self.ascii_only:
            text.encode("ascii")
        self.text += text

    def flush(self) -> None:
        pass


@pytest.mark.parametrize(
    ("tty", "no_color", "styled"),
    [
        (True, None, True),
        (True, "", True),
        (True, "1", False),
        (False, None, False),
    ],
)
def test_the_style_needs_a_terminal_without_no_color(
    monkeypatch: pytest.MonkeyPatch, tty: bool, no_color: str | None, styled: bool
) -> None:
    stream = _Stream(tty)
    monkeypatch.setattr("sys.stderr", stream)
    if no_color is None:
        monkeypatch.delenv("NO_COLOR", raising=False)
    else:
        monkeypatch.setenv("NO_COLOR", no_color)

    print_info("information")

    plain = "[INFO] information\n"
    assert stream.text == (f"{INFO}[INFO] information{RESET}\n" if styled else plain)


def test_a_terminal_keeps_the_style_in_the_ascii_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = _Stream(True, ascii_only=True)
    monkeypatch.setattr("sys.stderr", stream)
    monkeypatch.delenv("NO_COLOR", raising=False)

    print_warning("café")

    assert stream.text == f"{WARNING}[WARNING] caf?{RESET}\n"
