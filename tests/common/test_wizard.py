"""The equivalent command of common/wizard.py, quoted for the platform's shell."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from chronominer.common.wizard import command_display, command_line, quote_word

AWKWARD = [
    "C:\\a b\\x.pdf",
    "a;b",
    "$HOME",
    "back`tick",
    'say "hi"',
    "it's",
    "it\u2019s",
    "",
    "--model=-x",
    "-x.y",
    "@list",
    "<tmp>",
]


@pytest.mark.parametrize(
    ("word", "quoted"),
    [
        ("plain", "plain"),
        ("C:\\plain\\path.pdf", "C:\\plain\\path.pdf"),
        ("md,txt", "md,txt"),
        ("--formats", "--formats"),
        ("with space", "'with space'"),
        ("a;b", "'a;b'"),
        ("$HOME", "'$HOME'"),
        ("back`tick", "'back`tick'"),
        ('say "hi"', "'say \"hi\"'"),
        ("it's", "'it''s'"),
        ("it\u2019s", "'it\u2019\u2019s'"),
        ("", "''"),
        ("--model=-x", "'--model=-x'"),
        ("-x.y", "'-x.y'"),
    ],
)
def test_windows_words_are_quoted_for_powershell(word: str, quoted: str) -> None:
    assert quote_word(word, windows=True) == quoted


def test_posix_words_are_quoted_for_a_posix_shell() -> None:
    assert quote_word("it's", windows=False) == "'it'\"'\"'s'"
    assert command_line(["tool", "a b", "$x"], windows=False) == "tool 'a b' '$x'"


def test_the_display_quotes_like_the_command_line() -> None:
    words = ["tool", "run", "--input", "C:\\it's here\\a;b.pdf", "--model", "-m.x"]
    line = command_line(words, windows=True)
    shown = command_display(words, width=200, windows=True)
    assert shown == ["  " + line]
    assert line == "tool run --input 'C:\\it''s here\\a;b.pdf' --model '-m.x'"


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="needs PowerShell 7")
def test_powershell_passes_the_quoted_words_unchanged(tmp_path: Path) -> None:
    probe = tmp_path / "argv.py"
    probe.write_text(
        "import json, sys\nprint(json.dumps(sys.argv[1:]))\n", encoding="utf-8"
    )
    script = tmp_path / "paste.ps1"
    line = command_line([sys.executable, str(probe), *AWKWARD], windows=True)
    script.write_text("& " + line + "\n", encoding="utf-8")
    result = subprocess.run(
        ["pwsh", "-NoProfile", "-NonInteractive", "-File", str(script)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == AWKWARD
