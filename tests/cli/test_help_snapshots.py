"""Golden snapshots of the help of every command at a fixed width.

The files in ``golden/help/`` are the reference the documentation follows.
They hold the public option table only: the extension arguments are
replaced by a no-op, so the snapshots do not depend on the ext.py build.
``CM_UPDATE_GOLDEN=1`` rewrites them instead of comparing.
"""

from __future__ import annotations

import argparse
import difflib
import os
from pathlib import Path

import pytest

from chronominer import __main__ as entry
from chronominer import ext
from chronominer.common.testing import WriteGuard

GOLDEN_DIR = Path(__file__).resolve().parent / "golden" / "help"
UPDATE_ENV = "CM_UPDATE_GOLDEN"
COLUMNS = 100

CASES = {
    "chronominer": (),
    "extract": ("extract",),
    "ranges": ("ranges",),
    "ranges-generate": ("ranges", "generate"),
    "ranges-adjust": ("ranges", "adjust"),
    "batch": ("batch",),
    "batch-check": ("batch", "check"),
    "batch-cancel": ("batch", "cancel"),
    "repair": ("repair",),
    "render": ("render",),
    "drop": ("drop",),
    "classify": ("classify",),
    "classify-chunks": ("classify", "chunks"),
    "classify-entries": ("classify", "entries"),
    "classify-documents": ("classify", "documents"),
    "ui": ("ui",),
}


def _no_extension_args(parser: argparse.ArgumentParser) -> None:
    return None


def _help(capsys: pytest.CaptureFixture[str], argv: tuple[str, ...]) -> str:
    with pytest.raises(SystemExit) as exc:
        entry.main([*argv, "--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    return captured.out


@pytest.mark.parametrize("name", list(CASES))
def test_help_matches_the_golden(
    name: str,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    write_guard: WriteGuard,
) -> None:
    monkeypatch.setenv("COLUMNS", str(COLUMNS))
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setattr(ext, "register_args", _no_extension_args)
    actual = _help(capsys, CASES[name])
    golden = GOLDEN_DIR / f"{name}.txt"
    if os.environ.get(UPDATE_ENV) == "1":
        golden.parent.mkdir(parents=True, exist_ok=True)
        golden.write_text(actual, encoding="utf-8", newline="\n")
        write_guard.consume(GOLDEN_DIR.parent)
        return
    if not golden.is_file():
        pytest.fail(f"No golden {golden.name}; run with {UPDATE_ENV}=1 to create.")
    expected = golden.read_bytes().decode("utf-8")
    if actual != expected:
        diff = "".join(
            difflib.unified_diff(
                expected.splitlines(keepends=True),
                actual.splitlines(keepends=True),
                fromfile=f"golden/help/{name}.txt",
                tofile="help",
            )
        )
        pytest.fail(f"Help of {name!r} changed (set {UPDATE_ENV}=1 to accept):\n{diff}")
