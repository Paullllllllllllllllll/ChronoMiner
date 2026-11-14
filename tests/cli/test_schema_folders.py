"""Schema folders on the command line: ``--schemas-dir`` and ``schemas_dirs``.

Every command that resolves a schema searches the flag folders, then the
settings folders, then the repository's ``schemas/``; a name in two user
folders exits 2 naming both files, also where an unknown name is passed over
(the ``--schema`` filter of repair and batch check). The schemas are copies of
a bundled example under an invented name.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from chronominer import __main__ as entry
from chronominer.spec import SCHEMAS_DIR
from tests.classify.conftest import Chunk, seed, write_text
from tests.cli.conftest import json_lines

NAME = "ProjectLedger"
"""An invented schema name that no bundled schema has."""
CHUNKS = (Chunk(1, 20, entries=({"item": "bread"},)), Chunk(21, 40))
"""A stored document with one done chunk and nothing to repair."""


def _schema_folder(folder: Path, file_name: str = "project_ledger.json") -> Path:
    """Write a copy of a bundled example named :data:`NAME` into *folder*."""
    folder.mkdir(parents=True, exist_ok=True)
    body = json.loads((SCHEMAS_DIR / "price_list_entries.json").read_text("utf-8"))
    path = folder / file_name
    path.write_text(json.dumps({**body, "name": NAME}), encoding="utf-8")
    return path


def _run(
    capsys: pytest.CaptureFixture[str], *argv: str | Path
) -> tuple[int, dict[str, Any], str]:
    """Run a command with ``--json``; return its exit code, summary and stderr."""
    code = entry.main([*(str(arg) for arg in argv), "--json"])
    captured = capsys.readouterr()
    (summary,) = json_lines(captured.out)
    return code, summary, captured.err


def _extract(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, *extra: str | Path
) -> tuple[int, str]:
    text = write_text(tmp_path / "in" / "prices.txt")
    code, _summary, err = _run(
        capsys,
        "extract",
        "--schema",
        NAME,
        "--input",
        text,
        "--output",
        tmp_path / "out",
        "--dry-run",
        *extra,
    )
    return code, err


def test_extract_finds_a_schema_in_a_schemas_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = _schema_folder(tmp_path / "project").parent

    assert _extract(capsys, tmp_path)[0] == 2
    assert _extract(capsys, tmp_path, "--schemas-dir", folder)[0] == 0


def test_extract_finds_a_schema_in_the_settings_folders(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = _schema_folder(tmp_path / "project").parent
    settings = tmp_path / "settings.yaml"
    settings.write_text(f"schemas_dirs: ['{folder.as_posix()}']\n", encoding="utf-8")

    code, _err = _extract(capsys, tmp_path, "--settings", settings)
    assert code == 0
    same = _extract(capsys, tmp_path, "--settings", settings, "--schemas-dir", folder)
    assert same[0] == 0


def test_an_unknown_name_names_the_folders_searched(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = tmp_path / "empty"
    folder.mkdir()

    code, err = _extract(capsys, tmp_path, "--schemas-dir", folder)

    assert code == 2
    assert str(folder.resolve()) in err and str(SCHEMAS_DIR) in err


def test_a_schemas_dir_must_be_a_folder(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _summary, err = _run(
        capsys, "extract", "--schemas-dir", tmp_path / "missing", "--input", tmp_path
    )

    assert code == 2
    assert "not a folder" in err


def _two_folders(tmp_path: Path) -> tuple[Path, Path]:
    first = _schema_folder(tmp_path / "one")
    second = _schema_folder(tmp_path / "two", "copy.json")
    return first, second


def test_a_name_in_a_flag_and_a_settings_folder_is_ambiguous(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first, second = _two_folders(tmp_path)
    settings = tmp_path / "settings.yaml"
    settings.write_text(
        f"schemas_dirs: ['{second.parent.as_posix()}']\n", encoding="utf-8"
    )

    code, err = _extract(
        capsys, tmp_path, "--schemas-dir", first.parent, "--settings", settings
    )

    assert code == 2
    assert "ambiguous" in err
    assert str(first.resolve()) in err and str(second) in err


def test_ranges_adjust_refuses_an_ambiguous_name(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first, second = _two_folders(tmp_path)
    text = write_text(tmp_path / "in" / "prices.txt")
    (tmp_path / "in" / "prices_line_ranges.txt").write_text(
        "(1, 20)\n(21, 40)\n", encoding="utf-8"
    )

    code, _summary, err = _run(
        capsys,
        "ranges",
        "adjust",
        "--input",
        text,
        "--schema",
        NAME,
        "--schemas-dir",
        first.parent,
        "--schemas-dir",
        second.parent,
        "--dry-run",
    )

    assert code == 2
    assert "ambiguous" in err


@pytest.mark.parametrize("command", [("repair",), ("batch", "check")])
def test_the_schema_filter_does_not_pass_over_an_ambiguous_name(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], command: tuple[str, ...]
) -> None:
    first, second = _two_folders(tmp_path)
    out = tmp_path / "out"
    seed(out, write_text(tmp_path / "in" / "prices.txt"), CHUNKS)

    code, _summary, err = _run(
        capsys,
        *command,
        "--input",
        out,
        "--schema",
        NAME,
        "--schemas-dir",
        first.parent,
        "--schemas-dir",
        second.parent,
    )

    assert code == 2
    assert "ambiguous" in err
    assert str(first.resolve()) in err and str(second.resolve()) in err


@pytest.mark.parametrize("command", [("repair",), ("batch", "check")])
def test_the_schema_filter_passes_over_an_unknown_name(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], command: tuple[str, ...]
) -> None:
    out = tmp_path / "out"
    seed(out, write_text(tmp_path / "in" / "prices.txt"), CHUNKS)

    code, _summary, err = _run(capsys, *command, "--input", out, "--schema", NAME)

    assert code == 0, err
    assert "ambiguous" not in err
