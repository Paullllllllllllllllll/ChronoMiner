from __future__ import annotations

from pathlib import Path

import pytest

from modules.cli.args_parser import get_files_from_path, resolve_path


@pytest.mark.unit
def test_resolve_path_relative_uses_cwd(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = resolve_path("a/b.txt")
    assert p.is_absolute()
    assert str(p).endswith(str(Path("a") / "b.txt"))


@pytest.mark.unit
def test_get_files_from_path_excludes_output_dirs(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()

    (root / "in.txt").write_text("x", encoding="utf-8")

    out = root / "output"
    out.mkdir()
    (out / "out.txt").write_text("x", encoding="utf-8")

    outs = root / "something_outputs"
    outs.mkdir()
    (outs / "out2.txt").write_text("x", encoding="utf-8")

    files = get_files_from_path(root)
    assert (root / "in.txt") in files
    assert (out / "out.txt") not in files
    assert (outs / "out2.txt") not in files
