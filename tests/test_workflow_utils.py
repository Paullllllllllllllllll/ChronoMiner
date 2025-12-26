from __future__ import annotations

from pathlib import Path

import pytest

from modules.core.workflow_utils import collect_text_files, filter_text_files


@pytest.mark.unit
def test_filter_text_files_skips_auxiliary(tmp_path: Path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b_context.txt"
    c = tmp_path / "c_line_ranges.txt"
    d = tmp_path / "d.json"

    for p in [a, b, c, d]:
        p.write_text("x", encoding="utf-8")

    filtered = filter_text_files([a, b, c, d])
    assert filtered == [a]


@pytest.mark.unit
def test_collect_text_files_walks_dirs(tmp_path: Path):
    (tmp_path / "x.txt").write_text("x", encoding="utf-8")
    (tmp_path / "y_context.txt").write_text("x", encoding="utf-8")

    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "z.txt").write_text("x", encoding="utf-8")

    files = collect_text_files(tmp_path)
    assert (tmp_path / "x.txt") in files
    assert (sub / "z.txt") in files
    assert (tmp_path / "y_context.txt") not in files
