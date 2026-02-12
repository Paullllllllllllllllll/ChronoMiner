from __future__ import annotations

from pathlib import Path

import pytest

from modules.cli.args_parser import (
    create_process_parser,
    create_generate_ranges_parser,
    get_files_from_path,
    resolve_path,
)


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


# ---------------------------------------------------------------------------
# Chunk-slice CLI arguments
# ---------------------------------------------------------------------------

class TestChunkSliceArgs:
    """Tests for --first-n-chunks and --last-n-chunks CLI arguments."""

    def test_process_parser_first_n(self):
        parser = create_process_parser()
        args = parser.parse_args([
            "--schema", "Test", "--input", "data/", "--first-n-chunks", "5"
        ])
        assert args.first_n_chunks == 5
        assert args.last_n_chunks is None

    def test_process_parser_last_n(self):
        parser = create_process_parser()
        args = parser.parse_args([
            "--schema", "Test", "--input", "data/", "--last-n-chunks", "3"
        ])
        assert args.last_n_chunks == 3
        assert args.first_n_chunks is None

    def test_process_parser_neither(self):
        parser = create_process_parser()
        args = parser.parse_args(["--schema", "Test", "--input", "data/"])
        assert args.first_n_chunks is None
        assert args.last_n_chunks is None

    def test_process_parser_mutual_exclusion(self):
        parser = create_process_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "--schema", "Test", "--input", "data/",
                "--first-n-chunks", "5", "--last-n-chunks", "3"
            ])

    def test_generate_ranges_parser_first_n(self):
        parser = create_generate_ranges_parser()
        args = parser.parse_args(["--input", "data/", "--first-n-chunks", "2"])
        assert args.first_n_chunks == 2
        assert args.last_n_chunks is None

    def test_generate_ranges_parser_last_n(self):
        parser = create_generate_ranges_parser()
        args = parser.parse_args(["--input", "data/", "--last-n-chunks", "7"])
        assert args.last_n_chunks == 7
        assert args.first_n_chunks is None
