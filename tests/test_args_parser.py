from __future__ import annotations

from pathlib import Path

import pytest

from main.cli_args import (
    create_generate_ranges_parser,
    create_process_parser,
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


@pytest.mark.unit
def test_get_files_from_path_excludes_context_files(tmp_path: Path):
    """Context files (_extract_context, _adjust_context, _transcr_context) must
    never be returned as processable input files."""
    root = tmp_path / "root"
    root.mkdir()

    # Legitimate input file
    (root / "document.txt").write_text("content", encoding="utf-8")

    # Context files that must be excluded
    (root / "document_extract_context.txt").write_text("ctx", encoding="utf-8")
    (root / "document_adjust_context.txt").write_text("ctx", encoding="utf-8")
    (root / "document_transcr_context.txt").write_text("ctx", encoding="utf-8")
    # Folder-level context
    (root / "root_extract_context.txt").write_text("ctx", encoding="utf-8")

    files = get_files_from_path(
        root, pattern="*.txt", exclude_patterns=["*_line_ranges.txt", "*_context.txt"]
    )
    assert (root / "document.txt") in files
    assert len(files) == 1, f"Expected only document.txt, got {[f.name for f in files]}"


# ---------------------------------------------------------------------------
# Chunk-slice CLI arguments
# ---------------------------------------------------------------------------


class TestChunkSliceArgs:
    """Tests for --first-n-chunks and --last-n-chunks CLI arguments."""

    def test_process_parser_first_n(self):
        parser = create_process_parser()
        args = parser.parse_args(
            ["--schema", "Test", "--input", "data/", "--first-n-chunks", "5"]
        )
        assert args.first_n_chunks == 5
        assert args.last_n_chunks is None

    def test_process_parser_last_n(self):
        parser = create_process_parser()
        args = parser.parse_args(
            ["--schema", "Test", "--input", "data/", "--last-n-chunks", "3"]
        )
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
            parser.parse_args(
                [
                    "--schema",
                    "Test",
                    "--input",
                    "data/",
                    "--first-n-chunks",
                    "5",
                    "--last-n-chunks",
                    "3",
                ]
            )

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


class TestGenerateRangesParserPositiveInt:
    """Regression: --tokens/--first-n-chunks/--last-n-chunks on the
    generate_line_ranges parser used plain ``int``, so 0 silently fell back
    and negatives degraded or died with a generic error rather than a clear
    ArgumentTypeError. They must use the same ``_positive_int`` validator as
    the process parser's --max-output-tokens/--chunk-size/etc.
    """

    def test_tokens_accepts_positive(self):
        parser = create_generate_ranges_parser()
        args = parser.parse_args(["--input", "data/", "--tokens", "5000"])
        assert args.tokens == 5000

    def test_tokens_zero_rejected(self):
        parser = create_generate_ranges_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--input", "data/", "--tokens", "0"])

    def test_tokens_negative_rejected(self):
        parser = create_generate_ranges_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--input", "data/", "--tokens", "-1"])

    def test_first_n_chunks_zero_rejected(self):
        parser = create_generate_ranges_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--input", "data/", "--first-n-chunks", "0"])

    def test_last_n_chunks_negative_rejected(self):
        parser = create_generate_ranges_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--input", "data/", "--last-n-chunks", "-3"])


class TestProcessParserModelOverrides:
    """Tests for model-related CLI override options in process parser."""

    def test_process_parser_model_overrides(self):
        parser = create_process_parser()
        args = parser.parse_args(
            [
                "--schema",
                "Test",
                "--input",
                "data/",
                "--model",
                "gpt-5-mini",
                "--reasoning-effort",
                "high",
                "--verbosity",
                "low",
                "--max-output-tokens",
                "8192",
                "--chunk-size",
                "6000",
            ]
        )

        assert args.model == "gpt-5-mini"
        assert args.reasoning_effort == "high"
        assert args.verbosity == "low"
        assert args.max_output_tokens == 8192
        assert args.chunk_size == 6000

    def test_process_parser_max_output_tokens_must_be_positive(self):
        parser = create_process_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    "--schema",
                    "Test",
                    "--input",
                    "data/",
                    "--max-output-tokens",
                    "0",
                ]
            )

    def test_process_parser_chunk_size_must_be_positive(self):
        parser = create_process_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    "--schema",
                    "Test",
                    "--input",
                    "data/",
                    "--chunk-size",
                    "0",
                ]
            )
