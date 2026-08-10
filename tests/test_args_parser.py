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

    def test_generate_ranges_parser_accepts_mode_override_flags(self):
        """Regression: the only entry-point parser without the shared
        --interactive/--non-interactive flags made
        `generate_line_ranges.py --non-interactive` die with a usage error."""
        parser = create_generate_ranges_parser()
        args = parser.parse_args(["--input", "data/", "--non-interactive"])
        assert args.non_interactive is True
        args = parser.parse_args(["--input", "data/", "--interactive"])
        assert args.interactive is True


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


class TestSamplerBounds:
    """--temperature and --top-p document ranges (0.0-2.0 / 0.0-1.0) that were
    not enforced: a bare ``float`` type let an out-of-range value through to
    the provider, which rejects it only after the run has started."""

    @staticmethod
    def _process_args(*extra: str):
        return create_process_parser().parse_args(
            ["--schema", "Test", "--input", "data/", *extra]
        )

    def test_temperature_in_range_accepted(self):
        args = self._process_args("--temperature", "1.5")
        assert args.temperature == pytest.approx(1.5)

    def test_temperature_above_range_rejected(self):
        with pytest.raises(SystemExit):
            self._process_args("--temperature", "2.5")

    def test_temperature_negative_rejected(self):
        with pytest.raises(SystemExit):
            self._process_args("--temperature", "-0.1")

    def test_temperature_non_numeric_rejected(self):
        with pytest.raises(SystemExit):
            self._process_args("--temperature", "warm")

    def test_top_p_in_range_accepted(self):
        args = self._process_args("--top-p", "0.9")
        assert args.top_p == pytest.approx(0.9)

    def test_top_p_above_one_rejected(self):
        with pytest.raises(SystemExit):
            self._process_args("--top-p", "1.5")


class TestReadjusterNumericBounds:
    """line_range_readjuster.py accepted 0/negative --context-window and
    --max-output-tokens and unbounded sampler values; it must use the same
    validators as the other entry points."""

    @staticmethod
    def _parse(argv: list[str]):
        import sys as _sys

        import main.line_range_readjuster as lrr

        old_argv = _sys.argv
        try:
            _sys.argv = ["line_range_readjuster.py", *argv]
            return lrr.parse_arguments()
        finally:
            _sys.argv = old_argv

    def test_context_window_positive_accepted(self):
        args = self._parse(["--context-window", "12"])
        assert args.context_window == 12

    def test_context_window_zero_rejected(self):
        with pytest.raises(SystemExit):
            self._parse(["--context-window", "0"])

    def test_max_output_tokens_negative_rejected(self):
        with pytest.raises(SystemExit):
            self._parse(["--max-output-tokens", "-5"])

    def test_temperature_out_of_range_rejected(self):
        with pytest.raises(SystemExit):
            self._parse(["--temperature", "3"])

    def test_top_p_out_of_range_rejected(self):
        with pytest.raises(SystemExit):
            self._parse(["--top-p", "1.2"])
