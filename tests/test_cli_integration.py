"""CLI integration tests for main scripts.

Tests CLI argument parsing and execution flow without LLM calls.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.integration
class TestProcessTextFilesCLI:
    """Integration tests for process_text_files.py CLI."""

    def test_cli_args_parser_basic(self):
        """Test basic CLI argument parsing."""
        from modules.cli.args_parser import create_process_parser

        parser = create_process_parser()
        args = parser.parse_args([
            "--input", "test_input.txt",
            "--schema", "TestSchema",
        ])

        assert args.input == "test_input.txt"
        assert args.schema == "TestSchema"

    def test_cli_args_parser_batch_mode(self):
        """Test batch mode CLI arguments."""
        from modules.cli.args_parser import create_process_parser

        parser = create_process_parser()
        args = parser.parse_args([
            "--input", "test_input.txt",
            "--schema", "TestSchema",
            "--batch",
        ])

        assert args.batch is True

    def test_cli_args_parser_chunking_options(self):
        """Test chunking CLI arguments."""
        from modules.cli.args_parser import create_process_parser

        parser = create_process_parser()
        args = parser.parse_args([
            "--input", "test_input.txt",
            "--schema", "TestSchema",
            "--chunking", "line_ranges",
        ])

        assert args.chunking == "line_ranges"

    def test_mode_detector_detects_cli_args(self, config_loader, monkeypatch):
        """Test mode detector identifies CLI mode when args provided."""
        from modules.cli.mode_detector import detect_execution_mode

        monkeypatch.setattr(sys, "argv", ["script", "--input", "file.txt"])

        is_interactive, has_cli_args = detect_execution_mode(config_loader)
        assert has_cli_args is True
        assert is_interactive is False

    def test_mode_detector_interactive_mode(self, config_loader, monkeypatch):
        """Test mode detector identifies interactive mode when no args."""
        from modules.cli.mode_detector import detect_execution_mode

        monkeypatch.setattr(sys, "argv", ["script"])

        is_interactive, has_cli_args = detect_execution_mode(config_loader)
        assert has_cli_args is False


@pytest.mark.integration
class TestGenerateLineRangesCLI:
    """Integration tests for generate_line_ranges.py CLI."""

    def test_cli_args_parser(self):
        """Test CLI argument parsing for generate_line_ranges."""
        from modules.cli.args_parser import create_generate_ranges_parser

        parser = create_generate_ranges_parser()
        args = parser.parse_args([
            "--input", "test_input.txt",
            "--tokens", "5000",
        ])

        assert args.input == "test_input.txt"
        assert args.tokens == 5000

    def test_generate_line_ranges_function(self, tmp_path: Path):
        """Test line range generation on actual file."""
        from main.generate_line_ranges import generate_line_ranges_for_file

        text_file = tmp_path / "test.txt"
        text_file.write_text("line1\nline2\nline3\nline4\nline5\n" * 10, encoding="utf-8")

        ranges = generate_line_ranges_for_file(
            text_file=text_file,
            default_tokens_per_chunk=20,
            model_name="gpt-4o",
        )

        assert isinstance(ranges, list)
        assert all(isinstance(r, tuple) and len(r) == 2 for r in ranges)
        assert ranges[0][0] == 1  # First range starts at line 1


@pytest.mark.integration
class TestCheckBatchesCLI:
    """Integration tests for check_batches.py CLI."""

    def test_cli_args_parser_basic(self):
        """Test CLI argument parsing for check_batches."""
        from modules.cli.args_parser import create_check_batches_parser

        parser = create_check_batches_parser()
        args = parser.parse_args([])

        # Parser should work without any args
        assert hasattr(args, "schema") or hasattr(args, "input")

    def test_cli_args_parser_with_schema(self):
        """Test CLI argument parsing with schema."""
        from modules.cli.args_parser import create_check_batches_parser

        parser = create_check_batches_parser()
        args = parser.parse_args([
            "--schema", "TestSchema",
        ])

        assert args.schema == "TestSchema"


@pytest.mark.integration
class TestCancelBatchesCLI:
    """Integration tests for cancel_batches.py CLI."""

    def test_cli_args_parser_basic(self):
        """Test CLI argument parsing for cancel_batches."""
        from modules.cli.args_parser import create_cancel_batches_parser

        parser = create_cancel_batches_parser()
        args = parser.parse_args([])

        # Parser should work without any args
        assert hasattr(args, "force")

    def test_cli_args_parser_force(self):
        """Test CLI argument parsing with force flag."""
        from modules.cli.args_parser import create_cancel_batches_parser

        parser = create_cancel_batches_parser()
        args = parser.parse_args(["--force"])

        assert args.force is True


@pytest.mark.integration
class TestRepairExtractionsCLI:
    """Integration tests for repair_extractions.py CLI."""

    def test_cli_args_parser(self):
        """Test CLI argument parsing for repair_extractions."""
        from modules.cli.args_parser import create_repair_parser

        parser = create_repair_parser()
        args = parser.parse_args([
            "--schema", "TestSchema",
        ])

        assert args.schema == "TestSchema"

    def test_cli_args_parser_with_files(self):
        """Test CLI argument parsing with specific files."""
        from modules.cli.args_parser import create_repair_parser

        parser = create_repair_parser()
        args = parser.parse_args([
            "--files", "file1.jsonl", "file2.jsonl",
        ])

        assert args.files == ["file1.jsonl", "file2.jsonl"]


@pytest.mark.integration
class TestAdjustRangesCLI:
    """Integration tests for line_range_readjuster.py CLI."""

    def test_cli_args_parser(self):
        """Test CLI argument parsing for adjust ranges."""
        from modules.cli.args_parser import create_adjust_ranges_parser

        parser = create_adjust_ranges_parser()
        args = parser.parse_args([
            "--input", "test_file.txt",
            "--schema", "TestSchema",
        ])

        assert args.input == "test_file.txt"
        assert args.schema == "TestSchema"

    def test_cli_args_parser_with_context(self):
        """Test CLI argument parsing with context options."""
        from modules.cli.args_parser import create_adjust_ranges_parser

        parser = create_adjust_ranges_parser()
        args = parser.parse_args([
            "--input", "test_file.txt",
            "--schema", "TestSchema",
            "--use-context",
            "--context-window", "15",
        ])

        assert args.use_context is True
        assert args.context_window == 15


@pytest.mark.integration
class TestExampleFilesDiscovery:
    """Test file discovery using example_files directory structure."""

    def test_discover_files_from_directory(self, tmp_path: Path):
        """Test discovering text files from a directory."""
        from modules.cli.args_parser import get_files_from_path

        # Create test structure
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "file1.txt").write_text("content1", encoding="utf-8")
        (input_dir / "file2.txt").write_text("content2", encoding="utf-8")
        (input_dir / "file3.json").write_text("{}", encoding="utf-8")

        sub_dir = input_dir / "subdir"
        sub_dir.mkdir()
        (sub_dir / "file4.txt").write_text("content4", encoding="utf-8")

        files = get_files_from_path(input_dir)

        txt_files = [f for f in files if f.suffix == ".txt"]
        assert len(txt_files) == 3
        assert (input_dir / "file1.txt") in txt_files
        assert (input_dir / "file2.txt") in txt_files
        assert (sub_dir / "file4.txt") in txt_files

    def test_discover_single_file(self, tmp_path: Path):
        """Test file discovery with single file path."""
        from modules.cli.args_parser import get_files_from_path

        single_file = tmp_path / "single.txt"
        single_file.write_text("content", encoding="utf-8")

        files = get_files_from_path(single_file)

        assert len(files) == 1
        assert files[0] == single_file


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
