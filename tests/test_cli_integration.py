"""CLI integration tests for main scripts.

Tests CLI argument parsing and execution flow without LLM calls.
"""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest


@pytest.mark.integration
class TestProcessTextFilesCLI:
    """Integration tests for process_text_files.py CLI."""

    def test_cli_args_parser_basic(self):
        """Test basic CLI argument parsing."""
        from main.cli_args import create_process_parser

        parser = create_process_parser()
        args = parser.parse_args([
            "--input", "test_input.txt",
            "--schema", "TestSchema",
        ])

        assert args.input == "test_input.txt"
        assert args.schema == "TestSchema"

    def test_cli_args_parser_batch_mode(self):
        """Test batch mode CLI arguments."""
        from main.cli_args import create_process_parser

        parser = create_process_parser()
        args = parser.parse_args([
            "--input", "test_input.txt",
            "--schema", "TestSchema",
            "--batch",
        ])

        assert args.batch is True

    def test_cli_args_parser_chunking_options(self):
        """Test chunking CLI arguments."""
        from main.cli_args import create_process_parser

        parser = create_process_parser()
        args = parser.parse_args([
            "--input", "test_input.txt",
            "--schema", "TestSchema",
            "--chunking", "line_ranges",
        ])

        assert args.chunking == "line_ranges"

    def test_cli_args_parser_model_override_options(self):
        """Test model override arguments in process_text_files parser."""
        from main.cli_args import create_process_parser

        parser = create_process_parser()
        args = parser.parse_args([
            "--input", "test_input.txt",
            "--schema", "TestSchema",
            "--model", "gpt-5.2",
            "--reasoning-effort", "high",
            "--verbosity", "medium",
            "--max-output-tokens", "16384",
            "--chunk-size", "7000",
        ])

        assert args.model == "gpt-5.2"
        assert args.reasoning_effort == "high"
        assert args.verbosity == "medium"
        assert args.max_output_tokens == 16384
        assert args.chunk_size == 7000

    def test_build_effective_model_config_applies_overrides(self):
        """CLI model overrides should be merged into effective model config only."""
        from main.process_text_files import _build_effective_model_config

        base = {
            "extraction_model": {
                "name": "gpt-4o",
                "max_output_tokens": 4096,
                "reasoning": {"effort": "medium"},
                "text": {"verbosity": "high"},
            }
        }
        args = Namespace(
            model="gpt-5-mini",
            max_output_tokens=20000,
            reasoning_effort="low",
            verbosity="low",
        )

        effective = _build_effective_model_config(base, args)

        assert effective["extraction_model"]["name"] == "gpt-5-mini"
        assert effective["extraction_model"]["max_output_tokens"] == 20000
        assert effective["extraction_model"]["reasoning"]["effort"] == "low"
        assert effective["extraction_model"]["text"]["verbosity"] == "low"
        # Ensure original config was not mutated
        assert base["extraction_model"]["name"] == "gpt-4o"
        assert base["extraction_model"]["max_output_tokens"] == 4096

    def test_build_effective_paths_config_output_disables_input_as_output(self):
        """When --output is provided, output path mode should be enabled."""
        from main.process_text_files import _build_effective_paths_config

        base_paths = {"general": {"input_paths_is_output_path": True}}
        args = Namespace(output="C:/tmp/output")

        effective = _build_effective_paths_config(base_paths, args)
        assert effective["general"]["input_paths_is_output_path"] is False
        assert base_paths["general"]["input_paths_is_output_path"] is True

    def test_build_effective_chunking_config_applies_chunk_size_override(self):
        """CLI chunk-size should override per-run chunking config only."""
        from main.process_text_files import _build_effective_chunking_config

        base = {"chunking": {"default_tokens_per_chunk": 10000, "other_setting": True}}
        args = Namespace(chunk_size=4200)

        effective = _build_effective_chunking_config(base, args)

        assert effective["chunking"]["default_tokens_per_chunk"] == 4200
        assert effective["chunking"]["other_setting"] is True
        assert base["chunking"]["default_tokens_per_chunk"] == 10000

    def test_build_effective_chunking_config_keeps_default_when_no_override(self):
        """Without --chunk-size, effective chunking config should remain unchanged."""
        from main.process_text_files import _build_effective_chunking_config

        base = {"chunking": {"default_tokens_per_chunk": 10000}}
        args = Namespace(chunk_size=None)

        effective = _build_effective_chunking_config(base, args)

        assert effective["chunking"]["default_tokens_per_chunk"] == 10000
        assert effective is not base

    def test_cli_args_parser_temperature_and_top_p(self):
        """Test --temperature and --top-p argument parsing."""
        from main.cli_args import create_process_parser

        parser = create_process_parser()
        args = parser.parse_args([
            "--input", "test_input.txt",
            "--schema", "TestSchema",
            "--temperature", "0.5",
            "--top-p", "0.9",
        ])

        assert args.temperature == 0.5
        assert args.top_p == 0.9

    def test_build_effective_model_config_temperature_override(self):
        """CLI --temperature should override config value without mutating original."""
        from main.process_text_files import _build_effective_model_config

        base = {
            "extraction_model": {
                "name": "gpt-4o",
                "temperature": 0.0,
                "top_p": 1.0,
            }
        }
        args = Namespace(
            model=None, max_output_tokens=None,
            reasoning_effort=None, verbosity=None,
            temperature=0.7, top_p=None,
        )

        effective = _build_effective_model_config(base, args)

        assert effective["extraction_model"]["temperature"] == 0.7
        assert effective["extraction_model"]["top_p"] == 1.0
        assert base["extraction_model"]["temperature"] == 0.0

    def test_build_effective_model_config_top_p_override(self):
        """CLI --top-p should override config value without mutating original."""
        from main.process_text_files import _build_effective_model_config

        base = {
            "extraction_model": {
                "name": "gpt-4o",
                "temperature": 0.0,
                "top_p": 1.0,
            }
        }
        args = Namespace(
            model=None, max_output_tokens=None,
            reasoning_effort=None, verbosity=None,
            temperature=None, top_p=0.85,
        )

        effective = _build_effective_model_config(base, args)

        assert effective["extraction_model"]["top_p"] == 0.85
        assert effective["extraction_model"]["temperature"] == 0.0
        assert base["extraction_model"]["top_p"] == 1.0

    def test_build_effective_model_config_no_override_preserves_defaults(self):
        """When no CLI flags are set, all config values should be preserved."""
        from main.process_text_files import _build_effective_model_config

        base = {
            "extraction_model": {
                "name": "gpt-4o",
                "max_output_tokens": 4096,
                "reasoning": {"effort": "medium"},
                "text": {"verbosity": "high"},
                "temperature": 0.0,
                "top_p": 1.0,
            }
        }
        args = Namespace(
            model=None, max_output_tokens=None,
            reasoning_effort=None, verbosity=None,
            temperature=None, top_p=None,
        )

        effective = _build_effective_model_config(base, args)

        assert effective["extraction_model"]["name"] == "gpt-4o"
        assert effective["extraction_model"]["max_output_tokens"] == 4096
        assert effective["extraction_model"]["reasoning"]["effort"] == "medium"
        assert effective["extraction_model"]["text"]["verbosity"] == "high"
        assert effective["extraction_model"]["temperature"] == 0.0
        assert effective["extraction_model"]["top_p"] == 1.0
        assert effective is not base

    def test_cli_args_parser_context_flag(self):
        """Test --context argument parsing with auto, none, and file path."""
        from main.cli_args import create_process_parser

        parser = create_process_parser()

        args_auto = parser.parse_args([
            "--input", "test.txt", "--schema", "S", "--context", "auto",
        ])
        assert args_auto.context == "auto"

        args_none = parser.parse_args([
            "--input", "test.txt", "--schema", "S", "--context", "none",
        ])
        assert args_none.context == "none"

        args_path = parser.parse_args([
            "--input", "test.txt", "--schema", "S", "--context", "ctx/my_context.txt",
        ])
        assert args_path.context == "ctx/my_context.txt"

    def test_cli_args_parser_concurrency_flags(self):
        """Test --concurrency-limit and --delay argument parsing."""
        from main.cli_args import create_process_parser

        parser = create_process_parser()
        args = parser.parse_args([
            "--input", "test_input.txt",
            "--schema", "TestSchema",
            "--concurrency-limit", "10",
            "--delay", "2.5",
        ])

        assert args.concurrency_limit == 10
        assert args.delay == 2.5

    def test_build_effective_concurrency_config_applies_overrides(self):
        """CLI concurrency flags should override config without mutating original."""
        from main.process_text_files import _build_effective_concurrency_config

        base = {
            "concurrency": {
                "extraction": {
                    "concurrency_limit": 20,
                    "delay_between_tasks": 0.0,
                    "service_tier": "auto",
                }
            }
        }
        args = Namespace(concurrency_limit=5, delay=1.5)

        effective = _build_effective_concurrency_config(base, args)

        assert effective["concurrency"]["extraction"]["concurrency_limit"] == 5
        assert effective["concurrency"]["extraction"]["delay_between_tasks"] == 1.5
        assert effective["concurrency"]["extraction"]["service_tier"] == "auto"
        assert base["concurrency"]["extraction"]["concurrency_limit"] == 20
        assert base["concurrency"]["extraction"]["delay_between_tasks"] == 0.0

    def test_build_effective_concurrency_config_preserves_defaults(self):
        """Without CLI flags, concurrency config should be unchanged."""
        from main.process_text_files import _build_effective_concurrency_config

        base = {
            "concurrency": {
                "extraction": {
                    "concurrency_limit": 20,
                    "delay_between_tasks": 0.0,
                }
            }
        }
        args = Namespace(concurrency_limit=None, delay=None)

        effective = _build_effective_concurrency_config(base, args)

        assert effective["concurrency"]["extraction"]["concurrency_limit"] == 20
        assert effective["concurrency"]["extraction"]["delay_between_tasks"] == 0.0
        assert effective is not base

    @pytest.mark.asyncio
    async def test_process_script_run_cli_forwards_parsed_args(self):
        """ProcessTextFilesScript.run_cli should forward framework-parsed args."""
        from main.process_text_files import ProcessTextFilesScript

        script = ProcessTextFilesScript()
        cli_args = Namespace(schema="TestSchema", input="test_input.txt")

        with patch("main.process_text_files._run_cli_mode", new_callable=AsyncMock) as mock_run:
            await script.run_cli(cli_args)

        assert mock_run.await_count == 1
        assert mock_run.await_args.args[0] is cli_args

    def test_mode_detector_detects_cli_args(self, config_loader, monkeypatch):
        """Test mode detector identifies CLI mode when args provided."""
        from main.mode_detector import detect_execution_mode

        monkeypatch.setattr(sys, "argv", ["script", "--input", "file.txt"])

        is_interactive = detect_execution_mode(config_loader)
        assert is_interactive is False

    def test_mode_detector_interactive_mode(self, config_loader, monkeypatch):
        """Test mode detector respects config when no args."""
        from main.mode_detector import detect_execution_mode

        monkeypatch.setattr(sys, "argv", ["script"])

        # Config fixture has interactive_mode: False
        is_interactive = detect_execution_mode(config_loader)
        assert is_interactive is False


@pytest.mark.integration
class TestGenerateLineRangesCLI:
    """Integration tests for generate_line_ranges.py CLI."""

    def test_cli_args_parser(self):
        """Test CLI argument parsing for generate_line_ranges."""
        from main.cli_args import create_generate_ranges_parser

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
        from main.cli_args import create_check_batches_parser

        parser = create_check_batches_parser()
        args = parser.parse_args([])

        # Parser should work without any args
        assert hasattr(args, "schema") or hasattr(args, "input")

    def test_cli_args_parser_with_schema(self):
        """Test CLI argument parsing with schema."""
        from main.cli_args import create_check_batches_parser

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
        from main.cli_args import create_cancel_batches_parser

        parser = create_cancel_batches_parser()
        args = parser.parse_args([])

        # Parser should work without any args
        assert hasattr(args, "force")

    def test_cli_args_parser_force(self):
        """Test CLI argument parsing with force flag."""
        from main.cli_args import create_cancel_batches_parser

        parser = create_cancel_batches_parser()
        args = parser.parse_args(["--force"])

        assert args.force is True


@pytest.mark.integration
class TestRepairExtractionsCLI:
    """Integration tests for repair_extractions.py CLI."""

    def test_cli_args_parser(self):
        """Test CLI argument parsing for repair_extractions."""
        from main.cli_args import create_repair_parser

        parser = create_repair_parser()
        args = parser.parse_args([
            "--schema", "TestSchema",
        ])

        assert args.schema == "TestSchema"

    def test_cli_args_parser_with_files(self):
        """Test CLI argument parsing with specific files."""
        from main.cli_args import create_repair_parser

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
        from main.cli_args import create_adjust_ranges_parser

        parser = create_adjust_ranges_parser()
        args = parser.parse_args([
            "--input", "test_file.txt",
            "--schema", "TestSchema",
        ])

        assert args.input == "test_file.txt"
        assert args.schema == "TestSchema"

    def test_cli_args_parser_with_context_window(self):
        """Test CLI argument parsing with context-window option."""
        from main.cli_args import create_adjust_ranges_parser

        parser = create_adjust_ranges_parser()
        args = parser.parse_args([
            "--input", "test_file.txt",
            "--schema", "TestSchema",
            "--context-window", "15",
        ])

        assert args.context_window == 15

    def test_cli_args_parser_with_model_override(self):
        """Test CLI argument parsing with --model flag."""
        from main.cli_args import create_adjust_ranges_parser

        parser = create_adjust_ranges_parser()
        args = parser.parse_args([
            "--input", "test_file.txt",
            "--schema", "TestSchema",
            "--model", "claude-haiku-4-5",
        ])

        assert args.model == "claude-haiku-4-5"

    def test_cli_args_parser_with_reasoning_effort(self):
        """Test CLI argument parsing with --reasoning-effort flag."""
        from main.cli_args import create_adjust_ranges_parser

        parser = create_adjust_ranges_parser()
        args = parser.parse_args([
            "--input", "test_file.txt",
            "--schema", "TestSchema",
            "--reasoning-effort", "low",
        ])

        assert args.reasoning_effort == "low"

    def test_cli_args_parser_with_all_model_overrides(self):
        """Test CLI argument parsing with all model override flags."""
        from main.cli_args import create_adjust_ranges_parser

        parser = create_adjust_ranges_parser()
        args = parser.parse_args([
            "--input", "test_file.txt",
            "--schema", "TestSchema",
            "--model", "gpt-5.4-nano",
            "--reasoning-effort", "medium",
            "--max-output-tokens", "4096",
            "--temperature", "0.5",
            "--top-p", "0.9",
        ])

        assert args.model == "gpt-5.4-nano"
        assert args.reasoning_effort == "medium"
        assert args.max_output_tokens == 4096
        assert args.temperature == 0.5
        assert args.top_p == 0.9

    def test_cli_args_model_defaults_to_none(self):
        """Model flags default to None when not provided."""
        from main.cli_args import create_adjust_ranges_parser

        parser = create_adjust_ranges_parser()
        args = parser.parse_args([
            "--input", "test_file.txt",
            "--schema", "TestSchema",
        ])

        assert args.model is None
        assert args.reasoning_effort is None
        assert args.max_output_tokens is None
        assert args.temperature is None
        assert args.top_p is None


@pytest.mark.integration
class TestExampleFilesDiscovery:
    """Test file discovery using example_files directory structure."""

    def test_discover_files_from_directory(self, tmp_path: Path):
        """Test discovering text files from a directory."""
        from main.cli_args import get_files_from_path

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
        from main.cli_args import get_files_from_path

        single_file = tmp_path / "single.txt"
        single_file.write_text("content", encoding="utf-8")

        files = get_files_from_path(single_file)

        assert len(files) == 1
        assert files[0] == single_file


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
