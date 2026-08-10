# main/generate_line_ranges.py

"""
Script to generate line ranges for text files.

This script selects a schema, reads a text file (or multiple files), and
generates line ranges based on token-based chunking. The line ranges are
written to a '_line_ranges.txt' file.

Supports two execution modes:
1. Interactive Mode: User-friendly prompts
2. CLI Mode: Command-line arguments for automation

The generation workflow itself (file selection, chunking, sidecar
writing) lives in :mod:`modules.line_ranges`; this script only wires it
to the dual-mode CLI framework.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from argparse import ArgumentParser, Namespace
from typing import Any

from main.bootstrap import validate_schema_paths
from main.cli_args import (
    DEFAULT_EXCLUDE_PATTERNS,
    create_generate_ranges_parser,
    get_files_from_path,
    resolve_path,
)
from main.dual_mode import DualModeScript
from modules.config.schema_manager import SchemaManager
from modules.infra.chunking import ChunkSlice
from modules.line_ranges import process_files, select_input_files


class GenerateLineRangesScript(DualModeScript):
    """Script to generate line ranges for text files based on token chunking."""

    def __init__(self) -> None:
        super().__init__("generate_line_ranges")
        self.model_name: str | None = None
        self.tokens_per_chunk: int | None = None

    def create_argument_parser(self) -> ArgumentParser:
        """Create argument parser for CLI mode."""
        return create_generate_ranges_parser()

    def _get_model_config(self) -> tuple[str, int]:
        """Get model name and tokens per chunk from configuration."""
        chunking_config = self.chunking_and_context_config.get("chunking", {})
        model_cfg = self.model_config.get("extraction_model", {})
        model_name = model_cfg.get("name", "o3-mini")
        tokens_per_chunk = chunking_config.get("default_tokens_per_chunk", 7500)
        return model_name, tokens_per_chunk

    def _select_schema(self) -> str:
        """Prompt user to select a schema."""
        assert self.ui is not None
        schema_manager = SchemaManager()
        schema_manager.load_schemas()
        available_schemas = schema_manager.get_available_schemas()

        if not available_schemas:
            self.ui.print_error(
                "No schemas available. Please add schemas to the 'schemas/' folder."
            )
            sys.exit(1)

        result = self.ui.select_schema(schema_manager)
        if result is None:
            self.ui.print_info("Schema selection cancelled.")
            sys.exit(0)

        _, selected_schema_name = result
        return selected_schema_name

    def _get_input_directory(self, schema_name: str) -> Path:
        """Get input directory for the selected schema."""
        # Assumes validation has been done beforehand
        return Path(self.schemas_paths[schema_name].get("input", ""))

    def _process_files(
        self,
        files: list[Path],
        verbose: bool = False,
        chunk_slice: ChunkSlice | None = None,
    ) -> tuple[int, int]:
        """Delegate per-file range generation to the line-ranges package."""
        assert self.tokens_per_chunk is not None
        assert self.model_name is not None
        return process_files(
            files,
            tokens_per_chunk=self.tokens_per_chunk,
            model_name=self.model_name,
            logger=self.logger,
            ui=self.ui,
            verbose=verbose,
            chunk_slice=chunk_slice,
        )

    def run_interactive(self) -> None:
        """Run line range generation in interactive mode with back navigation."""
        assert self.ui is not None
        self.ui.print_section_header("Line Range Generation")
        self.ui.print_info("Loading configuration...")

        # Get model configuration
        self.model_name, self.tokens_per_chunk = self._get_model_config()

        # State machine for navigation
        # States: schema -> files -> chunk_slice -> confirm
        current_step = "schema"
        state: dict[str, Any] = {}

        while True:
            if current_step == "schema":
                selected_schema_name = self._select_schema()
                state["selected_schema_name"] = selected_schema_name

                # Validate schema has paths configured
                if not validate_schema_paths(
                    selected_schema_name, self.schemas_paths, self.ui
                ):
                    self.logger.error(
                        "Exiting: No path configuration for schema "
                        f"'{selected_schema_name}'"
                    )
                    sys.exit(1)

                state["raw_text_dir"] = self._get_input_directory(selected_schema_name)
                current_step = "files"

            elif current_step == "files":
                files = select_input_files(
                    self.ui, state["raw_text_dir"], allow_back=True
                )
                if files is None:
                    current_step = "schema"
                    continue
                state["files"] = files
                current_step = "chunk_slice"

            elif current_step == "chunk_slice":
                chunk_slice = self.ui.ask_chunk_slice(allow_back=True)
                if chunk_slice is None:
                    current_step = "files"
                    continue
                state["chunk_slice"] = chunk_slice
                current_step = "confirm"

            elif current_step == "confirm":
                if not self.ui.confirm(
                    f"Generate line ranges for {len(state['files'])} file(s)?",
                    default=True,
                ):
                    self.ui.print_info("Operation cancelled by user.")
                    return
                # Break out of loop to start processing
                break

        # Process files
        self.ui.print_section_header("Generating Line Ranges")
        success_count, fail_count = self._process_files(
            state["files"], verbose=False, chunk_slice=state.get("chunk_slice")
        )

        # Final summary
        self.ui.print_section_header("Generation Complete")
        self.ui.print_success(
            f"Successfully generated line ranges for {success_count} file(s)"
        )
        if fail_count > 0:
            self.ui.print_warning(f"Failed to process {fail_count} file(s)")

    def run_cli(self, args: Namespace) -> None:
        """Run line range generation in CLI mode."""
        self.logger.info("Starting line range generation (CLI Mode)")

        # Get model configuration
        self.model_name, default_tokens = self._get_model_config()
        self.tokens_per_chunk = args.tokens if args.tokens else default_tokens

        # Resolve input path
        input_path = resolve_path(args.input)
        if not input_path.exists():
            self.logger.error(f"Input path does not exist: {input_path}")
            print(f"[ERROR] Input path not found: {input_path}")
            sys.exit(1)

        # Get files. Collect both .txt and .md (excluding the tool's own
        # sidecar/report files), matching the other entry points.
        exclude_patterns = list(DEFAULT_EXCLUDE_PATTERNS)
        if input_path.is_file():
            files = get_files_from_path(input_path, exclude_patterns=exclude_patterns)
        else:
            seen: dict[Path, None] = {}
            for pattern in ("*.txt", "*.md"):
                for found in get_files_from_path(
                    input_path, pattern=pattern, exclude_patterns=exclude_patterns
                ):
                    seen[found] = None
            files = sorted(seen)

        if not files:
            self.logger.error(f"No text files found at: {input_path}")
            print(f"[ERROR] No text files found at: {input_path}")
            sys.exit(1)

        self.logger.info(f"Found {len(files)} file(s) to process")
        if args.verbose:
            print(
                f"[INFO] Processing {len(files)} file(s) "
                f"with {self.tokens_per_chunk} tokens per chunk"
            )

        # Build chunk slice from CLI args
        chunk_slice = None
        first_n = getattr(args, "first_n_chunks", None)
        last_n = getattr(args, "last_n_chunks", None)
        if first_n is not None:
            chunk_slice = ChunkSlice(first_n=first_n)
        elif last_n is not None:
            chunk_slice = ChunkSlice(last_n=last_n)

        # Process files
        success_count, fail_count = self._process_files(
            files, verbose=args.verbose, chunk_slice=chunk_slice
        )

        # Final summary
        self.logger.info(
            f"Generation complete: {success_count} succeeded, {fail_count} failed"
        )
        print(
            f"[SUCCESS] Generated line ranges for {success_count}/{len(files)} file(s)"
        )
        if fail_count > 0:
            print(f"[WARNING] {fail_count} file(s) failed")
            sys.exit(1)


def main() -> None:
    """Main entry point."""
    script = GenerateLineRangesScript()
    script.execute()


if __name__ == "__main__":
    main()
