# modules/operations/extraction/file_processor.py

"""
File processor for schema-based structured data extraction.
Uses modular components with simplified orchestration and separated concerns.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from modules.core.chunking_service import ChunkSlice, ChunkingService, apply_chunk_slice
from modules.core.processing_strategy import create_processing_strategy
from modules.core.context_resolver import resolve_context_for_extraction
from modules.core.resume import (
    FileStatus,
    build_extraction_metadata,
    detect_extraction_status,
    get_output_json_path,
    _METADATA_KEY,
)
from modules.core.text_utils import TextProcessor
from modules.core.path_utils import ensure_path_safe
from modules.llm.prompt_utils import render_prompt_with_schema
from modules.operations.extraction.schema_handlers import get_schema_handler
from modules.ui import print_info, print_success, print_warning, print_error, ui_print

logger = logging.getLogger(__name__)


class _MessagingAdapter:
    """Simple messaging adapter for file processor output."""
    
    def __init__(self, ui: Any = None) -> None:
        self.ui = ui
    
    def info(self, message: str, log: bool = True) -> None:
        if self.ui:
            self.ui.print_info(message)
        else:
            print_info(message)
        if log:
            logger.info(message)
    
    def success(self, message: str, log: bool = True) -> None:
        if self.ui:
            self.ui.print_success(message)
        else:
            print_success(message)
        if log:
            logger.info(f"SUCCESS: {message}")
    
    def warning(self, message: str, log: bool = True) -> None:
        if self.ui:
            self.ui.print_warning(message)
        else:
            print_warning(message)
        if log:
            logger.warning(message)
    
    def error(self, message: str, log: bool = True, exc_info: Any = None) -> None:
        if self.ui:
            self.ui.print_error(message)
        else:
            print_error(message)
        if log:
            if exc_info:
                logger.error(message, exc_info=exc_info)
            else:
                logger.error(message)
    
    def console_print(self, message: str) -> None:
        if self.ui and hasattr(self.ui, 'console_print'):
            self.ui.console_print(message)
        else:
            ui_print(message)


def _create_messaging_adapter(ui: Any = None) -> _MessagingAdapter:
    """Factory function to create messaging adapter."""
    return _MessagingAdapter(ui)


class FileProcessorRefactored:
    """
    Refactored file processor with modular architecture.
    Orchestrates text processing, chunking, and API interactions.
    """

    def __init__(
        self,
        paths_config: Dict[str, Any],
        model_config: Dict[str, Any],
        chunking_config: Dict[str, Any],
        concurrency_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize file processor.

        :param paths_config: Paths configuration
        :param model_config: Model configuration
        :param chunking_config: Chunking configuration
        :param concurrency_config: Concurrency configuration
        """
        self.paths_config = paths_config
        self.model_config = model_config
        self.chunking_config = chunking_config
        self.concurrency_config = concurrency_config or {}
        
        self.text_processor = TextProcessor()
        
        # Initialize chunking service
        chunking_settings = chunking_config.get("chunking", {})
        self.chunking_service = ChunkingService(
            model_name=model_config["transcription_model"]["name"],
            default_tokens_per_chunk=chunking_settings.get("default_tokens_per_chunk", 7500),
            text_processor=self.text_processor
        )

    async def process_file(
        self,
        file_path: Path,
        use_batch: bool,
        selected_schema: Dict[str, Any],
        prompt_template: str,
        schema_name: str,
        inject_schema: bool,
        schema_paths: Dict[str, Any],
        global_chunking_method: Optional[str] = None,
        ui: Any = None,
        resume: bool = False,
        chunk_slice: Optional[ChunkSlice] = None,
        context_override: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Process a single text file with refactored architecture.

        :param file_path: Path to the file to process
        :param use_batch: Whether to use batch processing
        :param selected_schema: The selected schema dictionary
        :param prompt_template: Base system prompt template text
        :param schema_name: Name of the selected schema
        :param inject_schema: Whether to inject the JSON schema into the system prompt
        :param schema_paths: Schema-specific paths
        :param global_chunking_method: Global chunking method if specified
        :param ui: UserInterface instance for user feedback
        :param resume: If True, skip completed chunks and resume partial outputs
        :param context_override: Optional dict with 'mode' ('auto'|'none'|'manual') and 'path' (CM-8)
        """
        # Create messaging adapter
        messenger = _create_messaging_adapter(ui)
        
        messenger.info(f"Processing file: {file_path.name}")
        logger.info(f"Starting processing for file: {file_path}")

        # Read and normalize text
        # OpenAI API requires UTF-8, so try UTF-8 first, then fallback to detection
        try:
            try:
                with file_path.open("r", encoding="utf-8") as f:
                    lines = f.readlines()
                logger.info(f"Successfully read file {file_path.name} using UTF-8 encoding")
            except UnicodeDecodeError:
                # Fallback for non-UTF-8 files
                messenger.warning(f"File {file_path.name} is not UTF-8, attempting encoding detection...")
                logger.warning(f"UTF-8 decode failed for {file_path.name}, using chardet detection")
                encoding = TextProcessor.detect_encoding(file_path)
                messenger.info(f"Detected encoding: {encoding}")
                with file_path.open("r", encoding=encoding) as f:
                    lines = f.readlines()
            
            normalized_lines = [TextProcessor.normalize_text(line) for line in lines]
            messenger.info(f"Successfully read and normalized {len(lines)} lines from {file_path.name}")
        except Exception as e:
            messenger.error(f"Failed to read file {file_path.name}: {e}", exc_info=e)
            return

        # Determine chunking strategy
        chunking_method = self._determine_chunking_method(
            file_path, global_chunking_method, messenger, ui
        )

        # Perform text chunking
        try:
            line_ranges_file = file_path.with_name(f"{file_path.stem}_line_ranges.txt")
            chunks, ranges = self.chunking_service.chunk_text(
                lines=normalized_lines,
                strategy=chunking_method,
                line_ranges_file=line_ranges_file if line_ranges_file.exists() else None,
                original_start_line=1
            )
            messenger.info(f"Generated {len(chunks)} text chunks from {file_path.name}")
            logger.info(f"Total chunks generated from {file_path.name}: {len(chunks)}")
        except Exception as e:
            messenger.error(f"Failed to chunk text from {file_path.name}: {e}", exc_info=e)
            return

        # Apply chunk slice (first/last N) if requested
        if chunk_slice is not None and (chunk_slice.first_n is not None or chunk_slice.last_n is not None):
            original_count = len(chunks)
            chunks, ranges = apply_chunk_slice(chunks, ranges, chunk_slice)
            messenger.info(
                f"Chunk slice applied: processing {len(chunks)}/{original_count} chunks"
            )

        # Resolve context — honour context_override if provided (CM-8)
        context: Optional[str]
        context_path: Optional[Path]
        override_mode = (context_override or {}).get("mode", "auto")
        if override_mode == "none":
            context, context_path = None, None
            logger.debug(f"Context disabled for '{file_path.name}'")
        elif override_mode == "manual" and (context_override or {}).get("path"):
            manual_path = Path(context_override["path"])  # type: ignore[arg-type]
            if manual_path.exists():
                context = manual_path.read_text(encoding="utf-8").strip()
                context_path = manual_path
                logger.info(f"Using manual context from: {manual_path}")
                messenger.info(f"Using context from: {manual_path.name}")
            else:
                logger.warning(f"Manual context path not found: {manual_path}; falling back to auto")
                context, context_path = resolve_context_for_extraction(text_file=file_path)
        else:
            context, context_path = resolve_context_for_extraction(text_file=file_path)

        if context_path and override_mode == "auto":
            logger.info(f"Using extraction context from: {context_path}")
            messenger.info(f"Using context from: {context_path.name}")
        elif not context_path and override_mode == "auto":
            logger.debug(f"No extraction context found for '{file_path.name}'")

        # Render system prompt
        schema_definition = selected_schema.get("schema", {})
        effective_dev_message = render_prompt_with_schema(
            prompt_template,
            schema_definition,
            schema_name=schema_name,
            inject_schema=inject_schema,
            context=context,
        )

        # Get schema handler
        try:
            handler = get_schema_handler(schema_name)
        except Exception as e:
            messenger.error(f"Failed to get schema handler: {e}", exc_info=e)
            return

        # Determine output paths
        try:
            working_folder, output_json_path, temp_jsonl_path = self._setup_output_paths(
                file_path, schema_paths
            )
            messenger.info(f"Output will be saved to: {output_json_path}")
        except Exception as e:
            messenger.error(f"Failed to set up output paths: {e}", exc_info=e)
            return

        # Resume: detect already-completed chunks
        completed_chunk_indices: set[int] = set()
        if resume and not use_batch:
            status, completed_chunk_indices = detect_extraction_status(
                output_json_path, expected_chunks=len(chunks)
            )
            if status == FileStatus.COMPLETE:
                messenger.info(f"Skipping {file_path.name}: already fully processed ({len(completed_chunk_indices)} chunks)")
                return
            if status == FileStatus.PARTIAL:
                messenger.info(
                    f"Resuming {file_path.name}: {len(completed_chunk_indices)}/{len(chunks)} chunks already done"
                )

        # Create processing strategy and execute
        strategy = create_processing_strategy(use_batch, self.concurrency_config)

        processing_cancelled = False
        processing_exception: Optional[Exception] = None

        try:
            await strategy.process_chunks(
                chunks=chunks,
                handler=handler,
                dev_message=effective_dev_message,
                model_config=self.model_config,
                schema=selected_schema["schema"],
                file_path=file_path,
                temp_jsonl_path=temp_jsonl_path,
                console_print=messenger.console_print,
                completed_chunk_indices=completed_chunk_indices,
            )
        except asyncio.CancelledError:
            processing_cancelled = True
            messenger.warning(
                "Processing interrupted by user. Attempting to persist completed chunks before exit..."
            )
            raise
        except Exception as e:
            processing_exception = e
            messenger.error(f"Error during processing: {e}", exc_info=e)
        finally:
            wrote_output = False
            if not use_batch:
                try:
                    if temp_jsonl_path.exists() and temp_jsonl_path.stat().st_size > 0:
                        # Build chunk slice metadata if a slice was applied
                        _cs_info: Optional[dict] = None
                        if chunk_slice is not None and (chunk_slice.first_n is not None or chunk_slice.last_n is not None):
                            _cs_info = {}
                            if chunk_slice.first_n is not None:
                                _cs_info["first_n"] = chunk_slice.first_n
                            if chunk_slice.last_n is not None:
                                _cs_info["last_n"] = chunk_slice.last_n
                        await asyncio.shield(
                            self._generate_output_files(
                                temp_jsonl_path,
                                output_json_path,
                                handler,
                                schema_paths,
                                messenger,
                                partial=processing_cancelled or processing_exception is not None,
                                chunk_slice_info=_cs_info,
                            )
                        )
                        wrote_output = True
                    elif processing_cancelled:
                        messenger.warning(
                            "Processing was interrupted before any chunks completed; no output file generated."
                        )
                except Exception as gen_exc:
                    messenger.error(f"Failed to write final output: {gen_exc}", exc_info=gen_exc)

            self._cleanup_temp_files(use_batch, temp_jsonl_path, messenger)

            if processing_cancelled:
                if wrote_output:
                    messenger.warning(
                        f"Processing cancelled. Partial results saved to {output_json_path}"
                    )
                # Allow cancellation to propagate after cleanup and messaging
            elif processing_exception is None:
                messenger.success(f"Completed processing of file: {file_path.name}")

        if processing_exception is not None:
            return

    def _determine_chunking_method(
        self,
        file_path: Path,
        global_chunking_method: Optional[str],
        messenger: _MessagingAdapter,
        ui: Any
    ) -> str:
        """Determine which chunking method to use."""
        if global_chunking_method == "per-file":
            global_chunking_method = None

        if global_chunking_method is not None:
            messenger.info(f"Using global chunking method '{global_chunking_method}' for file {file_path.name}")
            return global_chunking_method
        else:
            # Ask for per-file chunking method
            if ui and hasattr(ui, 'ask_file_chunking_method'):
                return ui.ask_file_chunking_method(file_path.name)
            else:
                return self._default_ask_file_chunking_method(file_path.name)


    def _setup_output_paths(
        self,
        file_path: Path,
        schema_paths: Dict[str, Any]
    ) -> tuple[Path, Path, Path]:
        """Set up output directory paths."""
        if self.paths_config["general"]["input_paths_is_output_path"]:
            working_folder = ensure_path_safe(file_path.parent)
            output_json_path = ensure_path_safe(working_folder / f"{file_path.stem}_output.json")
            temp_jsonl_path = ensure_path_safe(working_folder / f"{file_path.stem}_temp.jsonl")
            working_folder.mkdir(parents=True, exist_ok=True)
        else:
            output_path_str = schema_paths.get("output", "")
            # CM-7: Validate that output path is not empty or CWD
            if not output_path_str or not str(output_path_str).strip():
                raise ValueError(
                    "Output path is not configured. Set 'output' in paths_config.yaml for this schema, "
                    "or enable 'input_paths_is_output_path: true' in general settings."
                )
            working_folder = ensure_path_safe(Path(output_path_str))
            if working_folder.resolve() == Path.cwd().resolve():
                raise ValueError(
                    f"Output path '{working_folder}' resolves to the current working directory. "
                    "Configure a specific output directory to avoid mixing output with project files."
                )
            temp_folder = ensure_path_safe(working_folder / "temp_jsonl")
            working_folder.mkdir(parents=True, exist_ok=True)
            temp_folder.mkdir(parents=True, exist_ok=True)
            output_json_path = ensure_path_safe(working_folder / f"{file_path.stem}_output.json")
            temp_jsonl_path = ensure_path_safe(temp_folder / f"{file_path.stem}_temp.jsonl")

        return working_folder, output_json_path, temp_jsonl_path

    async def _generate_output_files(
        self,
        temp_jsonl_path: Path,
        output_json_path: Path,
        handler: Any,
        schema_paths: Dict[str, Any],
        messenger: _MessagingAdapter,
        *,
        partial: bool = False,
        chunk_slice_info: Optional[dict] = None,
    ) -> None:
        """Generate final output files from temporary JSONL."""
        try:
            messenger.info("Constructing final output from temporary file...")
            results = []
            
            if temp_jsonl_path.exists():
                with temp_jsonl_path.open("r", encoding="utf-8") as tempf:
                    for line in tempf:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                            if "custom_id" in record:
                                response_field = record.get("response", {}).get("body", {})
                                output_record = {
                                    "custom_id": record.get("custom_id"),
                                    "chunk_index": record.get("chunk_index"),
                                    "chunk_range": record.get("chunk_range"),
                                    "response": response_field,
                                }
                                results.append(output_record)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse line in temp file: {e}")
                            continue
            
            # Sort results by chunk_index, handling None values
            results.sort(key=lambda x: x.get("chunk_index") or 0)
            
            # Build output structure with metadata
            chunking_method = "unknown"
            model_name = self.model_config.get("transcription_model", {}).get("name", "unknown")
            schema_name_for_meta = getattr(handler, "schema_name", "unknown")
            output_data = {
                _METADATA_KEY: build_extraction_metadata(
                    schema_name=schema_name_for_meta,
                    model_name=model_name,
                    chunking_method=chunking_method,
                    total_chunks=len(results),
                    chunk_slice_info=chunk_slice_info,
                ),
                "records": results,
            }

            with output_json_path.open("w", encoding="utf-8") as outf:
                json.dump(output_data, outf, indent=2)

            if partial:
                messenger.warning(
                    f"Partial structured JSON output saved to {output_json_path}"
                )
            else:
                messenger.success(f"Final structured JSON output saved to {output_json_path}")
        except Exception as e:
            messenger.error(f"Failed to write final output: {e}", exc_info=e)
            return

        # Generate additional formats
        self._generate_additional_formats(output_json_path, handler, schema_paths, messenger)

    def _generate_additional_formats(
        self,
        output_json_path: Path,
        handler: Any,
        schema_paths: Dict[str, Any],
        messenger: _MessagingAdapter
    ) -> None:
        """Generate CSV, DOCX, and TXT outputs if configured."""
        if schema_paths.get("csv_output", False):
            try:
                output_csv_path = output_json_path.with_suffix(".csv")
                handler.convert_to_csv(output_json_path, output_csv_path)
                messenger.info(f"CSV output saved to {output_csv_path}")
            except Exception as e:
                messenger.warning(f"Failed to generate CSV output: {e}")

        if schema_paths.get("docx_output", False):
            try:
                output_docx_path = output_json_path.with_suffix(".docx")
                handler.convert_to_docx(output_json_path, output_docx_path)
                messenger.info(f"DOCX output saved to {output_docx_path}")
            except Exception as e:
                messenger.warning(f"Failed to generate DOCX output: {e}")

        if schema_paths.get("txt_output", False):
            try:
                output_txt_path = output_json_path.with_suffix(".txt")
                handler.convert_to_txt(output_json_path, output_txt_path)
                messenger.info(f"TXT output saved to {output_txt_path}")
            except Exception as e:
                messenger.warning(f"Failed to generate TXT output: {e}")

    def _cleanup_temp_files(
        self,
        use_batch: bool,
        temp_jsonl_path: Path,
        messenger: _MessagingAdapter
    ) -> None:
        """Clean up temporary files if not needed."""
        if use_batch:
            logger.info(f"Batch processing enabled. Keeping temporary JSONL")
            messenger.info(f"Preserving {temp_jsonl_path.name} for batch tracking")
        else:
            keep_temp = self.paths_config["general"].get("retain_temporary_jsonl", True)
            if not keep_temp:
                try:
                    temp_jsonl_path.unlink()
                    logger.info(f"Deleted temporary file: {temp_jsonl_path}")
                    messenger.info(f"Deleted temporary file: {temp_jsonl_path.name}")
                except Exception as e:
                    messenger.error(f"Could not delete temporary file {temp_jsonl_path.name}: {e}")

    def _default_ask_file_chunking_method(self, file_name: str) -> str:
        """Default implementation if UI not provided."""
        print(f"\nSelect chunking method for file '{file_name}':")
        print("  1. Automatic chunking - Split text based on token limits with no intervention")
        print("  2. Interactive chunking - View default chunks and manually adjust boundaries")
        print("  3. Predefined chunks - Use saved boundaries from {file}_line_ranges.txt file")

        choice = input("Enter option (1-3): ").strip()
        if choice == "1":
            return "auto"
        elif choice == "2":
            return "auto-adjust"
        elif choice == "3":
            return "line_ranges.txt"
        else:
            print("Invalid selection, defaulting to automatic chunking.")
            return "auto"
