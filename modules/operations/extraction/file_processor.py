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

from modules.core.chunking_service import ChunkingService
from modules.core.processing_strategy import create_processing_strategy
from modules.core.context_resolver import resolve_context_for_extraction
from modules.core.text_utils import TextProcessor
from modules.core.path_utils import ensure_path_safe
from modules.llm.prompt_utils import render_prompt_with_schema
from modules.operations.extraction.schema_handlers import get_schema_handler
from modules.ui import print_info, print_success, print_warning, print_error, ui_print

logger = logging.getLogger(__name__)


class _MessagingAdapter:
    """Simple messaging adapter for file processor output."""
    
    def __init__(self, ui=None):
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
    
    def error(self, message: str, log: bool = True, exc_info=None) -> None:
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


def _create_messaging_adapter(ui=None) -> _MessagingAdapter:
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
        use_context: bool = True,
        context_source: str = "default",
        ui=None
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

        context = None
        context_path = None
        if use_context:
            global_context_dir = None
            if str(context_source).lower().strip() == "file":
                global_context_dir = Path(__file__).resolve().parents[3] / "__no_context_dir__"

            context, context_path = resolve_context_for_extraction(
                schema_name=schema_name,
                text_file=file_path,
                global_context_dir=global_context_dir,
            )

            if context_path:
                logger.info(f"Using extraction context from: {context_path}")
                messenger.info(f"Using context from: {context_path.name}")
            else:
                logger.info(f"No context found for schema '{schema_name}'")
        else:
            logger.info("Context disabled for this run")

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
                        await asyncio.shield(
                            self._generate_output_files(
                                temp_jsonl_path,
                                output_json_path,
                                handler,
                                schema_paths,
                                messenger,
                                partial=processing_cancelled or processing_exception is not None,
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
        ui
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
            working_folder = ensure_path_safe(Path(schema_paths["output"]))
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
        handler,
        schema_paths: Dict[str, Any],
        messenger: _MessagingAdapter,
        *,
        partial: bool = False,
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
            
            with output_json_path.open("w", encoding="utf-8") as outf:
                json.dump(results, outf, indent=2)

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
        handler,
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
