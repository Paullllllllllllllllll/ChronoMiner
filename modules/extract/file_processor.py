# modules/extract/file_processor.py

"""
File processor for schema-based structured data extraction.
Uses modular components with simplified orchestration and separated concerns.
"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

from modules.config.constants import (
    SUPPORTED_PDF_EXTENSIONS,
    SUPPORTED_VISUAL_EXTENSIONS,
)
from modules.config.context import resolve_context_for_extraction
from modules.conversion.json_utils import lean_response
from modules.extract.processing_strategy import create_processing_strategy
from modules.extract.resume import (
    CHUNKING_TEXT_VERSION,
    METADATA_KEY,
    TEMP_JSONL_VERSION,
    FileStatus,
    build_extraction_metadata,
    completed_indices_from_outputs,
    detect_extraction_status,
    is_resumable_temp_jsonl,
)
from modules.extract.schema_handlers import get_schema_handler
from modules.infra.chunking import (
    ChunkingService,
    ChunkSlice,
    TextProcessor,
    apply_chunk_slice,
    chunk_slice_indices,
)
from modules.infra.paths import ensure_path_safe
from modules.infra.token_tracker import check_and_wait_for_token_limit
from modules.llm.prompt_utils import (
    PROMPTS_DIR,
    load_prompt_template,
    render_prompt_with_schema,
)
from modules.ui import (
    print_error,
    print_info,
    print_success,
    print_warning,
    ui_input,
    ui_print,
)

# Backward compatibility alias for callers still importing the private name.
_METADATA_KEY = METADATA_KEY

logger = logging.getLogger(__name__)


def _completed_indices_from_temp(temp_jsonl_path: Path) -> set[int]:
    """Read the chunk/page indices already written to the live temp JSONL.

    The temp JSONL is the authoritative record of finished units within a run:
    every successfully processed chunk writes one line carrying its
    ``chunk_index``. Deferred (budget-skipped) and errored units write no line,
    so they are naturally excluded and re-attempted on the next pass.
    """
    done: set[int] = set()
    if not temp_jsonl_path.exists():
        return done
    try:
        with temp_jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                idx = record.get("chunk_index")
                if isinstance(idx, int):
                    done.add(idx)
    except OSError as exc:
        logger.warning(f"Could not read temp JSONL {temp_jsonl_path}: {exc}")
    return done


def is_visual_input(file_path: Path) -> bool:
    """Check if the path is a visual input (image, PDF, or directory of images)."""
    if file_path.is_file():
        return file_path.suffix.lower() in SUPPORTED_VISUAL_EXTENSIONS
    if file_path.is_dir():
        return any(
            f.suffix.lower() in SUPPORTED_VISUAL_EXTENSIONS
            for f in file_path.rglob("*")
            if f.is_file()
        )
    return False


class _MessagingAdapter:
    """Simple messaging adapter for file processor output.

    Always uses the module-level UI functions (print_info, print_success, etc.)
    for console output and the module-level logger for logging.
    """

    def info(self, message: str, log: bool = True) -> None:
        print_info(message)
        if log:
            logger.info(message)

    def success(self, message: str, log: bool = True) -> None:
        print_success(message)
        if log:
            logger.info(f"SUCCESS: {message}")

    def warning(self, message: str, log: bool = True) -> None:
        print_warning(message)
        if log:
            logger.warning(message)

    def error(self, message: str, log: bool = True, exc_info: Any = None) -> None:
        print_error(message)
        if log:
            if exc_info:
                logger.error(message, exc_info=exc_info)
            else:
                logger.error(message)

    def console_print(self, message: str) -> None:
        ui_print(message)


def _preprocess_context_image(
    image_path: Path,
    provider: str,
    model_name: str,
    image_detail: str,
) -> dict[str, Any]:
    """Preprocess and encode a context image for LLM injection.

    Uses the same ImageProcessor pipeline as the visual extraction path
    to normalize and resize the context image before base64 encoding.
    """
    from PIL import Image

    from modules.config.loader import get_config_loader
    from modules.images import ImageProcessor, encode_bytes_to_base64

    # Pass the FULL image config: ImageProcessor resolves its provider
    # section internally. (Passing the section here, as before, made it
    # look up the section name inside the section and fall back to
    # defaults.)
    image_config = get_config_loader().get_image_processing_config()

    processor = ImageProcessor(
        provider=provider,
        model_name=model_name,
        image_config=image_config,
    )
    with Image.open(image_path) as img:
        jpeg_bytes = processor.process_pil(img)
    return {
        "base64": encode_bytes_to_base64(jpeg_bytes, "image/jpeg"),
        "mime_type": "image/jpeg",
        "detail": image_detail,
    }


class FileProcessor:
    """
    Refactored file processor with modular architecture.
    Orchestrates text processing, chunking, and API interactions.
    """

    def __init__(
        self,
        paths_config: dict[str, Any],
        model_config: dict[str, Any],
        chunking_config: dict[str, Any],
        concurrency_config: dict[str, Any] | None = None,
        *,
        input_root: Path | None = None,
        output_mode: str = "flat",
    ):
        """
        Initialize file processor.

        :param paths_config: Paths configuration
        :param model_config: Model configuration
        :param chunking_config: Chunking configuration
        :param concurrency_config: Concurrency configuration
        :param input_root: Resolved input root for mirror mode
        :param output_mode: Output layout: "flat" or "mirror"
        """
        self.paths_config = paths_config
        self.input_root = input_root
        self.output_mode = output_mode
        self.model_config = model_config
        self.chunking_config = chunking_config
        self.concurrency_config = concurrency_config or {}

        self.text_processor = TextProcessor()

        # Initialize chunking service
        chunking_settings = chunking_config.get("chunking", {})
        self.chunking_service = ChunkingService(
            model_name=model_config["extraction_model"]["name"],
            default_tokens_per_chunk=chunking_settings.get(
                "default_tokens_per_chunk", 7500
            ),
            text_processor=self.text_processor,
        )

    @staticmethod
    def _is_visual_input(file_path: Path) -> bool:
        """Check if the path is a visual input; kept as a class method alias
        for backward compatibility. New code should use the module-level
        :func:`is_visual_input`.
        """
        return is_visual_input(file_path)

    async def process_file(
        self,
        file_path: Path,
        use_batch: bool,
        selected_schema: dict[str, Any],
        prompt_template: str,
        schema_name: str,
        inject_schema: bool,
        schema_paths: dict[str, Any],
        global_chunking_method: str | None = None,
        ui: Any = None,
        resume: bool = False,
        chunk_slice: ChunkSlice | None = None,
        context_override: dict[str, Any] | None = None,
        image_detail: str | None = None,
        context_image_enabled: bool = False,
    ) -> str:
        """
        Process a single file with refactored architecture.

        Routes to _process_visual_file() for image/PDF inputs, otherwise
        runs the existing text processing pipeline.

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
        :param context_override: Optional dict with 'mode' ('auto'|'none'|'manual')
            and 'path' (CM-8)
        :param image_detail: Image detail level for vision models
        """
        if self._is_visual_input(file_path):
            return await self._process_visual_file(
                file_path=file_path,
                use_batch=use_batch,
                selected_schema=selected_schema,
                schema_name=schema_name,
                inject_schema=inject_schema,
                schema_paths=schema_paths,
                ui=ui,
                resume=resume,
                chunk_slice=chunk_slice,
                context_override=context_override,
                image_detail=image_detail,
                context_image_enabled=context_image_enabled,
            )

        # Create messaging adapter
        messenger = _MessagingAdapter()

        messenger.info(f"Processing file: {file_path.name}")
        logger.info(f"Starting processing for file: {file_path}")

        # Read and normalize text
        # OpenAI API requires UTF-8, so try UTF-8 first, then fallback to detection
        try:
            try:
                with file_path.open("r", encoding="utf-8") as f:
                    lines = f.readlines()
                logger.info(
                    f"Successfully read file {file_path.name} using UTF-8 encoding"
                )
            except UnicodeDecodeError:
                # Fallback for non-UTF-8 files
                messenger.warning(
                    f"File {file_path.name} is not UTF-8,"
                    " attempting encoding detection..."
                )
                logger.warning(
                    f"UTF-8 decode failed for {file_path.name}, using chardet detection"
                )
                encoding = TextProcessor.detect_encoding(file_path)
                messenger.info(f"Detected encoding: {encoding}")
                with file_path.open("r", encoding=encoding) as f:
                    lines = f.readlines()

            # Strip only the trailing line terminator, preserving indentation
            # and interior whitespace. split_text_into_chunks re-joins these
            # lines with "\n"; stripping both sides (the old normalize_text)
            # and joining with "" merged words across line boundaries.
            normalized_lines = [line.rstrip("\n\r") for line in lines]
            messenger.info(
                f"Successfully read and normalized {len(lines)} lines"
                f" from {file_path.name}"
            )
        except Exception as e:
            messenger.error(f"Failed to read file {file_path.name}: {e}", exc_info=e)
            return "failed"

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
                line_ranges_file=line_ranges_file
                if line_ranges_file.exists()
                else None,
                original_start_line=1,
            )
            messenger.info(f"Generated {len(chunks)} text chunks from {file_path.name}")
            logger.info(f"Total chunks generated from {file_path.name}: {len(chunks)}")
        except Exception as e:
            messenger.error(
                f"Failed to chunk text from {file_path.name}: {e}", exc_info=e
            )
            return "failed"

        # Apply chunk slice (first/last N or page range) if requested. Carry
        # the absolute (document-space) chunk index of each retained chunk so
        # custom_ids and resume records are absolute rather than slice-relative
        # (mirrors the visual path's absolute page numbering).
        if chunk_slice is not None and (
            chunk_slice.first_n is not None
            or chunk_slice.last_n is not None
            or chunk_slice.page_range is not None
        ):
            original_count = len(chunks)
            chunk_indices = chunk_slice_indices(original_count, chunk_slice)
            chunks, ranges = apply_chunk_slice(chunks, ranges, chunk_slice)
            messenger.info(
                f"Chunk slice applied: processing {len(chunks)}/{original_count} chunks"
            )
        else:
            chunk_indices = list(range(1, len(chunks) + 1))

        # Delegate shared orchestration to _execute_extraction
        return await self._execute_extraction(
            file_path=file_path,
            chunks=chunks,
            prompt_template=prompt_template,
            selected_schema=selected_schema,
            schema_name=schema_name,
            inject_schema=inject_schema,
            schema_paths=schema_paths,
            use_batch=use_batch,
            messenger=messenger,
            resume=resume,
            chunk_slice=chunk_slice,
            context_override=context_override,
            context_image_enabled=context_image_enabled,
            chunk_indices=chunk_indices,
            chunk_ranges=ranges,
            ui=ui,
        )

    async def _process_visual_file(
        self,
        file_path: Path,
        use_batch: bool,
        selected_schema: dict[str, Any],
        schema_name: str,
        inject_schema: bool,
        schema_paths: dict[str, Any],
        ui: Any = None,
        resume: bool = False,
        chunk_slice: ChunkSlice | None = None,
        context_override: dict[str, Any] | None = None,
        image_detail: str | None = None,
        context_image_enabled: bool = False,
    ) -> str:
        """Process a visual input file (image or PDF) through the LLM vision
        pipeline.

        Pages are streamed: rendered, preprocessed, and encoded one at a
        time (``modules.images.page_stream``) instead of materializing the
        whole document in memory. The resume skip-set and any page slice
        are resolved BEFORE rendering, so already-completed and out-of-slice
        pages are never rendered at all.
        """
        from modules.config.capabilities import detect_capabilities
        from modules.images import (
            PagePayload,
            build_image_provenance,
            stream_page_payloads,
        )
        from modules.images.page_stream import resolve_image_section

        messenger = _MessagingAdapter()
        messenger.info(f"Processing visual file: {file_path.name}")
        logger.info(f"Starting visual processing for file: {file_path}")

        # 1. Validate vision support
        model_name = self.model_config["extraction_model"]["name"]
        caps = detect_capabilities(model_name)
        if not caps.supports_image_input:
            messenger.error(
                f"Model '{model_name}' does not support image inputs. "
                "Use a vision-capable model"
                " (e.g., gpt-5-mini, claude-sonnet-4-5, gemini-2.5-flash)."
            )
            return "failed"

        # 2. Determine provider and load image config
        from modules.config.loader import get_config_loader
        from modules.llm.langchain_provider import ProviderConfig

        provider = ProviderConfig._detect_provider(model_name)
        image_config = get_config_loader().get_image_processing_config()
        img_cfg = resolve_image_section(image_config, provider, model_name)

        # Determine effective detail level
        if image_detail is None:
            from modules.images import detect_model_type

            model_type = detect_model_type(provider, model_name)
            if model_type == "google":
                image_detail = img_cfg.get("media_resolution", "high") or "high"
            elif model_type == "anthropic":
                image_detail = img_cfg.get("resize_profile", "auto") or "auto"
            else:
                image_detail = img_cfg.get("llm_detail", "high") or "high"

        # 3. Page count without rendering
        ext = file_path.suffix.lower()
        if ext in SUPPORTED_PDF_EXTENSIONS:
            from modules.images import PDFProcessor

            try:
                with PDFProcessor(file_path) as pdf:
                    page_count = pdf.get_page_count()
            except Exception as e:
                messenger.error(f"Failed to open PDF {file_path.name}: {e}", exc_info=e)
                return "failed"
            target_dpi = int(image_config.get("target_dpi", 300))
            messenger.info(
                f"PDF has {page_count} page(s); streaming at {target_dpi} DPI"
            )
        else:
            page_count = 1

        # 4. Resume detection BEFORE rendering: completed pages are never
        # rendered again.
        completed: set[int] = set()
        if resume and not use_batch:
            try:
                output_json_path = self._setup_output_paths(file_path, schema_paths)[1]
            except Exception as e:
                messenger.error(f"Failed to set up output paths: {e}", exc_info=e)
                return "failed"
            status, completed = detect_extraction_status(
                output_json_path, expected_chunks=page_count
            )
            if status == FileStatus.COMPLETE:
                messenger.info(
                    f"Skipping {file_path.name}: already fully processed "
                    f"({len(completed)} pages)"
                )
                return "skipped"
            if status == FileStatus.PARTIAL:
                messenger.info(
                    f"Resuming {file_path.name}: {len(completed)}/{page_count}"
                    " pages already done"
                )

        # 5. Page slice BEFORE rendering. Page indices are absolute 1-based
        # page numbers of the source document throughout (custom_ids,
        # provenance, resume records).
        sliced_indices = list(range(1, page_count + 1))
        if chunk_slice is not None:
            if chunk_slice.first_n is not None:
                sliced_indices = sliced_indices[: chunk_slice.first_n]
            elif chunk_slice.last_n is not None:
                sliced_indices = sliced_indices[-chunk_slice.last_n :]
            elif chunk_slice.page_range is not None:
                start, end = chunk_slice.page_range
                sliced_indices = sliced_indices[
                    max(start - 1, 0) : min(end, page_count)
                ]
            if len(sliced_indices) != page_count:
                messenger.info(
                    f"Chunk slice applied: processing"
                    f" {len(sliced_indices)}/{page_count} page(s)"
                )

        needed_indices = [i for i in sliced_indices if i not in completed]
        if not needed_indices:
            messenger.info(
                f"Skipping {file_path.name}: all selected pages already processed"
            )
            return "skipped"

        # Load visual extraction prompt
        visual_prompt_path = PROMPTS_DIR / "image_extraction_prompt.txt"
        try:
            visual_prompt_template = load_prompt_template(visual_prompt_path)
        except FileNotFoundError:
            messenger.error(f"Visual extraction prompt not found: {visual_prompt_path}")
            return "failed"

        # 6. File-level provenance (source hash + preprocessing params)
        file_provenance = await asyncio.to_thread(
            build_image_provenance,
            file_path,
            image_config,
            provider,
            model_name,
            image_detail,
        )

        # 7. Factory for streaming page sources. Synchronous runs rebuild the
        # source over the still-pending pages on each budget re-pass, so the
        # one-shot async iterator is created lazily rather than once.
        def make_page_source(indices: list[int]) -> AsyncIterator[Any]:
            return stream_page_payloads(
                file_path=file_path,
                page_indices=indices,
                image_config=image_config,
                provider=provider,
                model_name=model_name,
                image_detail=image_detail,
            )

        # Placeholder chunks: len() provides the unit total for progress and
        # metadata (the selected page count, matching prior behavior).
        chunks = [""] * len(sliced_indices)

        image_chunks: list[dict[str, Any]] | None = None
        source_factory: Callable[[list[int]], AsyncIterator[Any]] | None = None
        if use_batch:
            # Batch submission needs the materialized request set; raw pages
            # are still rendered/freed one at a time by the producer.
            image_chunks = [
                payload.as_chunk()
                async for payload in make_page_source(needed_indices)
                if isinstance(payload, PagePayload)
            ]
            messenger.info(f"Preprocessed {len(image_chunks)} page(s) for batch")
        else:
            source_factory = make_page_source

        # Delegate shared orchestration to _execute_extraction
        return await self._execute_extraction(
            file_path=file_path,
            chunks=chunks,
            prompt_template=visual_prompt_template,
            selected_schema=selected_schema,
            schema_name=schema_name,
            inject_schema=inject_schema,
            schema_paths=schema_paths,
            use_batch=use_batch,
            messenger=messenger,
            resume=resume,
            chunk_slice=chunk_slice,
            context_override=context_override,
            image_chunks=image_chunks,
            source_factory=source_factory,
            needed_indices=needed_indices,
            unit_label="page",
            context_image_enabled=context_image_enabled,
            image_detail=image_detail,
            precomputed_completed=completed,
            image_provenance=file_provenance,
            # Batch mode materializes needed_indices into image_chunks; carry
            # those absolute page numbers so batch custom_ids match the sync
            # (streaming) path's absolute page numbering.
            chunk_indices=needed_indices if use_batch else None,
            ui=ui,
        )

    async def _execute_extraction(
        self,
        *,
        file_path: Path,
        chunks: list[str],
        prompt_template: str,
        selected_schema: dict[str, Any],
        schema_name: str,
        inject_schema: bool,
        schema_paths: dict[str, Any],
        use_batch: bool,
        messenger: _MessagingAdapter,
        resume: bool,
        chunk_slice: ChunkSlice | None,
        context_override: dict[str, Any] | None,
        image_chunks: list[dict[str, Any]] | None = None,
        source_factory: Callable[[list[int]], Any] | None = None,
        needed_indices: list[int] | None = None,
        unit_label: str = "chunk",
        context_image_enabled: bool = False,
        image_detail: str | None = None,
        precomputed_completed: set[int] | None = None,
        image_provenance: dict[str, Any] | None = None,
        chunk_indices: list[int] | None = None,
        chunk_ranges: list[tuple[int, int]] | None = None,
        ui: Any = None,
    ) -> str:
        """Shared extraction orchestration for both text and visual pipelines.

        Handles context resolution, prompt rendering, schema handler lookup,
        output path setup, resume detection, strategy execution, output
        generation, and cleanup.

        ``precomputed_completed`` carries a resume skip-set already resolved
        by the caller (visual path, where detection happens before
        rendering); when given, the in-method detection is bypassed.
        ``source_factory`` builds a fresh streaming page source over a given
        list of page indices for the synchronous visual path (rebuilt each
        pass so a budget-interrupted run can resume the still-pending pages);
        ``image_chunks`` is the materialized list used by batch mode.

        When the daily token limit is enabled, the synchronous strategy gates
        each chunk/page on the budget. If it is exhausted mid-file, this method
        drains in-flight work, waits for the daily reset (via
        ``check_and_wait_for_token_limit``), and re-passes over the pending
        units, reusing the temp-JSONL resume record.
        """
        # Resolve context — honour context_override if provided (CM-8)
        context: str | None
        context_path: Path | None
        override_mode = (context_override or {}).get("mode", "auto")
        if override_mode == "none":
            context, context_path = None, None
            logger.debug(f"Context disabled for '{file_path.name}'")
        elif override_mode == "manual" and (context_override or {}).get("path"):
            manual_path = Path(context_override["path"])  # type: ignore[index]
            if manual_path.exists():
                context = manual_path.read_text(encoding="utf-8").strip()
                context_path = manual_path
                logger.info(f"Using manual context from: {manual_path}")
                messenger.info(f"Using context from: {manual_path.name}")
            else:
                logger.warning(
                    f"Manual context path not found: {manual_path};"
                    " falling back to auto"
                )
                context, context_path = resolve_context_for_extraction(
                    text_file=file_path
                )
        else:
            context, context_path = resolve_context_for_extraction(text_file=file_path)

        if context_path and override_mode == "auto":
            logger.info(f"Using extraction context from: {context_path}")
            messenger.info(f"Using context from: {context_path.name}")
        elif not context_path and override_mode == "auto":
            logger.debug(f"No extraction context found for '{file_path.name}'")

        # Resolve context image if enabled
        context_image_data: dict[str, Any] | None = None
        if context_image_enabled and override_mode != "none":
            from modules.config.context import resolve_context_image_for_extraction

            ctx_img_path, _ = resolve_context_image_for_extraction(text_file=file_path)
            if ctx_img_path is not None:
                from modules.llm.langchain_provider import ProviderConfig

                model_name = self.model_config["extraction_model"]["name"]
                provider = ProviderConfig._detect_provider(model_name)
                effective_detail = image_detail or "high"
                context_image_data = _preprocess_context_image(
                    ctx_img_path, provider, model_name, effective_detail
                )
                messenger.info(f"Using context image: {ctx_img_path.name}")
            else:
                logger.debug(f"No context image found for '{file_path.name}'")

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
            return "failed"

        # Determine output paths
        try:
            working_folder, output_json_path, temp_jsonl_path = (
                self._setup_output_paths(file_path, schema_paths)
            )
            messenger.info(f"Output will be saved to: {output_json_path}")
        except Exception as e:
            messenger.error(f"Failed to set up output paths: {e}", exc_info=e)
            return "failed"

        # Resume: detect already-completed chunks/pages. The visual path
        # resolves this before rendering and passes it in; the text path
        # detects here.
        completed_chunk_indices: set[int] = set()
        if precomputed_completed is not None:
            completed_chunk_indices = precomputed_completed
        elif resume and use_batch:
            # Batch resume parity: skip requests already present in a prior
            # output so a re-run only submits the remainder. Since v1.20.0
            # batch finalization writes the unified `_output.json`; the
            # `_final_output.json` path covers legacy on-disk batch outputs.
            final_output_path = output_json_path.with_name(
                f"{file_path.stem}_final_output.json"
            )
            completed_chunk_indices = completed_indices_from_outputs(
                output_json_path, final_output_path
            )
            if completed_chunk_indices:
                messenger.info(
                    f"Batch resume: {len(completed_chunk_indices)} unit(s) already "
                    "present in existing output; they will not be re-submitted."
                )
        elif resume and not use_batch:
            status, completed_chunk_indices = detect_extraction_status(
                output_json_path, expected_chunks=len(chunks)
            )
            if status == FileStatus.COMPLETE:
                messenger.info(
                    f"Skipping {file_path.name}: already fully processed "
                    f"({len(completed_chunk_indices)} {unit_label}s)"
                )
                return "skipped"
            if status == FileStatus.PARTIAL:
                messenger.info(
                    f"Resuming {file_path.name}:"
                    f" {len(completed_chunk_indices)}/{len(chunks)}"
                    f" {unit_label}s already done"
                )

        # Refuse to resume a temp JSONL written by an older, incompatible
        # format (its custom_ids may be slice-relative, which would corrupt the
        # resume/merge). Decision 1: version + refuse, no migration.
        if (
            resume
            and not use_batch
            and completed_chunk_indices
            and temp_jsonl_path.exists()
            and not is_resumable_temp_jsonl(temp_jsonl_path)
        ):
            messenger.error(
                f"Refusing to resume {file_path.name}: its temp file "
                f"{temp_jsonl_path.name} predates the current resume format "
                f"(v{TEMP_JSONL_VERSION}). Re-run from scratch with --force, or "
                f"finish it with the ChronoMiner version that created it."
            )
            return "failed"

        # Create processing strategy and execute
        strategy = create_processing_strategy(use_batch, self.concurrency_config)

        processing_cancelled = False
        processing_exception: Exception | None = None
        failed_indices: list[int] = []
        budget_incomplete = False
        # Whether a prior (resumed) output exists to merge the temp records
        # into. Captured before the budget re-pass loop so intra-run passes,
        # which grow completed_chunk_indices, do not spuriously flip it on.
        merge_existing_flag = bool(completed_chunk_indices)

        try:
            process_kwargs: dict[str, Any] = {
                "chunks": chunks,
                "handler": handler,
                "dev_message": effective_dev_message,
                "model_config": self.model_config,
                "schema": selected_schema["schema"],
                "file_path": file_path,
                "temp_jsonl_path": temp_jsonl_path,
                "console_print": messenger.console_print,
                "completed_chunk_indices": completed_chunk_indices,
            }
            if image_chunks is not None:
                process_kwargs["image_chunks"] = image_chunks
            if context_image_data is not None:
                process_kwargs["context_image_data"] = context_image_data
            if chunk_indices is not None:
                process_kwargs["chunk_indices"] = chunk_indices
            if chunk_ranges is not None:
                process_kwargs["chunk_ranges"] = chunk_ranges

            # Pages still needing work on the visual streaming path; rebuilt
            # into a fresh one-shot source each pass.
            remaining_indices = (
                list(needed_indices) if needed_indices is not None else None
            )
            results: list[dict[str, Any]] = []
            stalled_resets = 0

            # Chunk-level token-budget loop: each pass admits chunks/pages
            # until the daily budget is exhausted, then drains in-flight work
            # and waits for the daily reset before re-passing over the still-
            # pending units. A single pass when the limit is disabled or in
            # batch mode (which never defers).
            while True:
                if source_factory is not None:
                    process_kwargs["image_source"] = source_factory(
                        remaining_indices or []
                    )
                process_kwargs["completed_chunk_indices"] = completed_chunk_indices

                results = await strategy.process_chunks(**process_kwargs)

                # Units skipped because the daily token budget was exhausted
                # mid-file (synchronous strategy only; batch never defers).
                deferred_indices = sorted(
                    r["chunk_index"]
                    for r in (results or [])
                    if isinstance(r, dict) and r.get("budget_deferred")
                )
                if use_batch or not deferred_indices:
                    break

                # Recompute the completed set from the live temp JSONL (union
                # with any prior resume set) so the next pass skips finished
                # units; deferred/errored units are absent and re-attempted.
                before = len(completed_chunk_indices)
                completed_chunk_indices = (
                    completed_chunk_indices
                    | _completed_indices_from_temp(temp_jsonl_path)
                )
                made_progress = len(completed_chunk_indices) > before
                if remaining_indices is not None and needed_indices is not None:
                    remaining_indices = [
                        i for i in needed_indices if i not in completed_chunk_indices
                    ]

                messenger.warning(
                    f"Daily token budget reached after "
                    f"{len(completed_chunk_indices)} {unit_label}(s); "
                    f"{len(deferred_indices)} {unit_label}(s) deferred. "
                    f"Waiting for daily reset..."
                )
                if not await check_and_wait_for_token_limit(ui):
                    budget_incomplete = True
                    break

                # Safeguard: if even a full day's reset yields no progress
                # twice running, a single unit exceeds the entire daily budget;
                # stop rather than wait forever.
                if not made_progress:
                    stalled_resets += 1
                    if stalled_resets >= 2:
                        messenger.warning(
                            f"A single {unit_label} appears to exceed the entire "
                            f"daily token budget; stopping. Raise daily_tokens to "
                            f"process the remaining {unit_label}s."
                        )
                        budget_incomplete = True
                        break
                else:
                    stalled_resets = 0

            # Synchronous strategy returns per-chunk results; failed chunks
            # surface as {"error": ..., "chunk_index": N} and write no temp
            # record. Collect their indices so the output is correctly marked
            # partial rather than reported as a full success.
            failed_indices = sorted(
                r["chunk_index"]
                for r in (results or [])
                if isinstance(r, dict) and "error" in r and "chunk_index" in r
            )
        except asyncio.CancelledError:
            processing_cancelled = True
            messenger.warning(
                f"Processing interrupted by user. "
                f"Attempting to persist completed {unit_label}s before exit..."
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
                        _cs_info: dict | None = None
                        if chunk_slice is not None and (
                            chunk_slice.first_n is not None
                            or chunk_slice.last_n is not None
                            or chunk_slice.page_range is not None
                        ):
                            _cs_info = {}
                            if chunk_slice.first_n is not None:
                                _cs_info["first_n"] = chunk_slice.first_n
                            if chunk_slice.last_n is not None:
                                _cs_info["last_n"] = chunk_slice.last_n
                            if chunk_slice.page_range is not None:
                                _cs_info["page_range"] = list(chunk_slice.page_range)
                        await asyncio.shield(
                            self._generate_output_files(
                                temp_jsonl_path,
                                output_json_path,
                                handler,
                                schema_paths,
                                messenger,
                                total_units=len(chunks),
                                partial=processing_cancelled
                                or processing_exception is not None
                                or bool(failed_indices)
                                or budget_incomplete,
                                failed_chunks=failed_indices,
                                chunk_slice_info=_cs_info,
                                merge_existing=merge_existing_flag,
                                image_provenance=image_provenance,
                            )
                        )
                        wrote_output = True
                    elif processing_cancelled:
                        messenger.warning(
                            f"Processing was interrupted before any {unit_label}s "
                            "completed; no output file generated."
                        )
                except Exception as gen_exc:
                    messenger.error(
                        f"Failed to write final output: {gen_exc}", exc_info=gen_exc
                    )

            self._cleanup_temp_files(use_batch, temp_jsonl_path, messenger)

            if processing_cancelled:
                if wrote_output:
                    messenger.warning(
                        f"Processing cancelled. Partial results saved to "
                        f"{output_json_path}"
                    )
            elif processing_exception is None:
                messenger.success(f"Completed processing of file: {file_path.name}")

        # Machine-readable per-file status for exit-code aggregation.
        if processing_exception is not None:
            return "failed"
        if budget_incomplete or failed_indices:
            return "partial"
        return "complete"

    def _determine_chunking_method(
        self,
        file_path: Path,
        global_chunking_method: str | None,
        messenger: _MessagingAdapter,
        ui: Any,
    ) -> str:
        """Determine which chunking method to use."""
        if global_chunking_method == "per-file":
            global_chunking_method = None

        if global_chunking_method is not None:
            messenger.info(
                f"Using global chunking method '{global_chunking_method}' "
                f"for file {file_path.name}"
            )
            return global_chunking_method
        else:
            # Ask for per-file chunking method
            if ui and hasattr(ui, "ask_file_chunking_method"):
                return ui.ask_file_chunking_method(file_path.name)
            else:
                return self._default_ask_file_chunking_method(file_path.name)

    def _setup_output_paths(
        self, file_path: Path, schema_paths: dict[str, Any]
    ) -> tuple[Path, Path, Path]:
        """Set up output directory paths."""
        if self.paths_config["general"]["input_paths_is_output_path"]:
            working_folder = ensure_path_safe(file_path.parent)
            output_json_path = ensure_path_safe(
                working_folder / f"{file_path.stem}_output.json"
            )
            temp_jsonl_path = ensure_path_safe(
                working_folder / f"{file_path.stem}_temp.jsonl"
            )
            working_folder.mkdir(parents=True, exist_ok=True)
        else:
            output_path_str = schema_paths.get("output", "")
            # CM-7: Validate that output path is not empty or CWD
            if not output_path_str or not str(output_path_str).strip():
                raise ValueError(
                    "Output path is not configured. Set 'output' in paths_config.yaml "
                    "for this schema, or enable "
                    "'input_paths_is_output_path: true' in general settings."
                )
            working_folder = ensure_path_safe(Path(output_path_str))
            if working_folder.resolve() == Path.cwd().resolve():
                raise ValueError(
                    f"Output path '{working_folder}' resolves to the current working "
                    "directory. Configure a specific output directory to avoid mixing "
                    "output with project files."
                )
            if self.output_mode == "mirror" and self.input_root is not None:
                try:
                    rel = file_path.relative_to(self.input_root)
                except ValueError:
                    rel = Path(file_path.name)
                mirror_dir = ensure_path_safe(working_folder / rel.parent)
                mirror_dir.mkdir(parents=True, exist_ok=True)
                temp_folder = ensure_path_safe(mirror_dir / "temp_jsonl")
                temp_folder.mkdir(parents=True, exist_ok=True)
                output_json_path = ensure_path_safe(
                    mirror_dir / f"{file_path.stem}_output.json"
                )
                temp_jsonl_path = ensure_path_safe(
                    temp_folder / f"{file_path.stem}_temp.jsonl"
                )
                return mirror_dir, output_json_path, temp_jsonl_path

            temp_folder = ensure_path_safe(working_folder / "temp_jsonl")
            working_folder.mkdir(parents=True, exist_ok=True)
            temp_folder.mkdir(parents=True, exist_ok=True)
            output_json_path = ensure_path_safe(
                working_folder / f"{file_path.stem}_output.json"
            )
            temp_jsonl_path = ensure_path_safe(
                temp_folder / f"{file_path.stem}_temp.jsonl"
            )

        return working_folder, output_json_path, temp_jsonl_path

    async def _generate_output_files(
        self,
        temp_jsonl_path: Path,
        output_json_path: Path,
        handler: Any,
        schema_paths: dict[str, Any],
        messenger: _MessagingAdapter,
        *,
        total_units: int,
        partial: bool = False,
        failed_chunks: list[int] | None = None,
        chunk_slice_info: dict | None = None,
        merge_existing: bool = False,
        image_provenance: dict[str, Any] | None = None,
    ) -> None:
        """Generate final output files from temporary JSONL.

        When ``merge_existing`` is set (resume of a partial extraction), the
        records already saved in ``output_json_path`` are merged with those
        rebuilt from the temp JSONL, keyed by ``custom_id`` (freshly-processed
        records win). This prevents data loss when the prior temp JSONL was not
        retained: the temp file then holds only the newly-processed chunks,
        while the skip-set was derived from the existing output.json.

        ``total_units`` is the true number of units the file was chunked into
        (``len(chunks)``). It is stamped verbatim as ``total_chunks`` in the
        metadata so the value reflects the full denominator rather than the
        success count, even when the run ends partial or cancelled.
        """
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
                                response_field = record.get("response", {}).get(
                                    "body", {}
                                )
                                output_record = {
                                    "custom_id": record.get("custom_id"),
                                    "chunk_index": record.get("chunk_index"),
                                    "chunk_range": record.get("chunk_range"),
                                    "response": response_field,
                                }
                                if "image_provenance" in record:
                                    output_record["image_provenance"] = record[
                                        "image_provenance"
                                    ]
                                results.append(output_record)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse line in temp file: {e}")
                            continue

            if merge_existing:
                results = self._merge_with_existing_output(
                    output_json_path, results, messenger
                )

            # Keep only the response side in the final output. The full API
            # call (input messages + base64 images) remains in the temp JSONL
            # for reproducibility; downstream consumers read output_text and
            # response_data only. Applies to both temp-derived and merged
            # records so resumed/cleaned partials stay lean.
            for record in results:
                record["response"] = lean_response(record.get("response", {}))

            # Sort results by chunk_index, handling None values
            results.sort(key=lambda x: x.get("chunk_index") or 0)

            # Build output structure with metadata
            chunking_method = "unknown"
            model_name = self.model_config.get("extraction_model", {}).get(
                "name", "unknown"
            )
            schema_name_for_meta = getattr(handler, "schema_name", "unknown")
            output_data = {
                _METADATA_KEY: build_extraction_metadata(
                    schema_name=schema_name_for_meta,
                    model_name=model_name,
                    chunking_method=chunking_method,
                    total_chunks=total_units,
                    chunk_slice_info=chunk_slice_info,
                    partial=partial,
                    failed_chunks=failed_chunks,
                    image_provenance=image_provenance,
                    # Text runs stamp the chunking behaviour version; visual runs
                    # (image_provenance set) do not chunk text.
                    chunking_text_version=(
                        CHUNKING_TEXT_VERSION if image_provenance is None else None
                    ),
                ),
                "records": results,
            }

            with output_json_path.open("w", encoding="utf-8") as outf:
                json.dump(output_data, outf, indent=2, ensure_ascii=False)

            if partial:
                if failed_chunks:
                    messenger.warning(
                        f"Partial output: {len(failed_chunks)} chunk(s) failed "
                        f"{failed_chunks}. Saved to {output_json_path}"
                    )
                else:
                    messenger.warning(
                        f"Partial structured JSON output saved to {output_json_path}"
                    )
            else:
                messenger.success(
                    f"Final structured JSON output saved to {output_json_path}"
                )
        except Exception as e:
            messenger.error(f"Failed to write final output: {e}", exc_info=e)
            return

        # Generate additional formats
        self._generate_additional_formats(
            output_json_path, handler, schema_paths, messenger
        )

    def _merge_with_existing_output(
        self,
        output_json_path: Path,
        new_results: list[dict[str, Any]],
        messenger: _MessagingAdapter,
    ) -> list[dict[str, Any]]:
        """Overlay freshly-generated records onto previously-saved ones.

        Reads the records currently stored in ``output_json_path`` and merges
        them with ``new_results`` keyed by ``custom_id``; newly-processed
        records take precedence on conflict. Records without a ``custom_id``
        cannot be deduplicated and are kept as-is.

        :param output_json_path: Existing output JSON to read prior records from
        :param new_results: Records rebuilt from the temp JSONL this run
        :param messenger: Messaging adapter for user-facing notices
        :return: Merged record list
        """
        if not output_json_path.exists():
            return new_results

        try:
            with output_json_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            messenger.warning(
                f"Could not read existing output for resume merge "
                f"({output_json_path.name}): {exc}. Keeping new records only."
            )
            return new_results

        prior_records: list[dict[str, Any]] = []
        if isinstance(data, dict):
            prior_records = data.get("records", []) or []
        elif isinstance(data, list):
            prior_records = data

        merged: dict[str, dict[str, Any]] = {}
        extras: list[dict[str, Any]] = []
        for record in prior_records:
            custom_id = record.get("custom_id") if isinstance(record, dict) else None
            if custom_id:
                merged[str(custom_id)] = record
        new_cids: set[str] = set()
        for record in new_results:
            custom_id = record.get("custom_id")
            if custom_id:
                new_cids.add(str(custom_id))
                merged[str(custom_id)] = record
            else:
                extras.append(record)

        carried = sum(1 for cid in merged if cid not in new_cids)
        if carried:
            messenger.info(
                f"Resume: preserved {carried} previously-saved record(s) "
                "from existing output."
            )

        return list(merged.values()) + extras

    def _generate_additional_formats(
        self,
        output_json_path: Path,
        handler: Any,
        schema_paths: dict[str, Any],
        messenger: _MessagingAdapter,
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
        self, use_batch: bool, temp_jsonl_path: Path, messenger: _MessagingAdapter
    ) -> None:
        """Clean up temporary files if not needed."""
        if use_batch:
            logger.info("Batch processing enabled. Keeping temporary JSONL")
            messenger.info(f"Preserving {temp_jsonl_path.name} for batch tracking")
        else:
            keep_temp = self.paths_config["general"].get("retain_temporary_jsonl", True)
            if not keep_temp:
                try:
                    temp_jsonl_path.unlink()
                    logger.info(f"Deleted temporary file: {temp_jsonl_path}")
                    messenger.info(f"Deleted temporary file: {temp_jsonl_path.name}")
                except Exception as e:
                    messenger.error(
                        f"Could not delete temporary file {temp_jsonl_path.name}: {e}"
                    )

    def _default_ask_file_chunking_method(self, file_name: str) -> str:
        """Default implementation if UI not provided."""
        ui_print(f"\nSelect chunking method for file '{file_name}':")
        ui_print(
            "  1. Automatic chunking - Split text based on token limits "
            "with no intervention"
        )
        ui_print(
            "  2. Interactive chunking - View default chunks and manually "
            "adjust boundaries"
        )
        ui_print(
            "  3. Predefined chunks - Use saved boundaries from "
            "{file}_line_ranges.txt file"
        )

        choice = ui_input("Enter option (1-3): ").strip()
        if choice == "1":
            return "auto"
        elif choice == "2":
            return "auto-adjust"
        elif choice == "3":
            return "line_ranges.txt"
        else:
            print_warning("Invalid selection, defaulting to automatic chunking.")
            return "auto"
