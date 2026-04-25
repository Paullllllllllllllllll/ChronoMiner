"""ChronoMiner extraction workflow package.

Orchestrates structured-data extraction for a single input file (text, image,
or PDF) through chunking, LLM dispatch (sync or batch), and output
generation. Absorbs the former ``modules.extract.extraction`` and the
execution-strategy portion of ``modules.core`` (processing_strategy, resume,
file_processor, schema_handlers).

The future ``orchestrator.py`` / ``config_builder.py`` extraction from
``main/process_text_files.py`` lands in a follow-up step.
"""

from modules.extract.file_processor import FileProcessor, is_visual_input
from modules.extract.processing_strategy import (
    BatchProcessingStrategy,
    ProcessingStrategy,
    SynchronousProcessingStrategy,
    create_processing_strategy,
)
from modules.extract.resume import (
    METADATA_KEY,
    FileStatus,
    build_extraction_metadata,
    detect_extraction_status,
    get_output_json_path,
    read_extraction_metadata,
)
from modules.extract.schema_handlers import (
    BaseSchemaHandler,
    get_schema_handler,
    schema_handlers_registry,
)

__all__ = [
    "FileProcessor",
    "is_visual_input",
    "ProcessingStrategy",
    "SynchronousProcessingStrategy",
    "BatchProcessingStrategy",
    "create_processing_strategy",
    "FileStatus",
    "METADATA_KEY",
    "build_extraction_metadata",
    "detect_extraction_status",
    "get_output_json_path",
    "read_extraction_metadata",
    "BaseSchemaHandler",
    "get_schema_handler",
    "schema_handlers_registry",
]
