"""ChronoMiner extraction workflow package.

Orchestrates structured-data extraction for a single input file (text, image,
or PDF) through chunking, LLM dispatch (sync or batch), and output
generation.
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
    "read_extraction_metadata",
    "BaseSchemaHandler",
    "get_schema_handler",
    "schema_handlers_registry",
]
