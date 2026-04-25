"""Cross-cutting infrastructure primitives for ChronoMiner.

Logger setup, filesystem path safety, daily-token-budget tracking,
asynchronous concurrency, text chunking, and JSONL streaming. No
modules here import from higher-level layers (``llm``, ``batch``,
``conversion``, ``extract``, ``line_ranges``, ``ui``).
"""

from modules.infra.chunking import (
    ChunkHandler,
    ChunkingService,
    ChunkingStrategy,
    ChunkSlice,
    TextProcessor,
    TokenBasedChunking,
    apply_chunk_slice,
    load_line_ranges,
)
from modules.infra.concurrency import run_concurrent_tasks
from modules.infra.jsonl import (
    JsonlWriter,
    build_jsonl_header,
    compute_ranges_fingerprint,
    compute_stats_from_jsonl,
    extract_completed_ids,
    finalize_jsonl_header,
    is_jsonl_adjustment_complete,
    read_jsonl_header,
    read_jsonl_records,
    validate_jsonl_header,
)
from modules.infra.logger import setup_logger
from modules.infra.paths import (
    HASH_LENGTH,
    MAX_SAFE_NAME_LENGTH,
    create_safe_directory_name,
    create_safe_log_filename,
    ensure_path_safe,
)
from modules.infra.token_tracker import (
    DailyTokenTracker,
    check_and_wait_for_token_limit,
    check_token_limit_enabled,
    get_token_tracker,
)

__all__ = [
    "setup_logger",
    "HASH_LENGTH",
    "MAX_SAFE_NAME_LENGTH",
    "create_safe_directory_name",
    "create_safe_log_filename",
    "ensure_path_safe",
    "DailyTokenTracker",
    "get_token_tracker",
    "check_token_limit_enabled",
    "check_and_wait_for_token_limit",
    "run_concurrent_tasks",
    "TextProcessor",
    "ChunkingStrategy",
    "TokenBasedChunking",
    "ChunkHandler",
    "ChunkSlice",
    "ChunkingService",
    "apply_chunk_slice",
    "load_line_ranges",
    "JsonlWriter",
    "read_jsonl_records",
    "extract_completed_ids",
    "build_jsonl_header",
    "compute_ranges_fingerprint",
    "compute_stats_from_jsonl",
    "finalize_jsonl_header",
    "is_jsonl_adjustment_complete",
    "read_jsonl_header",
    "validate_jsonl_header",
]
