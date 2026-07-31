"""Batch lifecycle package: submit, check, cancel, repair.

Provider-agnostic orchestration layered on top of
:mod:`modules.batch.backends`. Shared helpers live in
``modules.batch.ops``; ``modules.batch.diagnostics`` provides
``extract_custom_id_mapping``.
"""

from modules.batch.backends.base import (
    BatchBackend,
    BatchHandle,
    BatchRequest,
    BatchResultItem,
    BatchStatus,
    BatchStatusInfo,
)
from modules.batch.backends.factory import (
    clear_backend_cache,
    get_batch_backend,
    supports_batch,
)
from modules.batch.diagnostics import extract_custom_id_mapping

__all__ = [
    "BatchBackend",
    "BatchHandle",
    "BatchRequest",
    "BatchResultItem",
    "BatchStatus",
    "BatchStatusInfo",
    "get_batch_backend",
    "supports_batch",
    "clear_backend_cache",
    "extract_custom_id_mapping",
]
