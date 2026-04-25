"""Batch lifecycle package: submit, check, cancel, repair.

Provider-agnostic orchestration layered on top of
:mod:`modules.batch.backends`. Replaces the former split between
``modules.batch`` (providers) and the three batch-related scripts in
``main/`` (check, cancel, repair); shared helpers live in
``modules.batch.ops``; diagnostics live in ``modules.batch.diagnostics``.
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
from modules.batch.diagnostics import (
    diagnose_batch_failure,
    extract_custom_id_mapping,
)

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
    "diagnose_batch_failure",
    "extract_custom_id_mapping",
]
