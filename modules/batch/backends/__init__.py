"""Provider-specific batch backends.

Canonical location for ``BatchBackend`` implementations (OpenAI, Anthropic,
Google). Consumers should import from :mod:`modules.batch` rather than
reaching into the ``backends`` sub-package directly.
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
]
