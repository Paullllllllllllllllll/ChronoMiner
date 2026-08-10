"""
OpenAI SDK object utilities.

Normalizes provider SDK response objects into plain dictionaries for the
batch backends. Not used by the LangChain-based synchronous processing
pipeline.

For LangChain multi-provider support, see langchain_provider.py.
"""

from __future__ import annotations

import json
from typing import Any

from modules.infra.logger import setup_logger

logger = setup_logger(__name__)


def sdk_to_dict(obj: Any) -> dict[str, Any]:
    """Convert an OpenAI SDK object into a plain dict when possible."""
    if isinstance(obj, dict):
        return obj
    for attr in ("model_dump", "to_dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                pass
    try:
        j = getattr(obj, "json", None)
        if callable(j):
            return json.loads(j())
    except Exception:
        pass

    data: dict[str, Any] = {}
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            val = getattr(obj, name)
            if not callable(val):
                data[name] = val
        except Exception:
            continue
    if not data:
        logger.warning(
            "Unable to convert SDK object %s to dict; returning empty mapping",
            type(obj),
        )
    return data
