from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from modules.logger import setup_logger

logger = setup_logger(__name__)


def sdk_to_dict(obj: Any) -> Dict[str, Any]:
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

    data: Dict[str, Any] = {}
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
        logger.warning("Unable to convert SDK object %s to dict; returning empty mapping", type(obj))
    return data


def list_all_batches(client: Any, limit: int = 100) -> List[Dict[str, Any]]:
    """List all batches with pagination, returning plain dictionaries."""
    batches: List[Dict[str, Any]] = []
    after: Optional[str] = None
    page_index = 0

    while True:
        page_index += 1
        page = client.batches.list(limit=limit, after=after) if after else client.batches.list(limit=limit)
        data = getattr(page, "data", None) or page
        page_items = [sdk_to_dict(item) for item in data]
        batches.extend(page_items)

        has_more = False
        last_id = None
        try:
            has_more = bool(getattr(page, "has_more", False))
            last_id = getattr(page, "last_id", None)
        except Exception:
            try:
                has_more = bool(page.get("has_more", False))
                last_id = page.get("last_id")
            except Exception:
                has_more = False
                last_id = None

        logger.info("Retrieved batches page %s (%s item(s)); has_more=%s", page_index, len(page_items), has_more)
        if not has_more or not last_id:
            break
        after = last_id

    return batches


def coerce_file_id(candidate: Any) -> Optional[str]:
    """Coerce various response shapes into a file id string."""
    if isinstance(candidate, str) and candidate:
        return candidate
    if isinstance(candidate, dict):
        cid = candidate.get("id") or candidate.get("file_id")
        if isinstance(cid, str) and cid:
            return cid
    if isinstance(candidate, list) and candidate:
        first = candidate[0]
        if isinstance(first, str) and first:
            return first
        if isinstance(first, dict):
            cid = first.get("id") or first.get("file_id")
            if isinstance(cid, str) and cid:
                return cid
    return None
