"""Turn model answers into extraction objects and their entries.

An answer under the ``schema`` format is the provider's JSON. A prompted
answer (``json``) may be loosely shaped: wrapped in a code fence, preceded by
a preamble, or followed by prose. :func:`parse_answer` finds the JSON object
in either, as ChronoMiner's converters always have: the text itself when it
starts with a brace or bracket, else the first fenced block, else the first
balanced object; when none of those parses, the fence and prose removal of
:func:`normalize_llm_text` and the last complete object near the end (of
concatenated objects, the last one).
:func:`parse_extraction` returns the entries of an answer: none when the
answer sets ``contains_no_content_of_requested_type``, has no ``entries``
list or holds no JSON object.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

__all__ = [
    "INVALID_JSON",
    "NO_CONTENT_FLAG",
    "NO_OBJECT",
    "answer_outcome",
    "extract_json_text",
    "normalize_llm_text",
    "parse_answer",
    "parse_extraction",
]

logger = logging.getLogger(__name__)

NO_CONTENT_FLAG = "contains_no_content_of_requested_type"
"""The schema flag an answer sets when its chunk holds nothing to extract."""

_FIRST_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


def _balanced_object(text: str, start: int) -> str | None:
    """Return the object that opens at *start*, respecting JSON strings."""
    depth = 0
    in_string = False
    escape_next = False
    for index in range(start, len(text)):
        char = text[index]
        if escape_next:
            escape_next = False
            continue
        if char == "\\":
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def extract_json_text(text: str) -> str | None:
    """Return the JSON text of an answer that may carry a fence or a preamble.

    Tries, in order: the stripped text when it starts with ``{`` or ``[``;
    the content of the first code fence; the first balanced ``{...}``.
    """
    stripped = text.strip()
    if not stripped:
        return None
    if stripped[0] in ("{", "["):
        return stripped
    match = _FIRST_FENCE_RE.search(text)
    if match:
        candidate = match.group(1).strip()
        if candidate:
            return candidate
    start = text.find("{")
    if start != -1:
        return _balanced_object(text, start)
    return None


def _try_parse_json(text: str) -> dict[str, Any] | None:
    try:
        result = json.loads(text)
        return result if isinstance(result, dict) else None
    except (json.JSONDecodeError, ValueError, TypeError):
        return None
    except RecursionError:
        # Deeply nested input exhausts the recursion limit; treat it as unparseable.
        return None


_CODE_FENCE_RE = re.compile(
    r"```(?:json|JSON)?\s*\n(.*?)\n\s*```",
    re.DOTALL,
)

_FENCE_DELIM_RE = re.compile(r"^\s*```")


def _strip_code_fences(text: str) -> str:
    """Extract content from markdown code fences if present.

    Applies only when the response starts with a fence. Two shapes start
    with one:

    - A JSON answer wrapped in one fence (wrapped content opens with ``{``):
      only the outermost fence is stripped, so fenced blocks inside string
      values survive.
    - A "thinking" fenced block followed by the fenced answer: the last
      fenced block is the answer.

    Otherwise returns the original text unchanged.
    """
    if not text.lstrip().startswith("```"):
        return text
    stripped = text.strip()
    lines = stripped.split("\n")
    if (
        len(lines) >= 2
        and _FENCE_DELIM_RE.match(lines[0])
        and _FENCE_DELIM_RE.match(lines[-1])
    ):
        inner = "\n".join(lines[1:-1])
        if inner.lstrip().startswith("{"):
            return inner.strip()
    matches = _CODE_FENCE_RE.findall(stripped)
    if matches:
        return str(matches[-1]).strip()
    return text


def normalize_llm_text(text: str) -> str:
    """Normalize raw LLM text output for JSON extraction.

    Strips surrounding whitespace, extracts the content of a code fence if
    present, and otherwise isolates a JSON object by removing preamble and
    postamble prose. Clean JSON passes through unchanged.
    """
    if not text:
        return text

    stripped = text.strip()

    defenced = _strip_code_fences(stripped)
    if defenced != stripped:
        return defenced.strip()

    trimmed = stripped.lstrip()
    if not trimmed.startswith("{") and "{" in trimmed:
        first_brace = trimmed.index("{")
        candidate = trimmed[first_brace:]
        last_brace = candidate.rfind("}")
        if last_brace != -1:
            json_candidate = candidate[: last_brace + 1]
            parsed = _try_parse_json(json_candidate)
            if parsed is not None:
                return json_candidate.strip()

    return stripped


# Caps the opening braces tried by _salvage_last_json_object, so a string with
# thousands of unbalanced braces cannot cause quadratic re-scanning.
_MAX_SALVAGE_CANDIDATES = 1000


def _salvage_last_json_object(text: str) -> dict[str, Any] | None:
    """Return the last valid JSON object near the end of concatenated or mixed text.

    The backward walk tries at most ``_MAX_SALVAGE_CANDIDATES`` opening
    braces and returns ``None`` beyond that.
    """
    if not text:
        return None
    last_close = text.rfind("}")
    if last_close == -1:
        return None
    attempts = 0
    i = last_close
    while i >= 0:
        if text[i] == "{":
            candidate = text[i : last_close + 1]
            obj = _try_parse_json(candidate)
            if obj is not None:
                return obj
            attempts += 1
            if attempts >= _MAX_SALVAGE_CANDIDATES:
                return None
        i -= 1
    return None


def parse_answer(text: str | None) -> dict[str, Any] | None:
    """Return the JSON object of a model answer, or None when it holds none."""
    if not text or not text.strip():
        return None
    candidate = extract_json_text(text)
    if candidate is not None:
        parsed = _try_parse_json(candidate)
        if parsed is not None:
            return parsed
    normalized = normalize_llm_text(text)
    parsed = _try_parse_json(normalized) or _salvage_last_json_object(normalized)
    if parsed is None and normalized != text.strip():
        parsed = _salvage_last_json_object(text)
    return parsed


def parse_extraction(text: str | None) -> list[dict[str, Any]]:
    """Return the entries of a model answer; an empty list when there are none.

    An answer that sets :data:`NO_CONTENT_FLAG`, has no ``entries`` list or
    holds no JSON object has no entries; entries that are not objects are
    dropped.
    """
    parsed = parse_answer(text)
    if parsed is None:
        if text and text.strip():
            logger.warning("Could not extract JSON from the answer: %s...", text[:200])
        return []
    if parsed.get(NO_CONTENT_FLAG, False):
        return []
    entries = parsed.get("entries")
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, dict)]


NO_OBJECT = "Answer holds no JSON object"
"""The error of a chunk whose answer holds no JSON object, which fails it."""
INVALID_JSON = "invalid_json"
"""The error type of such a chunk."""


def answer_outcome(text: str | None) -> str:
    """Return how an answer reads: ``no_content``, ``unparsed`` or ``ok``.

    ``unparsed`` marks an answer kept as text without a JSON object; ``ok``
    any other answer, with or without entries.
    """
    parsed = parse_answer(text)
    if parsed is None:
        return "unparsed"
    if parsed.get(NO_CONTENT_FLAG, False):
        return "no_content"
    return "ok"
