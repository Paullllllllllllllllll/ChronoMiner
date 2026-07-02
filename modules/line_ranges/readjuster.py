"""
Line range readjustment module for semantic boundary detection.

Supports multiple LLM providers via LangChain:
- OpenAI (default)
- Anthropic (Claude)
- Google (Gemini)
- OpenRouter (multi-provider access)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import unicodedata
from collections.abc import Generator, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from modules.config.capabilities import detect_capabilities
from modules.config.context import resolve_context_for_readjustment
from modules.infra.chunking import TextProcessor, load_line_ranges
from modules.infra.jsonl import (
    JsonlWriter,
    build_jsonl_header,
    compute_ranges_fingerprint,
    compute_stats_from_jsonl,
    extract_completed_ids,
    finalize_jsonl_header,
    read_jsonl_header,
    validate_jsonl_header,
)
from modules.infra.paths import ensure_path_safe
from modules.infra.token_tracker import (
    check_and_wait_for_token_limit,
    get_token_tracker,
)
from modules.llm.langchain_provider import ProviderConfig
from modules.llm.openai_utils import LLMExtractor, open_extractor, process_text_chunk
from modules.llm.prompt_utils import (
    PROMPTS_DIR,
    load_prompt_template,
    render_prompt_with_schema,
)

logger = logging.getLogger(__name__)

# Sentinel injected into the model's input text immediately above the line
# where the current chunk starts, so the model knows which boundary it is
# judging. Not part of the source text; never usable as a semantic marker.
BOUNDARY_SENTINEL = "<<<CURRENT_CHUNK_START>>>"

# Upper bound on the number of lines sent in a single no-content verification
# call. Ranges are chunk-sized (a few hundred lines) in practice; pathological
# longer ranges are scanned in consecutive full-coverage windows of this size
# rather than sampled with gaps.
MAX_VERIFY_WINDOW_LINES = 1000


def clamp_ranges_to_length(
    ranges: Iterable[tuple[int, int]], total_lines: int
) -> list[tuple[int, int]]:
    """Clamp ``(start, end)`` line ranges to ``[1, total_lines]``.

    A ``_line_ranges.txt`` whose ``end`` exceeds the file length would
    otherwise raise ``IndexError`` deep in the context formatter. Ranges that
    are wholly out of bounds (``start > end`` after clamping) are dropped with
    a warning rather than silently producing an empty or reversed slice.
    """
    clamped: list[tuple[int, int]] = []
    for start, end in ranges:
        clamped_start = max(1, start)
        clamped_end = min(total_lines, end)
        if clamped_start > clamped_end:
            logger.warning(
                "Dropping out-of-bounds line range (%d, %d): file has %d line(s)",
                start,
                end,
                total_lines,
            )
            continue
        clamped.append((clamped_start, clamped_end))
    return clamped


SEMANTIC_BOUNDARY_SCHEMA: dict[str, Any] = {
    "name": "SemanticBoundaryResponse",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "contains_no_semantic_boundary": {
                "type": "boolean",
                "description": (
                    "Set to true when the provided text contains NO content of the"
                    " required semantic type at all. Use this when you are confident"
                    " no relevant content exists anywhere in the visible context."
                ),
            },
            "needs_more_context": {
                "type": "boolean",
                "description": (
                    "Set to true when you believe the semantic boundary exists"
                    " somewhere around the visible text but you need to see more"
                    " surrounding content to accurately identify it."
                ),
            },
            "boundary_already_on_target": {
                "type": "boolean",
                "description": (
                    "Set to true when the current range start is already positioned"
                    " at a semantic boundary and no adjustment is needed. Leave"
                    " semantic_marker empty when this is true."
                ),
            },
            "certainty": {
                "type": "integer",
                "description": (
                    "Your confidence level in this response as an integer from 0-100."
                    " Use 0-40 for low confidence, 41-70 for moderate confidence,"
                    " 71-100 for high confidence. This applies to whatever decision"
                    " you make (boundary found, no content, or needs more context)."
                ),
            },
            "semantic_marker": {
                "type": "string",
                "description": (
                    "A precise 5-15 character verbatim substring that marks the"
                    " semantic boundary. Leave empty if contains_no_semantic_boundary"
                    " or needs_more_context is true."
                ),
            },
        },
        "required": [
            "contains_no_semantic_boundary",
            "needs_more_context",
            "boundary_already_on_target",
            "certainty",
            "semantic_marker",
        ],
        "additionalProperties": False,
    },
}


@dataclass
class BoundaryDecision:
    contains_no_semantic_boundary: bool
    needs_more_context: bool
    certainty: int
    boundary_already_on_target: bool = False
    semantic_marker: str | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> BoundaryDecision:
        return cls(
            contains_no_semantic_boundary=bool(
                payload.get("contains_no_semantic_boundary", False)
            ),
            needs_more_context=bool(payload.get("needs_more_context", False)),
            certainty=int(payload.get("certainty", 0)),
            boundary_already_on_target=bool(
                payload.get("boundary_already_on_target", False)
            ),
            semantic_marker=payload.get("semantic_marker") or None,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSONL storage."""
        return {
            "contains_no_semantic_boundary": self.contains_no_semantic_boundary,
            "needs_more_context": self.needs_more_context,
            "boundary_already_on_target": self.boundary_already_on_target,
            "certainty": self.certainty,
            "semantic_marker": self.semantic_marker,
        }


@dataclass
class RangeResult:
    """Result of processing a single line range, including an audit trail."""

    range_index: int
    original_range: tuple[int, int]
    adjusted_range: tuple[int, int]
    should_delete: bool
    decision: BoundaryDecision
    attempts: list[dict[str, Any]] = field(default_factory=list)
    total_llm_calls: int = 0

    def to_jsonl_record(self, stem: str) -> dict[str, Any]:
        """Convert to a JSONL record using the shared envelope format."""
        return {
            "custom_id": f"{stem}-range-{self.range_index}",
            "response": {
                "body": {
                    "original_range": list(self.original_range),
                    "adjusted_range": list(self.adjusted_range),
                    "should_delete": self.should_delete,
                    "decision": self.decision.to_dict(),
                    "attempts": self.attempts,
                    "total_llm_calls": self.total_llm_calls,
                },
            },
        }


class LineRangeReadjuster:
    """Adjust ``_line_ranges.txt`` files so chunk boundaries align with semantic
    boundaries."""

    def __init__(
        self,
        model_config: dict[str, Any],
        *,
        context_window: int = 6,
        prompt_path: Path | None = None,
        matching_config: dict[str, Any] | None = None,
        retry_config: dict[str, Any] | None = None,
    ) -> None:
        transcription_cfg = model_config.get("extraction_model", {})
        model_name: str = transcription_cfg.get("name", "")
        if not model_name:
            raise ValueError(
                "extraction_model.name must be configured to use LineRangeReadjuster"
            )

        self.model_name = model_name
        self._model_config = model_config
        self.context_window = max(1, int(context_window))
        self.prompt_path = prompt_path or (PROMPTS_DIR / "semantic_boundary_prompt.txt")
        self.prompt_template = load_prompt_template(self.prompt_path)
        self.prompt_hash = hashlib.sha256(
            self.prompt_template.encode("utf-8")
        ).hexdigest()
        self.text_processor = TextProcessor()
        self._enable_cache_control = detect_capabilities(
            model_name
        ).supports_prompt_caching

        # Load matching configuration with defaults
        self.matching_config = matching_config or {}
        self.normalize_whitespace = self.matching_config.get(
            "normalize_whitespace", True
        )
        self.case_sensitive = self.matching_config.get("case_sensitive", False)
        self.normalize_diacritics = self.matching_config.get(
            "normalize_diacritics", True
        )
        self.strip_punctuation = self.matching_config.get("strip_punctuation", False)
        self.allow_substring_match = self.matching_config.get(
            "allow_substring_match", True
        )
        self.min_substring_length = self.matching_config.get("min_substring_length", 8)

        # Load retry configuration with defaults
        self.retry_config = retry_config or {}
        self.certainty_threshold = self.retry_config.get("certainty_threshold", 70)
        self.max_low_certainty_retries = self.retry_config.get(
            "max_low_certainty_retries", 3
        )
        self.max_context_expansion_attempts = self.retry_config.get(
            "max_context_expansion_attempts", 3
        )
        self.delete_ranges_with_no_content = self.retry_config.get(
            "delete_ranges_with_no_content", True
        )
        # scan_range_multiplier is deprecated: window growth is now geometric
        # and sized by the retry budgets; a leftover YAML key is tolerated.
        if "scan_range_multiplier" in self.retry_config:
            logger.debug(
                "retry.scan_range_multiplier is deprecated and ignored;"
                " window expansion is derived from the retry budgets."
            )
        self.max_marker_mismatch_retries = self.retry_config.get(
            "max_marker_mismatch_retries", 2
        )

        max_gap_setting = self.retry_config.get("max_gap_between_ranges")
        if max_gap_setting is None:
            self.max_gap_between_ranges: int | None = None
        else:
            try:
                self.max_gap_between_ranges = max(0, int(max_gap_setting))
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid max_gap_between_ranges value '%s';"
                    " disabling gap enforcement.",
                    max_gap_setting,
                )
                self.max_gap_between_ranges = None

    async def ensure_adjusted_line_ranges(
        self,
        *,
        text_file: Path,
        line_ranges_file: Path | None = None,
        dry_run: bool = False,
        boundary_type: str | None = None,
        retain_temp_jsonl: bool = True,
        force_fresh: bool = False,
        first_n_chunks: int | None = None,
        last_n_chunks: int | None = None,
    ) -> list[tuple[int, int]]:
        """Ensure the provided line ranges align with semantic boundaries."""
        text_file = text_file.resolve()
        if line_ranges_file is None:
            line_ranges_file = self._infer_line_ranges_file(text_file)
        line_ranges_file = line_ranges_file.resolve()

        if not line_ranges_file.exists():
            raise FileNotFoundError(f"Line ranges file not found: {line_ranges_file}")

        if not boundary_type:
            raise ValueError(
                "boundary_type must be provided when readjusting line ranges"
            )

        ranges = load_line_ranges(line_ranges_file)
        if not ranges:
            logger.warning("No ranges found in %s", line_ranges_file)
            return []

        safe_text_file = ensure_path_safe(text_file)
        with safe_text_file.open("r", encoding="utf-8") as handle:
            raw_lines = handle.readlines()

        # Clamp ranges to the actual file length before any indexing. An `end`
        # past the line count would otherwise raise IndexError in the context
        # formatter; out-of-bounds ranges are dropped.
        ranges = clamp_ranges_to_length(ranges, len(raw_lines))
        if not ranges:
            logger.warning(
                "All ranges fell outside the file bounds (%d lines); nothing "
                "to readjust in %s",
                len(raw_lines),
                line_ranges_file,
            )
            return []

        # Chunk slicing selects which ranges to adjust; unselected ranges pass
        # through unchanged. Indices (and custom_ids) stay absolute so that a
        # sliced run never truncates the line-ranges file and its temp JSONL
        # can be safely merged with later runs.
        total_range_count = len(ranges)
        if first_n_chunks is not None:
            selected_indices = set(
                range(1, min(first_n_chunks, total_range_count) + 1)
            )
            logger.info(
                "Adjusting first %d range(s) of %d total",
                len(selected_indices),
                total_range_count,
            )
        elif last_n_chunks is not None:
            selected_indices = set(
                range(
                    max(1, total_range_count - last_n_chunks + 1),
                    total_range_count + 1,
                )
            )
            logger.info(
                "Adjusting last %d range(s) of %d total",
                len(selected_indices),
                total_range_count,
            )
        else:
            selected_indices = set(range(1, total_range_count + 1))

        # Detect provider from model name and get appropriate API key
        provider = ProviderConfig._detect_provider(self.model_name)
        api_key = ProviderConfig._get_api_key(provider)
        if not api_key:
            raise RuntimeError(
                f"API key not found for provider {provider}."
                " Set the appropriate environment variable."
            )

        # Resolve unified context using hierarchical resolution
        context, context_path = resolve_context_for_readjustment(
            text_file=text_file,
        )

        if context_path:
            logger.info(f"Using line ranges context from: {context_path}")
        else:
            logger.debug(f"No adjust_context found for '{text_file.name}'")

        # Compute fingerprint early (needed for header and staleness
        # validation before the processing loop starts).
        current_fingerprint = compute_ranges_fingerprint(line_ranges_file)
        prompt_hash = self.prompt_hash

        # Temp JSONL for per-range persistence and resume
        stem = line_ranges_file.stem
        temp_jsonl_path = ensure_path_safe(
            line_ranges_file.parent / f"{stem}_adjust_temp.jsonl"
        )

        # When force_fresh is set, discard any stale temp JSONL from a
        # previous run so we don't accidentally reuse outdated results.
        if force_fresh and temp_jsonl_path.exists():
            temp_jsonl_path.unlink()
            logger.info("Removed stale temp JSONL (force_fresh): %s", temp_jsonl_path)

        # Staleness validation: only resume from a JSONL whose header
        # fingerprint and settings match the current run.  Legacy JSONLs
        # (without a header) are always discarded.
        _RANGE_ID_PATTERN = re.compile(r"-range-(\d+)$")
        completed_ids: set[int] = set()
        file_mode = "w"

        if not force_fresh and temp_jsonl_path.exists():
            header = read_jsonl_header(temp_jsonl_path)
            if header is not None and validate_jsonl_header(
                header,
                ranges_fingerprint=current_fingerprint,
                boundary_type=boundary_type,
                model_name=self.model_name,
                context_window=self.context_window,
                matching_config=self.matching_config,
                retry_config=self.retry_config,
                prompt_hash=prompt_hash,
            ):
                completed_ids = extract_completed_ids(
                    temp_jsonl_path, id_pattern=_RANGE_ID_PATTERN
                )
                file_mode = "a"
                logger.info(
                    "Resuming adjustment: %d range(s) already processed",
                    len(completed_ids),
                )
            else:
                reason = (
                    "no header (legacy JSONL)"
                    if header is None
                    else "fingerprint/settings mismatch"
                )
                logger.warning(
                    "Discarding stale JSONL (%s): %s", reason, temp_jsonl_path.name
                )
                temp_jsonl_path.unlink()

        adjusted_ranges: list[tuple[int, int]] = []
        ranges_to_delete: list[int] = []  # Track indices of ranges with no content
        total_llm_calls = 0

        # Range-level token-budget gate state. The tracker is a no-op when the
        # daily limit is disabled (try_reserve returns 0, check_and_wait returns
        # True immediately). budget_cancelled records a user-cancelled wait so
        # the line-ranges file is left untouched for a later resume.
        tracker = get_token_tracker()
        budget_cancelled = False

        async with open_extractor(
            api_key=api_key,
            prompt_path=self.prompt_path,
            model=self.model_name,
            model_config_override=self._model_config,
        ) as extractor:
            with JsonlWriter(temp_jsonl_path, mode=file_mode) as writer:
                # Write header as first record on fresh start
                if file_mode == "w":
                    writer.write_record(
                        build_jsonl_header(
                            ranges_fingerprint=current_fingerprint,
                            total_ranges=len(ranges),
                            boundary_type=boundary_type,
                            model_name=self.model_name,
                            context_window=self.context_window,
                            matching_config=self.matching_config,
                            retry_config=self.retry_config,
                            prompt_hash=prompt_hash,
                            context_path=str(context_path) if context_path else None,
                        )
                    )

                for index, original_range in enumerate(ranges, start=1):
                    if index not in selected_indices:
                        # Outside the requested chunk slice: keep unchanged,
                        # no LLM call, no JSONL record.
                        adjusted_ranges.append(original_range)
                        continue

                    if index in completed_ids:
                        # Range already processed in a previous run -- keep
                        # its original_range as placeholder; the final line
                        # ranges are re-derived from the full result set
                        # after overlap removal anyway.
                        adjusted_ranges.append(original_range)
                        continue

                    # Range-level token-budget gate (mirrors the extraction
                    # path). A range may make several LLM calls, so reserve once
                    # before it and release afterward; when the daily budget is
                    # exhausted, wait for the midnight reset and retry the same
                    # range. Ranges run sequentially, so no cross-range
                    # coordination is needed. No-op when the limit is disabled.
                    reserved = tracker.try_reserve()
                    while reserved is None:
                        if not tracker.is_limit_reached():
                            # Remaining budget is positive but below the
                            # per-range estimate; waiting cannot help until
                            # midnight and the estimate may exceed this range's
                            # actual cost, so proceed (overshoot <= one range).
                            break
                        if not await check_and_wait_for_token_limit(logger=logger):
                            budget_cancelled = True
                            break
                        reserved = tracker.try_reserve()
                    if budget_cancelled:
                        break

                    try:
                        result = await self._process_single_range(
                            extractor=extractor,
                            raw_lines=raw_lines,
                            original_range=original_range,
                            range_index=index,
                            boundary_type=boundary_type,
                            context=context,
                        )
                    finally:
                        tracker.release(reserved or 0)
                    total_llm_calls += result.total_llm_calls

                    # Persist to temp JSONL immediately
                    writer.write_record(result.to_jsonl_record(stem))

                    if result.should_delete:
                        logger.warning(
                            "[Range %d] Confirmed no semantic content of type"
                            " '%s' exists; marking for deletion",
                            index,
                            boundary_type,
                        )
                        ranges_to_delete.append(index - 1)
                    else:
                        adjusted_ranges.append(result.adjusted_range)
                        if result.decision.boundary_already_on_target:
                            logger.info(
                                "[Range %d] Boundary already on target;"
                                " kept original %s",
                                index,
                                result.adjusted_range,
                            )
                        elif result.decision.contains_no_semantic_boundary:
                            logger.info(
                                "[Range %d] No semantic boundary detected in"
                                " context; kept original %s",
                                index,
                                result.adjusted_range,
                            )
                        elif result.adjusted_range != result.original_range:
                            logger.info(
                                "[Range %d] Adjusted to %s via boundary '%s'",
                                index,
                                result.adjusted_range,
                                result.decision.semantic_marker,
                            )

        # Budget-interrupted run: leave the line-ranges file and temp JSONL
        # untouched so a later resume run continues from the ranges already
        # completed. Persisting now would truncate the file to the partial set
        # and drop the not-yet-processed ranges.
        if budget_cancelled:
            logger.warning(
                "Line-range readjustment stopped before completion for %s "
                "(daily token budget / user cancel); the line-ranges file was "
                "left unchanged. Re-run with resume to continue.",
                text_file.name,
            )
            return adjusted_ranges

        # If we resumed, re-read the full temp JSONL to reconstruct
        # accurate adjusted_ranges (the placeholders above are stale).
        if completed_ids:
            adjusted_ranges, ranges_to_delete = self._rebuild_ranges_from_jsonl(
                temp_jsonl_path, ranges
            )

        # Spans confirmed to contain no semantic content. Gap enforcement must
        # not re-extend a preceding range across them, or deletion would be
        # silently undone for every interior deleted range.
        deleted_spans = [ranges[i] for i in ranges_to_delete if 0 <= i < len(ranges)]

        adjusted_ranges = self._remove_overlaps(adjusted_ranges)

        if self.max_gap_between_ranges is not None:
            adjusted_ranges = self._enforce_max_gap(adjusted_ranges, deleted_spans)

        if ranges_to_delete:
            logger.info(
                "Deleted %d range(s) with no semantic content: %s",
                len(ranges_to_delete),
                [i + 1 for i in ranges_to_delete],
            )

        if not dry_run:
            self._write_line_ranges(line_ranges_file, adjusted_ranges)

            # Finalize the header only when the JSONL covers every range, so
            # a sliced partial run is never marked complete and later resume
            # runs still process the remaining ranges.
            recorded_ids = extract_completed_ids(
                temp_jsonl_path, id_pattern=_RANGE_ID_PATTERN
            )
            all_indices = set(range(1, total_range_count + 1))
            if recorded_ids >= all_indices:
                # Derive accurate stats from the complete JSONL (covers both
                # newly-processed and resumed ranges) and finalize the header.
                stats = compute_stats_from_jsonl(temp_jsonl_path)
                finalize_jsonl_header(
                    temp_jsonl_path,
                    stats=stats,
                    source_file=line_ranges_file.name,
                    final_fingerprint=compute_ranges_fingerprint(line_ranges_file),
                )
            else:
                logger.info(
                    "Adjustment covered %d of %d range(s) (chunk slice); "
                    "temp JSONL left unfinalized. A later run re-adjusts from "
                    "the updated ranges file.",
                    len(recorded_ids & all_indices),
                    total_range_count,
                )

            # Clean up temp JSONL unless retention requested
            if not retain_temp_jsonl and temp_jsonl_path.exists():
                temp_jsonl_path.unlink()
                logger.info("Removed temp JSONL: %s", temp_jsonl_path)

        return adjusted_ranges

    @staticmethod
    def _rebuild_ranges_from_jsonl(
        temp_jsonl_path: Path,
        original_ranges: Sequence[tuple[int, int]],
    ) -> tuple[list[tuple[int, int]], list[int]]:
        """Rebuild adjusted ranges and deletion list from a complete temp JSONL."""
        from modules.infra.jsonl import read_jsonl_records

        results_by_index: dict[int, dict[str, Any]] = {}
        for record in read_jsonl_records(temp_jsonl_path):
            custom_id = record.get("custom_id", "")
            match = re.search(r"-range-(\d+)$", str(custom_id))
            if match:
                idx = int(match.group(1))
                body = record.get("response", {}).get("body", {})
                results_by_index[idx] = body

        adjusted: list[tuple[int, int]] = []
        deleted: list[int] = []
        for index, original_range in enumerate(original_ranges, start=1):
            body = results_by_index.get(index)
            if body is None:
                # Not in JSONL (shouldn't happen if run completed)
                adjusted.append(original_range)
                continue
            # Defense-in-depth: warn if JSONL record was made for a
            # different range than the current one at this index.
            stored_orig = body.get("original_range")
            if stored_orig is not None and list(stored_orig) != list(original_range):
                logger.warning(
                    "[Range %d] JSONL original_range %s != current %s "
                    "(possible stale JSONL leak)",
                    index,
                    stored_orig,
                    list(original_range),
                )
            if body.get("should_delete", False):
                deleted.append(index - 1)
            else:
                adj = body.get("adjusted_range")
                if adj and len(adj) == 2:
                    adjusted.append((int(adj[0]), int(adj[1])))
                else:
                    adjusted.append(original_range)
        return adjusted, deleted

    async def _process_single_range(
        self,
        *,
        extractor: LLMExtractor,
        raw_lines: Sequence[str],
        original_range: tuple[int, int],
        range_index: int,
        boundary_type: str,
        context: str | None,
    ) -> RangeResult:
        """
        Process a single range to find semantic boundaries.

        Strategy (certainty-based routing):
        1. Check certainty first - if below threshold, retry with different window
        2. Route by high-certainty decision:
           - needs_more_context: Expand window progressively
           - contains_no_semantic_boundary: Verify with broad scan -> delete if
             confirmed
           - Success (marker found): Validate and apply

        Returns:
            A ``RangeResult`` capturing the final decision, adjusted range,
            deletion flag, and a full audit trail of every LLM call.
        """
        total_lines = len(raw_lines)
        windows = list(self._generate_windows(original_range, total_lines))
        decision: BoundaryDecision | None = None
        adjusted_range = original_range

        low_certainty_retries = 0
        context_expansion_attempts = 0

        stop_processing = False
        exhausted_at_low_certainty = False

        failed_marker_history: list[str] = []
        attempts: list[dict[str, Any]] = []
        llm_calls = 0

        for window_idx, window in enumerate(windows):
            marker_mismatch_retries = 0

            while True:
                payload = await self._run_model(
                    extractor=extractor,
                    raw_lines=raw_lines,
                    original_range=original_range,
                    context_window=window,
                    window_index=window_idx,
                    boundary_type=boundary_type,
                    context=context,
                    failed_markers=failed_marker_history,
                )
                decision = BoundaryDecision.from_payload(payload)
                llm_calls += 1

                # ====================================================================
                # PRIORITY 1: Check certainty threshold (applies to all responses)
                # ====================================================================
                if decision.certainty < self.certainty_threshold:
                    attempts.append(
                        {
                            "window": list(window),
                            "window_index": window_idx,
                            "decision_type": "low_certainty",
                            "certainty": decision.certainty,
                            "semantic_marker": decision.semantic_marker,
                            "marker_matched": False,
                        }
                    )
                    low_certainty_retries += 1
                    if low_certainty_retries <= self.max_low_certainty_retries:
                        logger.info(
                            "[Range %d, Window %d] Low certainty"
                            " (%d < %d); retry %d/%d",
                            range_index,
                            window_idx,
                            decision.certainty,
                            self.certainty_threshold,
                            low_certainty_retries,
                            self.max_low_certainty_retries,
                        )
                        break  # Move to next window
                    else:
                        logger.warning(
                            "[Range %d] Certainty remains low after"
                            " %d retries (best: %d)",
                            range_index,
                            self.max_low_certainty_retries,
                            decision.certainty,
                        )
                        exhausted_at_low_certainty = True
                        stop_processing = True
                        break

                # ====================================================================
                # High certainty response - route by decision type
                # ====================================================================

                # ROUTE 0: Boundary already at correct position
                if decision.boundary_already_on_target:
                    attempts.append(
                        {
                            "window": list(window),
                            "window_index": window_idx,
                            "decision_type": "already_on_target",
                            "certainty": decision.certainty,
                            "semantic_marker": decision.semantic_marker,
                            "marker_matched": False,
                        }
                    )
                    logger.info(
                        "[Range %d, Window %d] Boundary already on target "
                        "(certainty: %d); keeping original range %s",
                        range_index,
                        window_idx,
                        decision.certainty,
                        original_range,
                    )
                    stop_processing = True
                    break

                # ROUTE 1: Model requests more context -> Expand window
                elif decision.needs_more_context:
                    attempts.append(
                        {
                            "window": list(window),
                            "window_index": window_idx,
                            "decision_type": "needs_more_context",
                            "certainty": decision.certainty,
                            "semantic_marker": decision.semantic_marker,
                            "marker_matched": False,
                        }
                    )
                    context_expansion_attempts += 1
                    if (
                        context_expansion_attempts
                        <= self.max_context_expansion_attempts
                    ):
                        logger.info(
                            "[Range %d, Window %d] Requests more context"
                            " (certainty: %d); expansion %d/%d",
                            range_index,
                            window_idx,
                            decision.certainty,
                            context_expansion_attempts,
                            self.max_context_expansion_attempts,
                        )
                        break  # Move to next (larger) window
                    else:
                        logger.warning(
                            "[Range %d] Max context expansions reached (%d);"
                            " keeping original range",
                            range_index,
                            self.max_context_expansion_attempts,
                        )
                        stop_processing = True
                        break

                # ROUTE 2: Model confident no content exists -> Verify with broad scan
                elif decision.contains_no_semantic_boundary:
                    attempts.append(
                        {
                            "window": list(window),
                            "window_index": window_idx,
                            "decision_type": "no_semantic_boundary",
                            "certainty": decision.certainty,
                            "semantic_marker": decision.semantic_marker,
                            "marker_matched": False,
                        }
                    )
                    logger.info(
                        "[Range %d, Window %d] High-certainty no-content response"
                        " (certainty: %d); triggering verification",
                        range_index,
                        window_idx,
                        decision.certainty,
                    )

                    if self.delete_ranges_with_no_content:
                        (
                            should_delete,
                            reanchored_range,
                            verify_attempts,
                        ) = await self._verify_no_content(
                            extractor=extractor,
                            raw_lines=raw_lines,
                            original_range=original_range,
                            range_index=range_index,
                            boundary_type=boundary_type,
                            context=context,
                        )
                        llm_calls += len(verify_attempts)
                        attempts.extend(verify_attempts)
                        if should_delete:
                            return RangeResult(
                                range_index=range_index,
                                original_range=original_range,
                                adjusted_range=original_range,
                                should_delete=True,
                                decision=decision,
                                attempts=attempts,
                                total_llm_calls=llm_calls,
                            )
                        if reanchored_range:
                            adjusted_range = reanchored_range
                            logger.info(
                                "[Range %d] Re-anchored to %s using"
                                " verify-interior marker",
                                range_index,
                                reanchored_range,
                            )
                        else:
                            logger.info(
                                "[Range %d] Verification found content but could"
                                " not resolve marker; keeping range",
                                range_index,
                            )
                        stop_processing = True
                        break
                    else:
                        logger.info(
                            "[Range %d] No content detected but deletion disabled;"
                            " keeping range",
                            range_index,
                        )
                        stop_processing = True
                        break

                # ROUTE 3: Success -> Marker found with high certainty, validate and
                # apply
                else:
                    candidate_range = self._validate_and_apply_decision(
                        decision=decision,
                        raw_lines=raw_lines,
                        context_window=window,
                        fallback_range=original_range,
                    )
                    if candidate_range:
                        adjusted_range = candidate_range
                        attempts.append(
                            {
                                "window": list(window),
                                "window_index": window_idx,
                                "decision_type": "marker_found",
                                "certainty": decision.certainty,
                                "semantic_marker": decision.semantic_marker,
                                "marker_matched": True,
                            }
                        )
                        logger.info(
                            "[Range %d] Successfully adjusted to %s using"
                            " marker '%s' (certainty: %d)",
                            range_index,
                            adjusted_range,
                            decision.semantic_marker,
                            decision.certainty,
                        )
                        stop_processing = True
                        break

                    marker_text = (decision.semantic_marker or "").strip()
                    attempts.append(
                        {
                            "window": list(window),
                            "window_index": window_idx,
                            "decision_type": "marker_mismatch",
                            "certainty": decision.certainty,
                            "semantic_marker": decision.semantic_marker,
                            "marker_matched": False,
                        }
                    )
                    if marker_text and marker_text not in failed_marker_history:
                        failed_marker_history.append(marker_text)
                    marker_mismatch_retries += 1

                    if marker_mismatch_retries <= self.max_marker_mismatch_retries:
                        logger.info(
                            "[Range %d, Window %d] Semantic marker '%s'"
                            " not matched; retry %d/%d",
                            range_index,
                            window_idx,
                            marker_text or "<empty>",
                            marker_mismatch_retries,
                            self.max_marker_mismatch_retries,
                        )
                        continue  # Retry same window with additional guidance

                    logger.warning(
                        "[Range %d, Window %d] Semantic marker not matched"
                        " after %d retries; trying next window",
                        range_index,
                        window_idx,
                        self.max_marker_mismatch_retries,
                    )
                    break

            if stop_processing:
                break

        # No windows examined or all attempts exhausted
        if decision is None:
            decision = BoundaryDecision(
                contains_no_semantic_boundary=True,
                needs_more_context=False,
                certainty=0,
            )
            logger.info(
                "[Range %d] No windows available; keeping original range %s",
                range_index,
                original_range,
            )

        # Exhaustion fallback: verify interior before keeping unchanged
        if (
            not stop_processing or exhausted_at_low_certainty
        ) and self.delete_ranges_with_no_content:
            logger.info(
                "[Range %d] All boundary windows exhausted; running fallback"
                " interior verification",
                range_index,
            )
            fb_delete, fb_reanchored, fb_attempts = await self._verify_no_content(
                extractor=extractor,
                raw_lines=raw_lines,
                original_range=original_range,
                range_index=range_index,
                boundary_type=boundary_type,
                context=context,
            )
            llm_calls += len(fb_attempts)
            attempts.extend(fb_attempts)
            if fb_delete:
                return RangeResult(
                    range_index=range_index,
                    original_range=original_range,
                    adjusted_range=original_range,
                    should_delete=True,
                    decision=decision,
                    attempts=attempts,
                    total_llm_calls=llm_calls,
                )
            if fb_reanchored:
                adjusted_range = fb_reanchored
                logger.info(
                    "[Range %d] Exhaustion fallback re-anchored to %s",
                    range_index,
                    fb_reanchored,
                )

        return RangeResult(
            range_index=range_index,
            original_range=original_range,
            adjusted_range=adjusted_range,
            should_delete=False,
            decision=decision,
            attempts=attempts,
            total_llm_calls=llm_calls,
        )

    async def _verify_no_content(
        self,
        *,
        extractor: LLMExtractor,
        raw_lines: Sequence[str],
        original_range: tuple[int, int],
        range_index: int,
        boundary_type: str,
        context: str | None,
    ) -> tuple[bool, tuple[int, int] | None, list[dict[str, Any]]]:
        """
        Verify that no semantic content of the required type exists within the range.

        Called after the initial model assessment returned
        contains_no_semantic_boundary. Scans the range interior to confirm the
        absence of content before deletion.
        When content is found with a resolvable marker, returns a re-anchored range
        so the caller can adjust the boundary to where the content actually is.

        Returns:
            Tuple of (should_delete, reanchored_range_or_None, verification_attempts).
        """
        start, end = original_range
        verification_attempts: list[dict[str, Any]] = []

        # Consecutive full-coverage scan windows within the range. Chunk-sized
        # ranges fit in one window; longer ranges are covered gaplessly.
        scan_windows: list[tuple[int, int]] = []
        window_start = start
        while window_start <= end:
            window_end = min(end, window_start + MAX_VERIFY_WINDOW_LINES - 1)
            scan_windows.append((window_start, window_end))
            window_start = window_end + 1

        logger.info(
            "[Range %d] Verifying no content: scanning %d interior window(s)"
            " within [%d-%d]",
            range_index,
            len(scan_windows),
            start,
            end,
        )

        for idx, (scan_start, scan_end) in enumerate(scan_windows):
            payload = await self._run_model(
                extractor=extractor,
                raw_lines=raw_lines,
                original_range=original_range,
                context_window=(scan_start, scan_end),
                window_index=-(idx + 1),
                boundary_type=boundary_type,
                context=context,
            )
            decision = BoundaryDecision.from_payload(payload)
            found_content = not decision.contains_no_semantic_boundary
            verification_attempts.append(
                {
                    "window": [scan_start, scan_end],
                    "window_index": -(idx + 1),
                    "decision_type": f"verify_interior_{idx + 1}",
                    "certainty": decision.certainty,
                    "semantic_marker": decision.semantic_marker,
                    "found_content": found_content,
                }
            )

            # Any content signal blocks deletion, marker or not: the model may
            # legitimately report content via boundary_already_on_target or
            # needs_more_context with an empty marker.
            if found_content:
                reanchored = None
                if decision.semantic_marker:
                    # Try to resolve the marker for re-anchoring; the new
                    # start must stay within the range.
                    matched_line = self._match_boundary_text(
                        marker=decision.semantic_marker,
                        raw_lines=raw_lines,
                        search_start=scan_start,
                        search_end=scan_end,
                        nearest_to=start,
                    )
                    if matched_line is not None and start <= matched_line <= end:
                        reanchored = (matched_line, end)
                        logger.info(
                            "[Range %d] Interior scan found content at line %d"
                            " via marker '%s'; re-anchoring",
                            range_index,
                            matched_line,
                            decision.semantic_marker,
                        )
                if reanchored is None:
                    logger.info(
                        "[Range %d] Interior scan found content (marker %s);"
                        " preserving range unchanged",
                        range_index,
                        repr(decision.semantic_marker or "<none>"),
                    )
                return False, reanchored, verification_attempts

            # A no-content verdict below the certainty threshold is not good
            # enough to destroy a range: abort deletion and keep it.
            if decision.certainty < self.certainty_threshold:
                logger.info(
                    "[Range %d] No-content verdict below certainty threshold"
                    " (%d < %d); keeping range",
                    range_index,
                    decision.certainty,
                    self.certainty_threshold,
                )
                return False, None, verification_attempts

        # All interior scans confirmed no content with high certainty
        logger.info(
            "[Range %d] Verified no semantic content in range interior;"
            " confirming deletion",
            range_index,
        )
        return True, None, verification_attempts

    async def _run_model(
        self,
        *,
        extractor: LLMExtractor,
        raw_lines: Sequence[str],
        original_range: tuple[int, int],
        context_window: tuple[int, int],
        window_index: int,
        boundary_type: str,
        context: str | None,
        failed_markers: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        start, _ = original_range
        context_start, context_end = context_window
        marker_line = start if context_start <= start <= context_end else None
        context_block = self._format_context(
            raw_lines, context_start, context_end, marker_line=marker_line
        )

        chunk_text = f"Input text:\n\n{context_block}\n"

        # Failed-marker retry guidance goes into the user message, not the
        # system prompt, so the cache_control-annotated system block stays
        # byte-stable across retries (Anthropic prompt caching).
        if failed_markers:
            sanitized_failures = [
                marker.strip() for marker in failed_markers if marker and marker.strip()
            ]
            if sanitized_failures:
                bullet_list = "\n".join(f"- {marker}" for marker in sanitized_failures)
                chunk_text += (
                    "\nThe following semantic markers previously failed to match"
                    " the source text. Do not reuse any of them. Provide a new"
                    " 5-15 character substring that excludes these markers:\n"
                    f"{bullet_list}\n"
                )

        # Render system prompt with unified context, schema, and the semantic
        # unit type (the schema name), so the model knows what boundary kind
        # to look for even when no context file is resolved.
        system_prompt = render_prompt_with_schema(
            self.prompt_template,
            SEMANTIC_BOUNDARY_SCHEMA["schema"],
            inject_schema=True,
            context=context,
        )
        system_prompt = system_prompt.replace("{{BOUNDARY_TYPE}}", boundary_type)

        response_payload = await process_text_chunk(
            text_chunk=chunk_text,
            extractor=extractor,
            system_message=system_prompt,
            json_schema=SEMANTIC_BOUNDARY_SCHEMA,
            enable_cache_control=self._enable_cache_control,
        )

        raw_output = response_payload.get("output_text", "")
        parsed = self._coerce_json(raw_output)
        if parsed is None:
            logger.warning(
                "Model response for window %d (%d-%d) could not be parsed as"
                " JSON; treating as unsure. Raw: %s",
                window_index,
                context_start,
                context_end,
                raw_output,
            )
            return {
                "contains_no_semantic_boundary": False,
                "needs_more_context": False,
                "boundary_already_on_target": False,
                "certainty": 0,
                "semantic_marker": "",
            }
        return parsed

    def _validate_and_apply_decision(
        self,
        *,
        decision: BoundaryDecision,
        raw_lines: Sequence[str],
        context_window: tuple[int, int],
        fallback_range: tuple[int, int],
    ) -> tuple[int, int] | None:
        if decision.semantic_marker is None or not decision.semantic_marker.strip():
            return None

        context_start, context_end = context_window
        matched_line = self._match_boundary_text(
            marker=decision.semantic_marker,
            raw_lines=raw_lines,
            search_start=context_start,
            search_end=context_end,
            nearest_to=fallback_range[0],
        )
        if matched_line is None:
            return None

        # Only the start boundary is adjusted; the original end is kept. A
        # match beyond the end would invert the range, so reject it and let
        # the mismatch-retry loop request another marker.
        new_start = matched_line
        new_end = fallback_range[1]
        if new_start > new_end:
            logger.warning(
                "Semantic marker '%s' matched line %d beyond range end %d;"
                " rejecting to avoid an inverted range",
                decision.semantic_marker,
                matched_line,
                new_end,
            )
            return None

        return new_start, new_end

    def _generate_windows(
        self,
        original_range: tuple[int, int],
        total_lines: int,
    ) -> Generator[tuple[int, int], None, None]:
        """Yield geometrically growing context windows around the range start.

        The radius doubles per window so the configured retry budgets are
        actually reachable: both a low-certainty retry and a context-expansion
        request advance to the next window, so the supply covers the sum of
        both budgets. Generation stops early once a window spans the whole
        document (further growth would only repeat it).
        """
        start, _ = original_range
        radius = self.context_window
        max_windows = (
            1 + self.max_context_expansion_attempts + self.max_low_certainty_retries
        )

        previous: tuple[int, int] | None = None
        for step in range(max_windows):
            expanded_radius = radius * (2**step)
            bounded = (
                max(1, start - expanded_radius),
                min(total_lines, start + expanded_radius),
            )
            if bounded[1] < bounded[0] or bounded == previous:
                break
            previous = bounded
            yield bounded
            if bounded == (1, total_lines):
                break

    def _format_context(
        self,
        raw_lines: Sequence[str],
        context_start: int,
        context_end: int,
        marker_line: int | None = None,
    ) -> str:
        """Render the window's lines, inserting the chunk-start sentinel.

        Without the sentinel the model has no way of knowing where the current
        chunk boundary sits inside the visible text, making decisions such as
        ``boundary_already_on_target`` guesswork.
        """
        snippet: list[str] = []
        for line_number in range(context_start, context_end + 1):
            if marker_line is not None and line_number == marker_line:
                snippet.append(BOUNDARY_SENTINEL)
            text = raw_lines[line_number - 1].rstrip("\n")
            snippet.append(text)
        return "\n".join(snippet)

    def _normalize_text(self, text: str) -> str:
        """Apply normalization rules to text according to matching configuration."""
        normalized = text

        # Strip leading/trailing whitespace
        normalized = normalized.strip()

        # Normalize whitespace (collapse multiple spaces)
        if self.normalize_whitespace:
            normalized = re.sub(r"\s+", " ", normalized)

        # Normalize diacritics (remove accents, umlauts, etc.)
        if self.normalize_diacritics:
            # Decompose unicode characters and filter out combining marks
            normalized = "".join(
                char
                for char in unicodedata.normalize("NFD", normalized)
                if unicodedata.category(char) != "Mn"
            )

        # Case normalization
        if not self.case_sensitive:
            normalized = normalized.lower()

        # Strip punctuation
        if self.strip_punctuation:
            # Remove common punctuation while preserving alphanumeric and spaces
            normalized = re.sub(r"[^\w\s]", "", normalized)

        return normalized

    def _collect_normalized_matches(
        self,
        raw_lines: Sequence[str],
        needle: str,
        search_range: tuple[int, int],
    ) -> list[int]:
        """
        Collect line numbers where the needle matches, applying configured
        normalization.

        Supports both exact line matching and substring matching based on
        configuration.
        """
        start_line, end_line = search_range
        end_line = min(len(raw_lines), max(start_line, end_line))
        matches: list[int] = []

        normalized_needle = self._normalize_text(needle)

        for line_number in range(start_line, end_line + 1):
            line_text = raw_lines[line_number - 1]
            normalized_line = self._normalize_text(line_text)

            if self.allow_substring_match:
                # Check if needle appears as substring in the line
                if normalized_needle in normalized_line:
                    matches.append(line_number)
            else:
                # Require exact match (after normalization)
                if normalized_line == normalized_needle:
                    matches.append(line_number)

        return matches

    def _match_boundary_text(
        self,
        *,
        marker: str,
        raw_lines: Sequence[str],
        search_start: int,
        search_end: int,
        nearest_to: int,
    ) -> int | None:
        """
        Match a semantic marker in the source lines using configured
        normalization.

        The search is bounded: it first tries the given window, then a single
        expansion by ``context_window`` lines per side. A whole-document
        fallback is deliberately avoided — a distant unique match would move
        the boundary far from the range it belongs to. Ambiguous matches
        resolve to the candidate nearest ``nearest_to`` (ties break to the
        earlier line). Markers shorter than ``min_substring_length`` are
        rejected so the mismatch-retry loop can request a longer one.
        """
        needle = (marker or "").strip()
        if not needle:
            return None

        if len(needle) < self.min_substring_length:
            logger.debug(
                "Rejecting semantic marker '%s': shorter than"
                " min_substring_length=%d",
                needle,
                self.min_substring_length,
            )
            return None

        total_lines = len(raw_lines)
        window = (max(1, search_start), min(total_lines, search_end))
        matches = self._collect_normalized_matches(raw_lines, needle, window)

        if not matches:
            expanded = (
                max(1, window[0] - self.context_window),
                min(total_lines, window[1] + self.context_window),
            )
            if expanded != window:
                matches = self._collect_normalized_matches(
                    raw_lines, needle, expanded
                )

        if not matches:
            return None
        if len(matches) > 1:
            logger.debug(
                "Multiple candidates for semantic marker '%s' at lines %s;"
                " choosing the one nearest line %d",
                needle,
                matches,
                nearest_to,
            )
        return min(matches, key=lambda line: (abs(line - nearest_to), line))

    def _coerce_json(self, raw_output: str) -> dict[str, Any] | None:
        text = raw_output.strip()
        if not text:
            return None

        code_block_pattern = re.compile(
            r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE
        )
        candidates = [m.group(1) for m in code_block_pattern.finditer(text)]
        if not candidates:
            candidates = [text]

        for candidate in candidates:
            stripped = candidate.strip()
            if not stripped:
                continue
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                continue
        return None

    def _infer_line_ranges_file(self, text_file: Path) -> Path:
        return text_file.with_name(f"{text_file.stem}_line_ranges.txt")

    def _write_line_ranges(
        self, line_ranges_file: Path, ranges: Sequence[tuple[int, int]]
    ) -> None:
        safe_line_ranges_file = ensure_path_safe(line_ranges_file)
        with safe_line_ranges_file.open("w", encoding="utf-8") as handle:
            for start, end in ranges:
                handle.write(f"({start}, {end})\n")

    @staticmethod
    def _remove_overlaps(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """
        Remove overlaps from a list of line ranges by adjusting end boundaries.

        When range[i].end >= range[i+1].start, we have an overlap.
        This is resolved by setting range[i].end = range[i+1].start - 1.

        This preserves the semantic start boundaries identified by the LLM
        while ensuring no overlaps exist. Gaps are acceptable and expected.

        Args:
            ranges: List of (start, end) tuples, assumed to be in sequential order

        Returns:
            List of non-overlapping (start, end) tuples
        """
        if not ranges:
            return []

        annotated = [
            {
                "start": start,
                "end": end,
                "original_start": start,
                "original_end": end,
                "original_index": idx,
            }
            for idx, (start, end) in enumerate(ranges)
        ]

        # Sort by start (stable on original index) so out-of-order ranges are
        # processed safely.
        annotated.sort(key=lambda item: (item["start"], item["original_index"]))

        processed: list[dict[str, int]] = []

        for entry in annotated:
            current_start = entry["start"]
            current_end = entry["end"]
            original_start = entry["original_start"]
            original_end = entry["original_end"]

            if current_start > current_end:
                logger.warning(
                    "Range (%d, %d) has end before start; clamping end to %d"
                    " before overlap removal.",
                    original_start,
                    original_end,
                    current_start,
                )
                current_end = current_start

            if processed:
                previous = processed[-1]

                if previous["end"] >= current_start:
                    trimmed_prev_end = min(
                        previous["end"], max(previous["start"], current_start - 1)
                    )

                    if trimmed_prev_end < previous["end"]:
                        logger.info(
                            "Overlap removed: range (%d, %d) trimmed to"
                            " (%d, %d) to respect next range (%d, %d)",
                            previous["original_start"],
                            previous["original_end"],
                            previous["start"],
                            trimmed_prev_end,
                            original_start,
                            original_end,
                        )
                        previous["end"] = trimmed_prev_end

                if previous["end"] >= current_start:
                    shift_amount = previous["end"] + 1 - current_start
                    current_start = previous["end"] + 1
                    if shift_amount > 0:
                        logger.info(
                            "Adjusted start of range (%d, %d) forward by %d"
                            " line(s) after trimming to avoid overlap.",
                            original_start,
                            original_end,
                            shift_amount,
                        )

            if current_start > current_end:
                logger.warning(
                    "Range (%d, %d) would invert after overlap handling;"
                    " clamping end to %d.",
                    original_start,
                    original_end,
                    current_start,
                )
                current_end = current_start

            processed.append(
                {
                    "start": current_start,
                    "end": current_end,
                    "original_start": original_start,
                    "original_end": original_end,
                    "original_index": entry["original_index"],
                }
            )

        return [(item["start"], item["end"]) for item in processed]

    def _enforce_max_gap(
        self,
        ranges: list[tuple[int, int]],
        deleted_spans: Sequence[tuple[int, int]] = (),
    ) -> list[tuple[int, int]]:
        """
        Ensure gaps between consecutive ranges do not exceed configured maximum.

        Ranges are assumed to be sorted and non-overlapping. When a gap exceeds
        ``max_gap_between_ranges``, the previous range is extended forward to
        reduce the gap while still ending before the next range.

        ``deleted_spans`` are ranges confirmed to contain no semantic content;
        the extension never reaches into them, otherwise deletion would be
        silently undone and the no-content lines re-extracted.
        """
        if not ranges:
            return []

        max_gap = self.max_gap_between_ranges
        if max_gap is None:
            return list(ranges)

        enforced: list[tuple[int, int]] = []

        for index, (start, end) in enumerate(ranges):
            if index == 0:
                enforced.append((start, end))
                continue

            prev_start, prev_end = enforced[-1]
            gap = start - prev_end - 1

            if gap > max_gap:
                new_prev_end = start - max_gap - 1
                # Cap the extension before any deleted span in the gap.
                for span_start, span_end in deleted_spans:
                    if span_end > prev_end and span_start <= new_prev_end:
                        new_prev_end = min(new_prev_end, span_start - 1)
                if new_prev_end > prev_end:
                    logger.info(
                        "Gap of %d lines detected between ranges (%d, %d) and"
                        " (%d, %d); extending previous range to (%d, %d)",
                        gap,
                        prev_start,
                        prev_end,
                        start,
                        end,
                        prev_start,
                        new_prev_end,
                    )
                    enforced[-1] = (prev_start, new_prev_end)

            enforced.append((start, end))

        return enforced


async def adjust_line_ranges_for_paths(
    *,
    model_config: dict[str, Any],
    text_files: Iterable[Path],
    context_window: int,
    prompt_path: Path | None = None,
    dry_run: bool = False,
    boundary_type: str,
) -> dict[Path, list[tuple[int, int]]]:
    """Adjust line ranges for multiple files.

    Context is resolved automatically using hierarchical fallback.
    """
    readjuster = LineRangeReadjuster(
        model_config,
        context_window=context_window,
        prompt_path=prompt_path,
    )

    results: dict[Path, list[tuple[int, int]]] = {}
    for text_file in text_files:
        ranges = await readjuster.ensure_adjusted_line_ranges(
            text_file=text_file,
            dry_run=dry_run,
            boundary_type=boundary_type,
        )
        results[text_file] = ranges
    return results


def run_adjustment_sync(
    *,
    model_config: dict[str, Any],
    text_file: Path,
    context_window: int,
    prompt_path: Path | None = None,
    dry_run: bool = False,
    boundary_type: str,
) -> list[tuple[int, int]]:
    """Run line range adjustment synchronously.

    Context is resolved automatically using hierarchical fallback.
    """

    async def _runner() -> list[tuple[int, int]]:
        readjuster = LineRangeReadjuster(
            model_config,
            context_window=context_window,
            prompt_path=prompt_path,
        )
        return await readjuster.ensure_adjusted_line_ranges(
            text_file=text_file,
            dry_run=dry_run,
            boundary_type=boundary_type,
        )

    return asyncio.run(_runner())
