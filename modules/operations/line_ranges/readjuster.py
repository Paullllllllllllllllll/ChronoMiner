from __future__ import annotations

"""
Line range readjustment module for semantic boundary detection.

Supports multiple LLM providers via LangChain:
- OpenAI (default)
- Anthropic (Claude)
- Google (Gemini)
- OpenRouter (multi-provider access)
"""

import asyncio
import hashlib
import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from modules.core.context_resolver import resolve_context_for_readjustment
from modules.core.jsonl_utils import JsonlWriter, extract_completed_ids
from modules.core.resume import write_adjustment_marker
from modules.core.text_utils import TextProcessor, load_line_ranges
from modules.llm.openai_utils import open_extractor, process_text_chunk
from modules.llm.langchain_provider import ProviderConfig
from modules.llm.model_capabilities import detect_capabilities
from modules.llm.prompt_utils import load_prompt_template, render_prompt_with_schema
from modules.core.path_utils import ensure_path_safe

logger = logging.getLogger(__name__)


SEMANTIC_BOUNDARY_SCHEMA: Dict[str, Any] = {
    "name": "SemanticBoundaryResponse",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "contains_no_semantic_boundary": {
                "type": "boolean",
                "description": "Set to true when the provided text contains NO content of the required semantic type at all. Use this when you are confident no relevant content exists anywhere in the visible context.",
            },
            "needs_more_context": {
                "type": "boolean",
                "description": "Set to true when you believe the semantic boundary exists somewhere around the visible text but you need to see more surrounding content to accurately identify it.",
            },
            "certainty": {
                "type": "integer",
                "description": "Your confidence level in this response as an integer from 0-100. Use 0-40 for low confidence, 41-70 for moderate confidence, 71-100 for high confidence. This applies to whatever decision you make (boundary found, no content, or needs more context).",
            },
            "semantic_marker": {
                "type": "string",
                "description": "A precise 5-15 character verbatim substring that marks the semantic boundary. Leave empty if contains_no_semantic_boundary or needs_more_context is true.",
            },
        },
        "required": ["contains_no_semantic_boundary", "needs_more_context", "certainty", "semantic_marker"],
        "additionalProperties": False,
    },
}

@dataclass
class BoundaryDecision:
    contains_no_semantic_boundary: bool
    needs_more_context: bool
    certainty: int
    semantic_marker: Optional[str] = None

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "BoundaryDecision":
        return cls(
            contains_no_semantic_boundary=bool(payload.get("contains_no_semantic_boundary", False)),
            needs_more_context=bool(payload.get("needs_more_context", False)),
            certainty=int(payload.get("certainty", 0)),
            semantic_marker=payload.get("semantic_marker") or None,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict for JSONL storage."""
        return {
            "contains_no_semantic_boundary": self.contains_no_semantic_boundary,
            "needs_more_context": self.needs_more_context,
            "certainty": self.certainty,
            "semantic_marker": self.semantic_marker,
        }


@dataclass
class RangeResult:
    """Result of processing a single line range, including an audit trail."""

    range_index: int
    original_range: Tuple[int, int]
    adjusted_range: Tuple[int, int]
    should_delete: bool
    decision: BoundaryDecision
    attempts: List[Dict[str, Any]] = field(default_factory=list)
    total_llm_calls: int = 0

    def to_jsonl_record(self, stem: str) -> Dict[str, Any]:
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
    """Adjust `_line_ranges.txt` files so that chunk boundaries align with semantic boundaries."""

    def __init__(
        self,
        model_config: Dict[str, Any],
        *,
        context_window: int = 6,
        prompt_path: Optional[Path] = None,
        matching_config: Optional[Dict[str, Any]] = None,
        retry_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        transcription_cfg = model_config.get("extraction_model", {})
        model_name: str = transcription_cfg.get("name", "")
        if not model_name:
            raise ValueError("extraction_model.name must be configured to use LineRangeReadjuster")

        self.model_name = model_name
        self.context_window = max(1, int(context_window))
        self.prompt_path = prompt_path or Path("prompts/semantic_boundary_prompt.txt")
        self.prompt_template = load_prompt_template(self.prompt_path)
        self.text_processor = TextProcessor()
        self._enable_cache_control = detect_capabilities(model_name).supports_prompt_caching
        
        # Load matching configuration with defaults
        self.matching_config = matching_config or {}
        self.normalize_whitespace = self.matching_config.get("normalize_whitespace", True)
        self.case_sensitive = self.matching_config.get("case_sensitive", False)
        self.normalize_diacritics = self.matching_config.get("normalize_diacritics", True)
        self.strip_punctuation = self.matching_config.get("strip_punctuation", False)
        self.allow_substring_match = self.matching_config.get("allow_substring_match", True)
        self.min_substring_length = self.matching_config.get("min_substring_length", 8)
        
        # Load retry configuration with defaults
        self.retry_config = retry_config or {}
        self.certainty_threshold = self.retry_config.get("certainty_threshold", 70)
        self.max_low_certainty_retries = self.retry_config.get("max_low_certainty_retries", 3)
        self.max_context_expansion_attempts = self.retry_config.get("max_context_expansion_attempts", 3)
        self.delete_ranges_with_no_content = self.retry_config.get("delete_ranges_with_no_content", True)
        self.scan_range_multiplier = self.retry_config.get("scan_range_multiplier", 3)
        self.max_marker_mismatch_retries = self.retry_config.get("max_marker_mismatch_retries", 2)

        max_gap_setting = self.retry_config.get("max_gap_between_ranges")
        if max_gap_setting is None:
            self.max_gap_between_ranges: Optional[int] = None
        else:
            try:
                self.max_gap_between_ranges = max(0, int(max_gap_setting))
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid max_gap_between_ranges value '%s'; disabling gap enforcement.",
                    max_gap_setting,
                )
                self.max_gap_between_ranges = None

    async def ensure_adjusted_line_ranges(
        self,
        *,
        text_file: Path,
        line_ranges_file: Optional[Path] = None,
        dry_run: bool = False,
        boundary_type: Optional[str] = None,
        retain_temp_jsonl: bool = True,
        force_fresh: bool = False,
        first_n_chunks: Optional[int] = None,
        last_n_chunks: Optional[int] = None,
    ) -> List[Tuple[int, int]]:
        """Ensure the provided line ranges align with semantic boundaries."""
        text_file = text_file.resolve()
        if line_ranges_file is None:
            line_ranges_file = self._infer_line_ranges_file(text_file)
        line_ranges_file = line_ranges_file.resolve()

        if not line_ranges_file.exists():
            raise FileNotFoundError(f"Line ranges file not found: {line_ranges_file}")

        if not boundary_type:
            raise ValueError("boundary_type must be provided when readjusting line ranges")

        ranges = load_line_ranges(line_ranges_file)
        if not ranges:
            logger.warning("No ranges found in %s", line_ranges_file)
            return []

        # Apply chunk slicing if requested
        if first_n_chunks is not None:
            logger.info("Slicing to first %d range(s) of %d total", first_n_chunks, len(ranges))
            ranges = ranges[:first_n_chunks]
        elif last_n_chunks is not None:
            logger.info("Slicing to last %d range(s) of %d total", last_n_chunks, len(ranges))
            ranges = ranges[-last_n_chunks:]

        safe_text_file = ensure_path_safe(text_file)
        with safe_text_file.open("r", encoding="utf-8") as handle:
            raw_lines = handle.readlines()

        # Detect provider from model name and get appropriate API key
        provider = ProviderConfig._detect_provider(self.model_name)
        api_key = ProviderConfig._get_api_key(provider)
        if not api_key:
            raise RuntimeError(f"API key not found for provider {provider}. Set the appropriate environment variable.")

        # Resolve unified context using hierarchical resolution
        context, context_path = resolve_context_for_readjustment(
            text_file=text_file,
        )

        if context_path:
            logger.info(f"Using line ranges context from: {context_path}")
        else:
            logger.debug(f"No adjust_context found for '{text_file.name}'")

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

        _RANGE_ID_PATTERN = re.compile(r"-range-(\d+)$")
        completed_ids = extract_completed_ids(
            temp_jsonl_path, id_pattern=_RANGE_ID_PATTERN
        )
        file_mode = "a" if completed_ids else "w"
        if completed_ids:
            logger.info(
                "Resuming adjustment: %d range(s) already processed",
                len(completed_ids),
            )

        adjusted_ranges: List[Tuple[int, int]] = []
        ranges_to_delete: List[int] = []  # Track indices of ranges with no content
        total_llm_calls = 0
        ranges_adjusted_count = 0
        ranges_kept_original_count = 0

        async with open_extractor(
            api_key=api_key,
            prompt_path=self.prompt_path,
            model=self.model_name,
        ) as extractor:
            with JsonlWriter(temp_jsonl_path, mode=file_mode) as writer:
                for index, original_range in enumerate(ranges, start=1):
                    if index in completed_ids:
                        # Range already processed in a previous run -- keep
                        # its original_range as placeholder; the final line
                        # ranges are re-derived from the full result set
                        # after overlap removal anyway.
                        adjusted_ranges.append(original_range)
                        continue

                    result = await self._process_single_range(
                        extractor=extractor,
                        raw_lines=raw_lines,
                        original_range=original_range,
                        range_index=index,
                        boundary_type=boundary_type,
                        context=context,
                    )
                    total_llm_calls += result.total_llm_calls

                    # Persist to temp JSONL immediately
                    writer.write_record(result.to_jsonl_record(stem))

                    if result.should_delete:
                        logger.warning(
                            "[Range %d] Confirmed no semantic content of type '%s' exists; marking for deletion",
                            index,
                            boundary_type,
                        )
                        ranges_to_delete.append(index - 1)
                    else:
                        adjusted_ranges.append(result.adjusted_range)
                        if result.decision.contains_no_semantic_boundary:
                            ranges_kept_original_count += 1
                            logger.info(
                                "[Range %d] No semantic boundary detected in context; kept original %s",
                                index,
                                result.adjusted_range,
                            )
                        elif result.adjusted_range != result.original_range:
                            ranges_adjusted_count += 1
                            logger.info(
                                "[Range %d] Adjusted to %s via boundary '%s'",
                                index,
                                result.adjusted_range,
                                result.decision.semantic_marker,
                            )
                        else:
                            ranges_kept_original_count += 1

        # If we resumed, re-read the full temp JSONL to reconstruct
        # accurate adjusted_ranges (the placeholders above are stale).
        if completed_ids:
            adjusted_ranges, ranges_to_delete = self._rebuild_ranges_from_jsonl(
                temp_jsonl_path, ranges
            )

        adjusted_ranges = self._remove_overlaps(adjusted_ranges)

        if self.max_gap_between_ranges is not None:
            adjusted_ranges = self._enforce_max_gap(adjusted_ranges)

        if ranges_to_delete:
            logger.info(
                "Deleted %d range(s) with no semantic content: %s",
                len(ranges_to_delete),
                [i + 1 for i in ranges_to_delete],
            )

        if not dry_run:
            self._write_line_ranges(line_ranges_file, adjusted_ranges)

            # Compute prompt hash for reproducibility
            prompt_hash = hashlib.sha256(
                self.prompt_template.encode("utf-8")
            ).hexdigest()

            write_adjustment_marker(
                line_ranges_file,
                boundary_type=boundary_type,
                context_window=self.context_window,
                model_name=self.model_name,
                matching_config=self.matching_config or None,
                retry_config=self.retry_config or None,
                prompt_hash=prompt_hash,
                context_path=str(context_path) if context_path else None,
                total_ranges=len(ranges),
                ranges_adjusted=ranges_adjusted_count,
                ranges_deleted=len(ranges_to_delete),
                ranges_kept_original=ranges_kept_original_count,
                total_llm_calls=total_llm_calls,
            )

            # Clean up temp JSONL unless retention requested
            if not retain_temp_jsonl and temp_jsonl_path.exists():
                temp_jsonl_path.unlink()
                logger.info("Removed temp JSONL: %s", temp_jsonl_path)

        return adjusted_ranges

    @staticmethod
    def _rebuild_ranges_from_jsonl(
        temp_jsonl_path: Path,
        original_ranges: Sequence[Tuple[int, int]],
    ) -> Tuple[List[Tuple[int, int]], List[int]]:
        """Rebuild adjusted ranges and deletion list from a complete temp JSONL."""
        from modules.core.jsonl_utils import read_jsonl_records

        results_by_index: Dict[int, Dict[str, Any]] = {}
        for record in read_jsonl_records(temp_jsonl_path):
            custom_id = record.get("custom_id", "")
            match = re.search(r"-range-(\d+)$", str(custom_id))
            if match:
                idx = int(match.group(1))
                body = record.get("response", {}).get("body", {})
                results_by_index[idx] = body

        adjusted: List[Tuple[int, int]] = []
        deleted: List[int] = []
        for index, original_range in enumerate(original_ranges, start=1):
            body = results_by_index.get(index)
            if body is None:
                # Not in JSONL (shouldn't happen if run completed)
                adjusted.append(original_range)
                continue
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
        extractor,
        raw_lines: Sequence[str],
        original_range: Tuple[int, int],
        range_index: int,
        boundary_type: str,
        context: Optional[str],
    ) -> RangeResult:
        """
        Process a single range to find semantic boundaries.

        Strategy (certainty-based routing):
        1. Check certainty first - if below threshold, retry with different window
        2. Route by high-certainty decision:
           - needs_more_context: Expand window progressively
           - contains_no_semantic_boundary: Verify with broad scan -> delete if confirmed
           - Success (marker found): Validate and apply

        Returns:
            A ``RangeResult`` capturing the final decision, adjusted range,
            deletion flag, and a full audit trail of every LLM call.
        """
        total_lines = len(raw_lines)
        windows = list(self._generate_windows(original_range, total_lines))
        decision: Optional[BoundaryDecision] = None
        adjusted_range = original_range

        low_certainty_retries = 0
        context_expansion_attempts = 0

        stop_processing = False

        failed_marker_history: List[str] = []
        attempts: List[Dict[str, Any]] = []
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
                    attempts.append({
                        "window": list(window),
                        "window_index": window_idx,
                        "decision_type": "low_certainty",
                        "certainty": decision.certainty,
                        "semantic_marker": decision.semantic_marker,
                        "marker_matched": False,
                    })
                    low_certainty_retries += 1
                    if low_certainty_retries <= self.max_low_certainty_retries:
                        logger.info(
                            "[Range %d, Window %d] Low certainty (%d < %d); retry %d/%d",
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
                            "[Range %d] Certainty remains low after %d retries (best: %d); keeping original range",
                            range_index,
                            self.max_low_certainty_retries,
                            decision.certainty,
                        )
                        stop_processing = True
                        break

                # ====================================================================
                # High certainty response - route by decision type
                # ====================================================================

                # ROUTE 1: Model requests more context -> Expand window
                if decision.needs_more_context:
                    attempts.append({
                        "window": list(window),
                        "window_index": window_idx,
                        "decision_type": "needs_more_context",
                        "certainty": decision.certainty,
                        "semantic_marker": decision.semantic_marker,
                        "marker_matched": False,
                    })
                    context_expansion_attempts += 1
                    if context_expansion_attempts <= self.max_context_expansion_attempts:
                        logger.info(
                            "[Range %d, Window %d] Requests more context (certainty: %d); expansion %d/%d",
                            range_index,
                            window_idx,
                            decision.certainty,
                            context_expansion_attempts,
                            self.max_context_expansion_attempts,
                        )
                        break  # Move to next (larger) window
                    else:
                        logger.warning(
                            "[Range %d] Max context expansions reached (%d); keeping original range",
                            range_index,
                            self.max_context_expansion_attempts,
                        )
                        stop_processing = True
                        break

                # ROUTE 2: Model confident no content exists -> Verify with broad scan
                elif decision.contains_no_semantic_boundary:
                    attempts.append({
                        "window": list(window),
                        "window_index": window_idx,
                        "decision_type": "no_semantic_boundary",
                        "certainty": decision.certainty,
                        "semantic_marker": decision.semantic_marker,
                        "marker_matched": False,
                    })
                    logger.info(
                        "[Range %d, Window %d] High-certainty no-content response (certainty: %d); triggering verification",
                        range_index,
                        window_idx,
                        decision.certainty,
                    )

                    if self.delete_ranges_with_no_content:
                        should_delete, verify_attempts = await self._verify_no_content(
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
                        logger.info(
                            "[Range %d] Verification scan found content in broader area; keeping range",
                            range_index,
                        )
                        stop_processing = True
                        break
                    else:
                        logger.info(
                            "[Range %d] No content detected but deletion disabled; keeping range",
                            range_index,
                        )
                        stop_processing = True
                        break

                # ROUTE 3: Success -> Marker found with high certainty, validate and apply
                else:
                    candidate_range = self._validate_and_apply_decision(
                        decision=decision,
                        raw_lines=raw_lines,
                        context_window=window,
                        fallback_range=original_range,
                    )
                    if candidate_range:
                        adjusted_range = candidate_range
                        attempts.append({
                            "window": list(window),
                            "window_index": window_idx,
                            "decision_type": "marker_found",
                            "certainty": decision.certainty,
                            "semantic_marker": decision.semantic_marker,
                            "marker_matched": True,
                        })
                        logger.info(
                            "[Range %d] Successfully adjusted to %s using marker '%s' (certainty: %d)",
                            range_index,
                            adjusted_range,
                            decision.semantic_marker,
                            decision.certainty,
                        )
                        stop_processing = True
                        break

                    marker_text = (decision.semantic_marker or "").strip()
                    attempts.append({
                        "window": list(window),
                        "window_index": window_idx,
                        "decision_type": "marker_mismatch",
                        "certainty": decision.certainty,
                        "semantic_marker": decision.semantic_marker,
                        "marker_matched": False,
                    })
                    if marker_text and marker_text not in failed_marker_history:
                        failed_marker_history.append(marker_text)
                    marker_mismatch_retries += 1

                    if marker_mismatch_retries <= self.max_marker_mismatch_retries:
                        logger.info(
                            "[Range %d, Window %d] Semantic marker '%s' not matched; retry %d/%d",
                            range_index,
                            window_idx,
                            marker_text or "<empty>",
                            marker_mismatch_retries,
                            self.max_marker_mismatch_retries,
                        )
                        continue  # Retry same window with additional guidance

                    logger.warning(
                        "[Range %d, Window %d] Semantic marker not matched after %d retries; trying next window",
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
        extractor,
        raw_lines: Sequence[str],
        original_range: Tuple[int, int],
        range_index: int,
        boundary_type: str,
        context: Optional[str],
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Verify that no semantic content of the required type exists within the range.

        Called after the initial model assessment returned contains_no_semantic_boundary.
        Scans the range interior to confirm the absence of content before deletion.

        Returns:
            Tuple of (should_delete, verification_attempts).
        """
        start, end = original_range
        range_size = end - start + 1
        verification_attempts: List[Dict[str, Any]] = []

        scan_radius = range_size * self.scan_range_multiplier

        # Build scan windows WITHIN the range (not adjacent areas)
        if range_size <= 2 * scan_radius:
            # Range is short enough to scan entirely in one call
            scan_windows = [(start, end)]
        else:
            # Long range: sample first and last portions
            scan_windows = [
                (start, min(end, start + scan_radius - 1)),
                (max(start, end - scan_radius + 1), end),
            ]

        logger.info(
            "[Range %d] Verifying no content: scanning %d interior window(s) within [%d-%d]",
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
            verification_attempts.append({
                "window": [scan_start, scan_end],
                "window_index": -(idx + 1),
                "decision_type": f"verify_interior_{idx + 1}",
                "certainty": decision.certainty,
                "semantic_marker": decision.semantic_marker,
                "found_content": (
                    not decision.contains_no_semantic_boundary
                    and bool(decision.semantic_marker)
                ),
            })

            # If we found content inside the range, don't delete
            if not decision.contains_no_semantic_boundary and decision.semantic_marker:
                logger.info(
                    "[Range %d] Found recipe content in interior scan window %d; preserving range",
                    range_index,
                    idx + 1,
                )
                return False, verification_attempts

        # All interior scans confirmed no content — safe to delete
        logger.info(
            "[Range %d] Verified no recipe content in range interior; confirming deletion",
            range_index,
        )
        return True, verification_attempts

    async def _run_model(
        self,
        *,
        extractor,
        raw_lines: Sequence[str],
        original_range: Tuple[int, int],
        context_window: Tuple[int, int],
        window_index: int,
        boundary_type: str,
        context: Optional[str],
        failed_markers: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        start, end = original_range
        context_start, context_end = context_window
        context_block = self._format_context(raw_lines, context_start, context_end)
        
        chunk_text = (
            f"Input text:\n\n"
            f"{context_block}\n"
        )

        # Render system prompt with unified context and schema
        system_prompt = render_prompt_with_schema(
            self.prompt_template,
            SEMANTIC_BOUNDARY_SCHEMA["schema"],
            inject_schema=True,
            context=context,
        )

        if failed_markers:
            sanitized_failures = [marker.strip() for marker in failed_markers if marker and marker.strip()]
            if sanitized_failures:
                bullet_list = "\n".join(f"- {marker}" for marker in sanitized_failures)
                system_prompt += (
                    "\n\nThe following semantic markers previously failed to match the source text. "
                    "Do not reuse any of them. Provide a new 5-15 character substring that excludes these markers:\n"
                    f"{bullet_list}"
                )

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
                "Model response could not be parsed as JSON; treating as unsure. Raw: %s",
                raw_output,
            )
            return {
                "contains_no_semantic_boundary": False,
                "needs_more_context": False,
                "certainty": 0,
                "semantic_marker": "",
            }
        return parsed

    def _validate_and_apply_decision(
        self,
        *,
        decision: BoundaryDecision,
        raw_lines: Sequence[str],
        context_window: Tuple[int, int],
        fallback_range: Tuple[int, int],
    ) -> Optional[Tuple[int, int]]:
        if decision.semantic_marker is None or not decision.semantic_marker.strip():
            return None

        context_start, context_end = context_window
        matched_line = self._match_boundary_text(
            boundary_text=decision.semantic_marker,
            raw_lines=raw_lines,
            search_start=context_start,
            search_end=context_end,
            substring_match=decision.semantic_marker,
        )
        if matched_line is None:
            return None

        # For now, keep the original end line - we only adjust the start boundary
        new_start = matched_line
        new_end = fallback_range[1]

        return new_start, new_end

    def _generate_windows(
        self,
        original_range: Tuple[int, int],
        total_lines: int,
    ) -> Iterable[Tuple[int, int]]:
        start, _ = original_range
        radius = self.context_window
        max_multiplier = max(1, int(self.scan_range_multiplier))

        yielded: set[Tuple[int, int]] = set()

        def emit(window_start: int, window_end: int) -> None:
            bounded = (max(1, window_start), min(total_lines, window_end))
            if bounded[1] < bounded[0]:
                return
            if bounded not in yielded:
                yielded.add(bounded)
                yield bounded

        # Base window: focus on the original start with configured radius
        yield from emit(start - radius, start + radius)

        # Expanded windows: scale outward around the start using scan_range_multiplier
        for multiplier in range(2, max_multiplier + 2):
            expanded_radius = radius * multiplier
            yield from emit(start - expanded_radius, start + expanded_radius)

    def _format_context(
        self,
        raw_lines: Sequence[str],
        context_start: int,
        context_end: int,
    ) -> str:
        snippet: List[str] = []
        for line_number in range(context_start, context_end + 1):
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
            normalized = re.sub(r'\s+', ' ', normalized)
        
        # Normalize diacritics (remove accents, umlauts, etc.)
        if self.normalize_diacritics:
            # Decompose unicode characters and filter out combining marks
            normalized = ''.join(
                char for char in unicodedata.normalize('NFD', normalized)
                if unicodedata.category(char) != 'Mn'
            )
        
        # Case normalization
        if not self.case_sensitive:
            normalized = normalized.lower()
        
        # Strip punctuation
        if self.strip_punctuation:
            # Remove common punctuation while preserving alphanumeric and spaces
            normalized = re.sub(r'[^\w\s]', '', normalized)
        
        return normalized

    def _collect_normalized_matches(
        self,
        raw_lines: Sequence[str],
        needle: str,
        search_range: Tuple[int, int],
    ) -> List[int]:
        """
        Collect line numbers where the needle matches, applying configured normalization.
        
        Supports both exact line matching and substring matching based on configuration.
        """
        start_line, end_line = search_range
        end_line = min(len(raw_lines), max(start_line, end_line))
        matches: List[int] = []
        
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
        boundary_text: str,
        raw_lines: Sequence[str],
        search_start: int,
        search_end: int,
        substring_match: Optional[str] = None,
    ) -> Optional[int]:
        """
        Match boundary text in the source lines using configured normalization.
        
        Prefers substring_match if provided, otherwise falls back to boundary_text.
        """
        # Prefer the precise substring_match if provided and meets minimum length
        if substring_match and len(substring_match.strip()) >= self.min_substring_length:
            needle = substring_match.strip()
        else:
            needle = boundary_text.strip()
        
        if not needle:
            return None

        # First, try to match within the specified search range
        matches = self._collect_normalized_matches(
            raw_lines,
            needle,
            (max(1, search_start), min(len(raw_lines), search_end)),
        )

        # If no match found in the search range, expand to the entire document
        if not matches:
            matches = self._collect_normalized_matches(
                raw_lines,
                needle,
                (1, len(raw_lines)),
            )

        # Return unique match, or None if ambiguous/not found
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            logger.debug(
                "Expected unique semantic boundary for '%s', found multiple candidates at lines %s",
                needle,
                matches,
            )
        return None

    def _coerce_json(self, raw_output: str) -> Optional[Dict[str, Any]]:
        text = raw_output.strip()
        if not text:
            return None

        code_block_pattern = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
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

    def _write_line_ranges(self, line_ranges_file: Path, ranges: Sequence[Tuple[int, int]]) -> None:
        safe_line_ranges_file = ensure_path_safe(line_ranges_file)
        with safe_line_ranges_file.open("w", encoding="utf-8") as handle:
            for start, end in ranges:
                handle.write(f"({start}, {end})\n")

    @staticmethod
    def _remove_overlaps(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
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

        if len(ranges) == 1:
            return list(ranges)

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

        # Sort by start (stable on original index) so out-of-order ranges are processed safely.
        annotated.sort(key=lambda item: (item["start"], item["original_index"]))

        processed: List[Dict[str, int]] = []

        for entry in annotated:
            current_start = entry["start"]
            current_end = entry["end"]
            original_start = entry["original_start"]
            original_end = entry["original_end"]

            if current_start > current_end:
                logger.warning(
                    "Range (%d, %d) has end before start; clamping end to %d before overlap removal.",
                    original_start,
                    original_end,
                    current_start,
                )
                current_end = current_start

            if processed:
                previous = processed[-1]

                if previous["end"] >= current_start:
                    trimmed_prev_end = min(previous["end"], max(previous["start"], current_start - 1))

                    if trimmed_prev_end < previous["end"]:
                        logger.info(
                            "Overlap removed: range (%d, %d) trimmed to (%d, %d) to respect next range (%d, %d)",
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
                            "Adjusted start of range (%d, %d) forward by %d line(s) after trimming to avoid overlap.",
                            original_start,
                            original_end,
                            shift_amount,
                        )

            if current_start > current_end:
                logger.warning(
                    "Range (%d, %d) would invert after overlap handling; clamping end to %d.",
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

    def _enforce_max_gap(self, ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Ensure gaps between consecutive ranges do not exceed configured maximum.

        Ranges are assumed to be sorted and non-overlapping. When a gap exceeds
        ``max_gap_between_ranges``, the previous range is extended forward to
        reduce the gap while still ending before the next range.
        """
        if not ranges:
            return []

        max_gap = self.max_gap_between_ranges
        if max_gap is None:
            return list(ranges)

        enforced: List[Tuple[int, int]] = []

        for index, (start, end) in enumerate(ranges):
            if index == 0:
                enforced.append((start, end))
                continue

            prev_start, prev_end = enforced[-1]
            gap = start - prev_end - 1

            if gap > max_gap:
                new_prev_end = start - max_gap - 1
                logger.info(
                    "Gap of %d lines detected between ranges (%d, %d) and (%d, %d); "
                    "extending previous range to (%d, %d)",
                    gap,
                    prev_start,
                    prev_end,
                    start,
                    end,
                    prev_start,
                    new_prev_end,
                )
                enforced[-1] = (prev_start, new_prev_end)
                prev_end = new_prev_end

            enforced.append((start, end))

        return enforced


async def adjust_line_ranges_for_paths(
    *,
    model_config: Dict[str, Any],
    text_files: Iterable[Path],
    context_window: int,
    prompt_path: Optional[Path] = None,
    dry_run: bool = False,
    boundary_type: str,
) -> Dict[Path, List[Tuple[int, int]]]:
    """Adjust line ranges for multiple files.
    
    Context is resolved automatically using hierarchical fallback.
    """
    readjuster = LineRangeReadjuster(
        model_config,
        context_window=context_window,
        prompt_path=prompt_path,
    )

    results: Dict[Path, List[Tuple[int, int]]] = {}
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
    model_config: Dict[str, Any],
    text_file: Path,
    context_window: int,
    prompt_path: Optional[Path] = None,
    dry_run: bool = False,
    boundary_type: str,
) -> List[Tuple[int, int]]:
    """Run line range adjustment synchronously.
    
    Context is resolved automatically using hierarchical fallback.
    """
    async def _runner() -> List[Tuple[int, int]]:
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
