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
import json
import logging
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from modules.core.context_resolver import resolve_context_for_readjustment
from modules.core.resume import write_adjustment_marker
from modules.core.text_utils import TextProcessor, load_line_ranges
from modules.llm.openai_utils import open_extractor, process_text_chunk
from modules.llm.langchain_provider import ProviderConfig, ProviderType
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
        transcription_cfg = model_config.get("transcription_model", {})
        model_name: str = transcription_cfg.get("name", "")
        if not model_name:
            raise ValueError("transcription_model.name must be configured to use LineRangeReadjuster")

        self.model_name = model_name
        self.context_window = max(1, int(context_window))
        self.prompt_path = prompt_path or Path("prompts/semantic_boundary_prompt.txt")
        self.prompt_template = load_prompt_template(self.prompt_path)
        self.text_processor = TextProcessor()
        
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

        adjusted_ranges: List[Tuple[int, int]] = []
        ranges_to_delete: List[int] = []  # Track indices of ranges with no content
        
        async with open_extractor(
            api_key=api_key,
            prompt_path=self.prompt_path,
            model=self.model_name,
        ) as extractor:
            for index, original_range in enumerate(ranges, start=1):
                decision, updated_range, should_delete = await self._process_single_range(
                    extractor=extractor,
                    raw_lines=raw_lines,
                    original_range=original_range,
                    range_index=index,
                    boundary_type=boundary_type,
                    context=context,
                )
                
                if should_delete:
                    logger.warning(
                        "[Range %d] Confirmed no semantic content of type '%s' exists; marking for deletion",
                        index,
                        boundary_type,
                    )
                    ranges_to_delete.append(index - 1)  # Store 0-based index for deletion
                else:
                    adjusted_ranges.append(updated_range)
                    if decision.contains_no_semantic_boundary:
                        logger.info(
                            "[Range %d] No semantic boundary detected in context; kept original %s",
                            index,
                            updated_range,
                        )
                    else:
                        logger.info(
                            "[Range %d] Adjusted to %s via boundary '%s'",
                            index,
                            updated_range,
                            decision.semantic_marker,
                        )

        adjusted_ranges = self._remove_overlaps(adjusted_ranges)

        if self.max_gap_between_ranges is not None:
            adjusted_ranges = self._enforce_max_gap(adjusted_ranges)

        if ranges_to_delete:
            logger.info(
                "Deleted %d range(s) with no semantic content: %s",
                len(ranges_to_delete),
                [i + 1 for i in ranges_to_delete],  # Convert back to 1-based for display
            )

        if not dry_run:
            self._write_line_ranges(line_ranges_file, adjusted_ranges)
            write_adjustment_marker(
                line_ranges_file,
                boundary_type=boundary_type,
                context_window=self.context_window,
                model_name=self.model_name,
            )

        return adjusted_ranges

    async def _process_single_range(
        self,
        *,
        extractor,
        raw_lines: Sequence[str],
        original_range: Tuple[int, int],
        range_index: int,
        boundary_type: str,
        context: Optional[str],
    ) -> Tuple[BoundaryDecision, Tuple[int, int], bool]:
        """
        Process a single range to find semantic boundaries.
        
        Strategy (certainty-based routing):
        1. Check certainty first - if below threshold, retry with different window
        2. Route by high-certainty decision:
           - needs_more_context: Expand window progressively
           - contains_no_semantic_boundary: Verify with broad scan → delete if confirmed
           - Success (marker found): Validate and apply
        
        Returns:
            Tuple of (decision, adjusted_range, should_delete)
            - decision: The final boundary decision
            - adjusted_range: The adjusted line range
            - should_delete: True if this range should be deleted (no content confirmed)
        """
        total_lines = len(raw_lines)
        windows = list(self._generate_windows(original_range, total_lines))
        decision: Optional[BoundaryDecision] = None
        adjusted_range = original_range

        low_certainty_retries = 0
        context_expansion_attempts = 0

        stop_processing = False

        failed_marker_history: List[str] = []

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

                # ====================================================================
                # PRIORITY 1: Check certainty threshold (applies to all responses)
                # ====================================================================
                if decision.certainty < self.certainty_threshold:
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

                # ROUTE 1: Model requests more context → Expand window
                if decision.needs_more_context:
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

                # ROUTE 2: Model confident no content exists → Verify immediately with broad scan
                elif decision.contains_no_semantic_boundary:
                    logger.info(
                        "[Range %d, Window %d] High-certainty no-content response (certainty: %d); triggering verification",
                        range_index,
                        window_idx,
                        decision.certainty,
                    )

                    if self.delete_ranges_with_no_content:
                        should_delete = await self._verify_no_content(
                            extractor=extractor,
                            raw_lines=raw_lines,
                            original_range=original_range,
                            range_index=range_index,
                            boundary_type=boundary_type,
                            context=context,
                        )
                        if should_delete:
                            return decision, original_range, True
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

                # ROUTE 3: Success → Marker found with high certainty, validate and apply
                else:
                    candidate_range = self._validate_and_apply_decision(
                        decision=decision,
                        raw_lines=raw_lines,
                        context_window=window,
                        fallback_range=original_range,
                    )
                    if candidate_range:
                        adjusted_range = candidate_range
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
        
        return decision, adjusted_range, False  # Default: keep range

    async def _verify_no_content(
        self,
        *,
        extractor,
        raw_lines: Sequence[str],
        original_range: Tuple[int, int],
        range_index: int,
        boundary_type: str,
        context: Optional[str],
    ) -> bool:
        """
        Verify that no semantic content of the required type exists in a broader scan.
        
        This is called after the model has returned contains_no_semantic_boundary twice.
        We scan both upward and downward from the original range to confirm.
        
        Returns:
            True if deletion is confirmed (no content found in broader scan)
            False if content is potentially found (keep the range)
        """
        start, end = original_range
        total_lines = len(raw_lines)
        range_size = end - start + 1
        
        # Create broader scan windows (both up and down)
        scan_radius = range_size * self.scan_range_multiplier
        
        # Scan upward
        scan_up_start = max(1, start - scan_radius)
        scan_up_end = max(1, start - 1)
        
        # Scan downward
        scan_down_start = min(total_lines, end + 1)
        scan_down_end = min(total_lines, end + scan_radius)
        
        logger.info(
            "[Range %d] Verifying no content: scanning up [%d-%d] and down [%d-%d]",
            range_index,
            scan_up_start,
            scan_up_end,
            scan_down_start,
            scan_down_end,
        )
        
        # Scan upward first
        if scan_up_end >= scan_up_start:
            payload_up = await self._run_model(
                extractor=extractor,
                raw_lines=raw_lines,
                original_range=original_range,
                context_window=(scan_up_start, scan_up_end),
                window_index=-1,  # Special index for verification scan
                boundary_type=boundary_type,
                context=context,
            )
            decision_up = BoundaryDecision.from_payload(payload_up)
            
            # If we found content upward, don't delete
            if not decision_up.contains_no_semantic_boundary and decision_up.semantic_marker:
                logger.info(
                    "[Range %d] Found potential content in upward scan; preserving range",
                    range_index,
                )
                return False
        
        # Scan downward
        if scan_down_end >= scan_down_start:
            payload_down = await self._run_model(
                extractor=extractor,
                raw_lines=raw_lines,
                original_range=original_range,
                context_window=(scan_down_start, scan_down_end),
                window_index=-2,  # Special index for verification scan
                boundary_type=boundary_type,
                context=context,
            )
            decision_down = BoundaryDecision.from_payload(payload_down)
            
            # If we found content downward, don't delete
            if not decision_down.contains_no_semantic_boundary and decision_down.semantic_marker:
                logger.info(
                    "[Range %d] Found potential content in downward scan; preserving range",
                    range_index,
                )
                return False
        
        # Both scans confirmed no content - safe to delete
        logger.warning(
            "[Range %d] Verified no semantic content in broader scan; confirming deletion",
            range_index,
        )
        return True

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
