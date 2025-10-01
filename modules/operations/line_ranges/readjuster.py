from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from modules.core.prompt_context import apply_context_placeholders, resolve_additional_context
from modules.core.text_utils import TextProcessor, load_line_ranges
from modules.llm.openai_utils import open_extractor, process_text_chunk
from modules.llm.prompt_utils import load_prompt_template

logger = logging.getLogger(__name__)


SEMANTIC_BOUNDARY_SCHEMA: Dict[str, Any] = {
    "name": "SemanticBoundaryResponse",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "no_semantic_marker": {
                "type": "boolean",
                "description": "Set to true when no reliable semantic marker exists in the provided context.",
            },
            "unsure": {
                "type": "boolean",
                "description": "Set to true when uncertain about the semantic marker identification.",
            },
            "semantic_marker": {
                "type": "string",
                "description": "A precise 10-15 character verbatim substring that marks the semantic boundary. Leave empty if no_semantic_marker or unsure is true.",
            },
        },
        "required": ["no_semantic_marker", "unsure", "semantic_marker"],
        "additionalProperties": False,
    },
}

@dataclass
class BoundaryDecision:
    no_semantic_marker: bool
    unsure: bool
    semantic_marker: Optional[str] = None

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "BoundaryDecision":
        return cls(
            no_semantic_marker=bool(payload.get("no_semantic_marker", False)),
            unsure=bool(payload.get("unsure", False)),
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
        self.max_retries_on_unsure = self.retry_config.get("max_retries_on_unsure", 2)
        self.max_retries_on_no_marker = self.retry_config.get("max_retries_on_no_marker", 1)
        self.expand_context_on_retry = self.retry_config.get("expand_context_on_retry", True)

    async def ensure_adjusted_line_ranges(
        self,
        *,
        text_file: Path,
        line_ranges_file: Optional[Path] = None,
        dry_run: bool = False,
        boundary_type: Optional[str] = None,
        basic_context: Optional[str] = None,
        context_settings: Optional[Dict[str, Any]] = None,
        context_manager: Optional[Any] = None,
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

        encoding = TextProcessor.detect_encoding(text_file)
        with text_file.open("r", encoding=encoding) as handle:
            raw_lines = handle.readlines()

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is required for readjustment")

        additional_context = resolve_additional_context(
            boundary_type,
            context_settings=context_settings,
            context_manager=context_manager,
            text_file=text_file,
        )

        adjusted_ranges: List[Tuple[int, int]] = []
        async with open_extractor(
            api_key=api_key,
            prompt_path=self.prompt_path,
            model=self.model_name,
        ) as extractor:
            for index, original_range in enumerate(ranges, start=1):
                decision, updated_range = await self._process_single_range(
                    extractor=extractor,
                    raw_lines=raw_lines,
                    original_range=original_range,
                    range_index=index,
                    boundary_type=boundary_type,
                    basic_context=basic_context,
                    additional_context=additional_context,
                )
                adjusted_ranges.append(updated_range)
                if decision.no_semantic_marker:
                    logger.info(
                        "[Range %d] No semantic boundary detected; kept original %s",
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

        if not dry_run:
            self._write_line_ranges(line_ranges_file, adjusted_ranges)

        return adjusted_ranges

    async def _process_single_range(
        self,
        *,
        extractor,
        raw_lines: Sequence[str],
        original_range: Tuple[int, int],
        range_index: int,
        boundary_type: str,
        basic_context: Optional[str],
        additional_context: Optional[str],
    ) -> Tuple[BoundaryDecision, Tuple[int, int]]:
        total_lines = len(raw_lines)
        windows = list(self._generate_windows(original_range, total_lines))
        decision: Optional[BoundaryDecision] = None
        adjusted_range = original_range

        unsure_retries = 0
        no_marker_retries = 0

        for window_idx, window in enumerate(windows):
            payload = await self._run_model(
                extractor=extractor,
                raw_lines=raw_lines,
                original_range=original_range,
                context_window=window,
                window_index=window_idx,
                boundary_type=boundary_type,
                basic_context=basic_context,
                additional_context=additional_context,
            )
            decision = BoundaryDecision.from_payload(payload)
            
            # Handle unsure responses with retry
            if decision.unsure:
                if unsure_retries < self.max_retries_on_unsure:
                    unsure_retries += 1
                    logger.info(
                        "[Range %d, Window %d] Model is unsure; retry %d/%d",
                        range_index,
                        window_idx,
                        unsure_retries,
                        self.max_retries_on_unsure,
                    )
                    continue  # Try next window or retry
                else:
                    logger.warning(
                        "[Range %d] Model remains unsure after %d retries; keeping original range",
                        range_index,
                        self.max_retries_on_unsure,
                    )
                    break
            
            # Handle no_semantic_marker responses with retry
            if decision.no_semantic_marker:
                if no_marker_retries < self.max_retries_on_no_marker:
                    no_marker_retries += 1
                    logger.info(
                        "[Range %d, Window %d] No semantic marker found; retry %d/%d",
                        range_index,
                        window_idx,
                        no_marker_retries,
                        self.max_retries_on_no_marker,
                    )
                    continue  # Try next window
                else:
                    logger.info(
                        "[Range %d] No semantic marker found after %d attempts; keeping original range",
                        range_index,
                        no_marker_retries + 1,
                    )
                    break

            # Try to validate and apply the decision
            candidate_range = self._validate_and_apply_decision(
                decision=decision,
                raw_lines=raw_lines,
                context_window=window,
                fallback_range=original_range,
            )
            if candidate_range:
                adjusted_range = candidate_range
                logger.info(
                    "[Range %d] Successfully adjusted to %s using semantic marker '%s'",
                    range_index,
                    adjusted_range,
                    decision.semantic_marker,
                )
                break
            else:
                # Marker found but couldn't match in text - continue searching
                logger.debug(
                    "[Range %d, Window %d] Semantic marker '%s' could not be matched in text",
                    range_index,
                    window_idx,
                    decision.semantic_marker,
                )

        if decision is None:
            decision = BoundaryDecision(no_semantic_marker=True, unsure=False)
            logger.info(
                "[Range %d] No windows examined; keeping original range %s",
                range_index,
                original_range,
            )
        
        return decision, adjusted_range

    async def _run_model(
        self,
        *,
        extractor,
        raw_lines: Sequence[str],
        original_range: Tuple[int, int],
        context_window: Tuple[int, int],
        window_index: int,
        boundary_type: str,
        basic_context: Optional[str],
        additional_context: Optional[str],
    ) -> Dict[str, Any]:
        start, end = original_range
        context_start, context_end = context_window
        context_block = self._format_context(raw_lines, context_start, context_end)
        
        chunk_text = (
            f"Input text:\n\n"
            f"{context_block}\n"
        )

        system_prompt = apply_context_placeholders(
            self.prompt_template,
            basic_context=basic_context,
            additional_context=additional_context,
        )

        # Inject the semantic boundary schema into the prompt
        schema_json = json.dumps(SEMANTIC_BOUNDARY_SCHEMA["schema"], indent=2, ensure_ascii=False)
        system_prompt = system_prompt.replace("{{TRANSCRIPTION_SCHEMA}}", schema_json)

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
            return {"no_semantic_marker": False, "unsure": True, "semantic_marker": ""}
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
        start, end = original_range
        radius = self.context_window
        base_start = max(1, start - radius)
        base_end = min(total_lines, end + radius)
        window_size = base_end - base_start
        step = max(1, radius // 2)

        yielded: set[Tuple[int, int]] = set()

        def emit(window_start: int, window_end: int) -> None:
            bounded = (max(1, window_start), min(total_lines, window_end))
            if bounded[1] < bounded[0]:
                return
            if bounded not in yielded:
                yielded.add(bounded)
                yield bounded

        yield from emit(base_start, base_end)

        for multiplier in range(2, 4):
            expanded_start = max(1, start - radius * multiplier)
            expanded_end = min(total_lines, end + radius * multiplier)
            yield from emit(expanded_start, expanded_end)

        window_length = max(1, base_end - base_start)

        cursor = base_start
        while cursor > 1:
            cursor = max(1, cursor - step)
            yield from emit(cursor, cursor + window_length)
            if cursor == 1:
                break

        cursor = base_start
        while cursor + window_length <= total_lines:
            cursor = min(total_lines - window_length, cursor + step)
            yield from emit(cursor, cursor + window_length)
            if cursor + window_length >= total_lines:
                break

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
        with line_ranges_file.open("w", encoding="utf-8") as handle:
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
        
        adjusted_ranges: List[Tuple[int, int]] = []
        
        for i in range(len(ranges)):
            start, end = ranges[i]
            
            # Check if this range overlaps with the next range
            if i < len(ranges) - 1:
                next_start, next_end = ranges[i + 1]
                
                if end >= next_start:
                    # Overlap detected - adjust this range's end
                    original_end = end
                    end = next_start - 1
                    
                    # Ensure the range is still valid (start <= end)
                    if end < start:
                        logger.warning(
                            "Range (%d, %d) would become invalid after overlap removal. "
                            "Adjusting start to %d.",
                            start,
                            original_end,
                            end,
                        )
                        start = end
                    
                    logger.info(
                        "Overlap removed: range (%d, %d) adjusted to (%d, %d) "
                        "to avoid overlap with next range (%d, %d)",
                        start,
                        original_end,
                        start,
                        end,
                        next_start,
                        next_end,
                    )
            
            adjusted_ranges.append((start, end))
        
        return adjusted_ranges


async def adjust_line_ranges_for_paths(
    *,
    model_config: Dict[str, Any],
    text_files: Iterable[Path],
    context_window: int,
    prompt_path: Optional[Path] = None,
    dry_run: bool = False,
    boundary_type: str,
    basic_context: Optional[str] = None,
    context_settings: Optional[Dict[str, Any]] = None,
    context_manager: Optional[Any] = None,
) -> Dict[Path, List[Tuple[int, int]]]:
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
            basic_context=basic_context,
            context_settings=context_settings,
            context_manager=context_manager,
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
    basic_context: Optional[str] = None,
    context_settings: Optional[Dict[str, Any]] = None,
    context_manager: Optional[Any] = None,
) -> List[Tuple[int, int]]:
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
            basic_context=basic_context,
            context_settings=context_settings,
            context_manager=context_manager,
        )

    return asyncio.run(_runner())
