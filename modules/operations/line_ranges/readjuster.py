from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from modules.core.prompt_context import apply_context_placeholders, resolve_additional_context
from modules.core.schema_manager import SchemaManager
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
            "no_semantic_boundary": {
                "type": "boolean",
                "description": "Set to true when no reliable semantic boundary exists in the provided context.",
            },
            "boundary_text": {
                "type": "string",
                "description": "Verbatim text that starts the semantic boundary.",
            },
            "closing_boundary_text": {
                "type": "string",
                "description": "Optional verbatim text that signals where the semantic span ends.",
            },
            "notes": {
                "type": "string",
                "description": "Optional explanation of the decision.",
            },
        },
        "required": ["no_semantic_boundary"],
        "additionalProperties": False,
        "allOf": [
            {
                "if": {
                    "properties": {"no_semantic_boundary": {"const": False}},
                },
                "then": {
                    "required": ["boundary_text"],
                },
            }
        ],
    },
}

@dataclass
class BoundaryDecision:
    no_boundary: bool
    boundary_text: Optional[str] = None
    closing_boundary_text: Optional[str] = None
    notes: Optional[str] = None

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "BoundaryDecision":
        return cls(
            no_boundary=bool(payload.get("no_semantic_boundary", False)),
            boundary_text=payload.get("boundary_text"),
            closing_boundary_text=payload.get("closing_boundary_text"),
            notes=payload.get("notes"),
        )


class LineRangeReadjuster:
    """Adjust `_line_ranges.txt` files so that chunk boundaries align with semantic boundaries."""

    def __init__(
        self,
        model_config: Dict[str, Any],
        *,
        context_window: int = 6,
        prompt_path: Optional[Path] = None,
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
        self._schema_manager = SchemaManager()
        self._schema_manager.load_schemas()
        self._schema_cache: Dict[str, Dict[str, Any]] = {}

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

        schema_context = self._get_schema_context(boundary_type)

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
                    schema_context=schema_context,
                    basic_context=basic_context,
                    additional_context=additional_context,
                )
                adjusted_ranges.append(updated_range)
                if decision.no_boundary:
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
                        decision.boundary_text,
                    )

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
        schema_context: str,
        basic_context: Optional[str],
        additional_context: Optional[str],
    ) -> Tuple[BoundaryDecision, Tuple[int, int]]:
        total_lines = len(raw_lines)
        windows = list(self._generate_windows(original_range, total_lines))
        decision: Optional[BoundaryDecision] = None
        adjusted_range = original_range

        for window_idx, window in enumerate(windows):
            payload = await self._run_model(
                extractor=extractor,
                raw_lines=raw_lines,
                original_range=original_range,
                context_window=window,
                window_index=window_idx,
                boundary_type=boundary_type,
                schema_context=schema_context,
                basic_context=basic_context,
                additional_context=additional_context,
            )
            decision = BoundaryDecision.from_payload(payload)
            if decision.no_boundary:
                continue

            candidate_range = self._validate_and_apply_decision(
                decision=decision,
                raw_lines=raw_lines,
                context_window=window,
                fallback_range=original_range,
            )
            if candidate_range:
                adjusted_range = candidate_range
                break

        if decision is None:
            decision = BoundaryDecision(no_boundary=True)
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
        schema_context: str,
        basic_context: Optional[str],
        additional_context: Optional[str],
    ) -> Dict[str, Any]:
        start, end = original_range
        context_start, context_end = context_window
        context_block = self._format_context(raw_lines, context_start, context_end)
        instruction_body = (
            f"Task: locate a closed semantic boundary of type '{boundary_type}'.\n"
            "- Use the schema description to understand what constitutes a boundary.\n"
            "- Provide only the verbatim text that marks the boundary.\n"
            "- Optionally provide verbatim closing text when it clearly marks the end of the same unit.\n"
            "- If no boundary exists, set no_semantic_boundary to true.\n"
            "- Respond with JSON only matching the provided schema."
        )

        chunk_text = (
            f"Window index: {window_index}\n"
            f"Original range: {start}-{end}\n"
            f"Context snippet:\n{context_block}\n"
        )

        system_prompt = apply_context_placeholders(
            self.prompt_template,
            basic_context=basic_context,
            additional_context=additional_context,
        )

        system_prompt = system_prompt.replace("{{TRANSCRIPTION_SCHEMA}}", schema_context)

        response_payload = await process_text_chunk(
            text_chunk=f"{instruction_body}\n\n{chunk_text}",
            extractor=extractor,
            system_message=system_prompt,
            json_schema=SEMANTIC_BOUNDARY_SCHEMA,
        )

        raw_output = response_payload.get("output_text", "")
        parsed = self._coerce_json(raw_output)
        if parsed is None:
            logger.warning(
                "Model response could not be parsed as JSON; treating as no boundary. Raw: %s",
                raw_output,
            )
            return {"no_semantic_boundary": True, "notes": "Unparseable JSON response."}
        return parsed

    def _validate_and_apply_decision(
        self,
        *,
        decision: BoundaryDecision,
        raw_lines: Sequence[str],
        context_window: Tuple[int, int],
        fallback_range: Tuple[int, int],
    ) -> Optional[Tuple[int, int]]:
        if decision.boundary_text is None:
            return None

        context_start, context_end = context_window
        matched_line = self._match_boundary_text(
            boundary_text=decision.boundary_text,
            raw_lines=raw_lines,
            search_start=context_start,
            search_end=context_end,
        )
        if matched_line is None:
            return None

        new_start = matched_line
        new_end = fallback_range[1]
        if decision.closing_boundary_text:
            closing_line = self._match_boundary_text(
                boundary_text=decision.closing_boundary_text,
                raw_lines=raw_lines,
                search_start=matched_line,
                search_end=len(raw_lines),
            )
            if closing_line is not None and closing_line >= new_start:
                new_end = closing_line

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

    def _match_boundary_text(
        self,
        *,
        boundary_text: str,
        raw_lines: Sequence[str],
        search_start: int,
        search_end: int,
    ) -> Optional[int]:
        needle = boundary_text.strip()
        if not needle:
            return None

        matches = self._collect_exact_matches(
            raw_lines,
            needle,
            (max(1, search_start), min(len(raw_lines), search_end)),
        )

        if not matches:
            matches = self._collect_exact_matches(
                raw_lines,
                needle,
                (1, len(raw_lines)),
            )

        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            logger.debug(
                "Expected unique semantic boundary for '%s', found multiple candidates at lines %s",
                needle,
                matches,
            )
        return None

    def _collect_exact_matches(
        self,
        raw_lines: Sequence[str],
        needle: str,
        search_range: Tuple[int, int],
    ) -> List[int]:
        start_line, end_line = search_range
        end_line = min(len(raw_lines), max(start_line, end_line))
        matches: List[int] = []
        for line_number in range(start_line, end_line + 1):
            line_text = raw_lines[line_number - 1].strip()
            if line_text == needle:
                matches.append(line_number)
        return matches

    def _get_schema_context(self, boundary_type: str) -> str:
        if boundary_type not in self._schema_cache:
            schema = self._schema_manager.get_available_schemas().get(boundary_type)
            if not schema:
                raise ValueError(f"Unknown boundary type '{boundary_type}'. Ensure schema exists in the schemas directory.")
            self._schema_cache[boundary_type] = schema
        return json.dumps(self._schema_cache[boundary_type], indent=2, ensure_ascii=False)

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
