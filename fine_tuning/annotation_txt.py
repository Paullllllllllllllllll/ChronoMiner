from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


CHUNK_HEADER_RE = re.compile(r"^===\s*chunk\s+(\d+)\s*===\s*$", re.IGNORECASE)
INPUT_BEGIN = "--- INPUT_TEXT_BEGIN ---"
INPUT_END = "--- INPUT_TEXT_END ---"
OUTPUT_BEGIN = "--- OUTPUT_JSON_BEGIN ---"
OUTPUT_END = "--- OUTPUT_JSON_END ---"


@dataclass
class ChunkAnnotation:
    chunk_index: int
    input_text: str
    output: Optional[Dict[str, Any]] = None


def read_annotations_txt(path: Path) -> List[ChunkAnnotation]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    annotations: List[ChunkAnnotation] = []

    current_chunk: Optional[int] = None
    input_lines: List[str] = []
    output_lines: List[str] = []
    mode: Optional[str] = None

    def _flush() -> None:
        nonlocal current_chunk, input_lines, output_lines, mode
        if current_chunk is None:
            return

        input_text = "\n".join(input_lines).strip("\n")
        raw_output = "\n".join(output_lines).strip()
        output: Optional[Dict[str, Any]]
        if not raw_output:
            output = None
        else:
            parsed = json.loads(raw_output)
            if not isinstance(parsed, dict):
                raise ValueError(
                    f"Chunk {current_chunk}: output JSON must be an object (got {type(parsed).__name__})"
                )
            output = parsed

        annotations.append(
            ChunkAnnotation(chunk_index=current_chunk, input_text=input_text, output=output)
        )

        current_chunk = None
        input_lines = []
        output_lines = []
        mode = None

    for line_no, line in enumerate(lines, 1):
        header_match = CHUNK_HEADER_RE.match(line.strip())
        if header_match:
            _flush()
            current_chunk = int(header_match.group(1))
            continue

        if current_chunk is None:
            if not line.strip():
                continue
            raise ValueError(
                f"Line {line_no}: content found before first chunk header. "
                f"Expected '=== chunk N ==='"
            )

        if line.strip() == INPUT_BEGIN:
            mode = "input"
            continue
        if line.strip() == INPUT_END:
            mode = None
            continue
        if line.strip() == OUTPUT_BEGIN:
            mode = "output"
            continue
        if line.strip() == OUTPUT_END:
            mode = None
            continue

        if mode == "input":
            input_lines.append(line)
        elif mode == "output":
            output_lines.append(line)
        else:
            if not line.strip():
                continue
            raise ValueError(
                f"Chunk {current_chunk}, line {line_no}: unexpected content outside input/output blocks"
            )

    _flush()

    if not annotations:
        raise ValueError("No chunks found")

    annotations.sort(key=lambda a: a.chunk_index)

    seen = set()
    duplicates = [a.chunk_index for a in annotations if (a.chunk_index in seen) or seen.add(a.chunk_index)]
    if duplicates:
        dup_str = ", ".join(str(x) for x in sorted(set(duplicates)))
        raise ValueError(f"Duplicate chunk indices found: {dup_str}")

    return annotations


def write_annotations_txt(
    annotations: List[ChunkAnnotation],
    path: Path,
    *,
    pretty_json: bool = True,
) -> None:
    out_lines: List[str] = []
    for ann in sorted(annotations, key=lambda a: a.chunk_index):
        out_lines.append(f"=== chunk {ann.chunk_index} ===")
        out_lines.append(INPUT_BEGIN)
        out_lines.extend((ann.input_text or "").splitlines())
        out_lines.append(INPUT_END)
        out_lines.append(OUTPUT_BEGIN)
        if ann.output is not None:
            if pretty_json:
                out_lines.append(json.dumps(ann.output, ensure_ascii=False, indent=2, sort_keys=True))
            else:
                out_lines.append(json.dumps(ann.output, ensure_ascii=False, separators=(",", ":"), sort_keys=True))
        out_lines.append(OUTPUT_END)
        out_lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out_lines).rstrip() + "\n", encoding="utf-8")
