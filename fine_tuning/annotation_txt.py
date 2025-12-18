from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


# Header pattern: === chunk N === or === chunk N | source_file.txt ===
# The source identifier helps research assistants locate the original input
CHUNK_HEADER_RE = re.compile(
    r"^===\s*chunk\s+(\d+)(?:\s*\|\s*(.+?))?\s*===\s*$",
    re.IGNORECASE,
)
SOURCE_INFO = "--- SOURCE_FILE ---"
INPUT_BEGIN = "--- INPUT_TEXT_BEGIN ---"
INPUT_END = "--- INPUT_TEXT_END ---"
OUTPUT_BEGIN = "--- OUTPUT_JSON_BEGIN ---"
OUTPUT_END = "--- OUTPUT_JSON_END ---"


@dataclass
class ChunkAnnotation:
    """Represents a single chunk annotation for fine-tuning.
    
    Attributes:
        chunk_index: The index of this chunk (0-based).
        input_text: The raw input text for this chunk.
        output: The expected JSON output for this chunk.
        source_file: Optional source file name for easy identification.
        source_path: Optional full path to the source file.
    """
    chunk_index: int
    input_text: str
    output: Optional[Dict[str, Any]] = None
    source_file: Optional[str] = None
    source_path: Optional[str] = None


def read_annotations_txt(path: Path) -> List[ChunkAnnotation]:
    """
    Read annotations from an editable text file.
    
    Format:
    === chunk 0 | source_file.txt ===
    --- SOURCE_FILE ---
    C:/path/to/source_file.txt
    --- INPUT_TEXT_BEGIN ---
    [input text content]
    --- INPUT_TEXT_END ---
    --- OUTPUT_JSON_BEGIN ---
    {
      "contains_no_content_of_requested_type": false,
      "entries": [...]
    }
    --- OUTPUT_JSON_END ---
    
    Args:
        path: Path to the editable text file.
        
    Returns:
        List of ChunkAnnotation objects.
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    annotations: List[ChunkAnnotation] = []

    current_chunk: Optional[int] = None
    current_source_file: Optional[str] = None
    current_source_path: Optional[str] = None
    input_lines: List[str] = []
    output_lines: List[str] = []
    mode: Optional[str] = None

    def _flush() -> None:
        nonlocal current_chunk, current_source_file, current_source_path
        nonlocal input_lines, output_lines, mode
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
            ChunkAnnotation(
                chunk_index=current_chunk,
                input_text=input_text,
                output=output,
                source_file=current_source_file,
                source_path=current_source_path,
            )
        )

        current_chunk = None
        current_source_file = None
        current_source_path = None
        input_lines = []
        output_lines = []
        mode = None

    for line_no, line in enumerate(lines, 1):
        header_match = CHUNK_HEADER_RE.match(line.strip())
        if header_match:
            _flush()
            current_chunk = int(header_match.group(1))
            # Extract source file from header if present
            if header_match.group(2):
                current_source_file = header_match.group(2).strip()
            continue

        if current_chunk is None:
            if not line.strip():
                continue
            raise ValueError(
                f"Line {line_no}: content found before first chunk header. "
                f"Expected '=== chunk N ===' or '=== chunk N | source_file ==='"
            )

        if line.strip() == SOURCE_INFO:
            mode = "source"
            continue
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

        if mode == "source":
            # The source line contains the full path to the source file
            if line.strip():
                current_source_path = line.strip()
        elif mode == "input":
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
    """
    Write annotations to an editable text file.
    
    The format is designed for easy editing in Notepad++:
    - Clear chunk headers with source file name for identification
    - Full source path so annotator can open the original file
    - Pretty-printed JSON output (2-space indent) for easy editing
    
    Args:
        annotations: List of ChunkAnnotation objects.
        path: Output path for the editable text file.
        pretty_json: If True, pretty-print JSON with indentation (default).
    """
    out_lines: List[str] = []
    
    for ann in sorted(annotations, key=lambda a: a.chunk_index):
        # Header includes source file name for easy identification
        if ann.source_file:
            out_lines.append(f"=== chunk {ann.chunk_index} | {ann.source_file} ===")
        else:
            out_lines.append(f"=== chunk {ann.chunk_index} ===")
        
        # Source path section - allows annotator to locate original file
        out_lines.append(SOURCE_INFO)
        if ann.source_path:
            out_lines.append(ann.source_path)
        elif ann.source_file:
            out_lines.append(f"[Source file: {ann.source_file}]")
        else:
            out_lines.append("[Source path not available]")
        
        # Input text section
        out_lines.append(INPUT_BEGIN)
        out_lines.extend((ann.input_text or "").splitlines())
        out_lines.append(INPUT_END)
        
        # Output JSON section - pretty-printed for Notepad++ editing
        out_lines.append(OUTPUT_BEGIN)
        if ann.output is not None:
            if pretty_json:
                # Use 2-space indent for easy reading in Notepad++
                out_lines.append(json.dumps(
                    ann.output,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ))
            else:
                out_lines.append(json.dumps(
                    ann.output,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ))
        out_lines.append(OUTPUT_END)
        
        # Blank line between chunks for readability
        out_lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out_lines).rstrip() + "\n", encoding="utf-8")
