#!/usr/bin/env python3
"""
Convert research assistant ground truth files from editable text format
to chunk-size-specific JSONL ground truth files for ChronoMiner evaluation.

Handles JSON malformation found in the RA files:
- Unquoted string values (e.g., "last_name": Abderhalden-Bischof,)
- Trailing commas before } or ]
- Missing commas between objects in arrays (}\n{)
- Missing commas between properties

Usage:
    python eval/convert_ra_ground_truth.py
"""

from __future__ import annotations

import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.jsonl_eval import (
    ChunkExtraction,
    DocumentExtractions,
    write_ground_truth_jsonl,
)

ZIP_PATH = Path(
    r"C:\Users\pagoetz\Documents\Arbeit\Doktor\01_Forschung"
    r"\Papers\ProcessingThePast\Auftrag json.zip"
)

GT_OUTPUT_DIR = SCRIPT_DIR / "test_data" / "ground_truth"

RA_FILE_MAP: Dict[str, Tuple[str, str, int]] = {
    "address_book_2500_edited_fertig.txt": (
        "address_books", "address_books", 2500
    ),
    "address_book_5000_edited_fertig.txt": (
        "address_books", "address_books", 5000
    ),
    "bibliography_2500_edited_fertig.txt": (
        "bibliography",
        "Whitaker_1913_English_Cookery_Books_to_the_Year_1850",
        2500,
    ),
    "bibliography_5000_edited_fertig.txt": (
        "bibliography",
        "Whitaker_1913_English_Cookery_Books_to_the_Year_1850",
        5000,
    ),
    "antonio_franco_2500_edited_fertig.txt": (
        "military_records", "Antonio Franco", 2500
    ),
    "antonio_franco_5000_edited_fertig.txt": (
        "military_records", "Antonio Franco", 5000
    ),
    "carlos_schmidt_2500_edited_fertig.txt": (
        "military_records", "Carlos Schimidt", 2500
    ),
    "carlos_schmidt_5000_edited_fertig.txt": (
        "military_records", "Carlos Schimidt", 5000
    ),
    "elza_elias_2500_edited_fertig.txt": (
        "military_records", "Elza Elias", 2500
    ),
    "elza_elias_5000_edited_fertig.txt": (
        "military_records", "Elza Elias", 5000
    ),
}


def repair_json(text: str) -> str:
    """
    Apply multi-pass repairs to fix JSON malformation from human editing.

    Known error patterns in the RA ground truth files:
    - Unquoted string values ("last_name": Abderhalden-Bischof,)
    - Trailing commas before } or ]
    - Missing commas between } and { in arrays
    - Missing commas between properties on adjacent lines
    - Stray quote before null ("occupation": "null,  ->  null,)
    - Unescaped quotes inside string values
    - Raw newlines inside string values (string not closed on same line)
    - Missing closing quote before comma ("name": "T.C,  ->  "T.C.",)
    """
    # Pass 1: Fix stray quote before null/true/false
    # "key": "null,  ->  "key": null,
    text = re.sub(
        r'("[\w.]+"\s*:\s*)"(null|true|false)([,\s}\]])',
        r'\1\2\3',
        text,
    )

    # Pass 2: Remove trailing commas before } or ]
    text = re.sub(r",(\s*[}\]])", r"\1", text)

    # Pass 3: Insert missing commas between } and { in arrays
    text = re.sub(r"(\})\s*\n(\s*)\{", r"\1,\n\2{", text)

    # Pass 4: Fix strings with raw newlines (string not terminated on line)
    # Detect: "key": "value...<newline>  "next_key":
    # Fix: close the string, add comma
    def fix_unterminated_string(match: re.Match) -> str:
        key_part = match.group(1)
        value_part = match.group(2).rstrip()
        # If value ends with a period or letter, close it
        return f'{key_part}"{value_part}",\n'

    text = re.sub(
        r'("[\w.]+"\s*:\s*")'          # "key": "
        r'([^"\n]{3,})'                 # value without closing quote (3+ chars)
        r'\s*\n'                        # newline (string not closed)
        r'(?=\s*"[\w.]+"\s*:)',         # lookahead: next line has "key":
        fix_unterminated_string,
        text,
    )

    # Pass 5: Fix missing closing quote before comma
    # "name": "T.C,  ->  "name": "T.C.",
    # Detect: "key": "value<no-closing-quote><comma-or-newline>
    text = re.sub(
        r'("[\w.]+"\s*:\s*")'          # "key": "
        r'([^"\n]{1,50}?)'             # short value without quote
        r'(?=\n\s*"[\w.]+"\s*:)',       # followed by next property (no comma)
        lambda m: f'{m.group(1)}{m.group(2).rstrip()}",',
        text,
    )

    # Pass 6: Quote unquoted string values
    def quote_unquoted(match: re.Match) -> str:
        prefix = match.group(1)
        value = match.group(2)
        suffix = match.group(3)

        stripped = value.strip()
        if stripped in ("null", "true", "false"):
            return match.group(0)
        try:
            float(stripped)
            return match.group(0)
        except ValueError:
            pass

        escaped = stripped.replace("\\", "\\\\").replace('"', '\\"')
        return f'{prefix}"{escaped}"{suffix}'

    text = re.sub(
        r'("[\w.]+"\s*:\s*)'
        r'([A-Za-z\u00C0-\u024F*][^\n\r]*?)'
        r'(\s*[,}\]\n])',
        quote_unquoted,
        text,
    )

    # Pass 7: Insert missing commas between adjacent properties
    # Detect: value\n  "key":  where no comma after value
    # Handles: null\n"key", "value"\n"key", true\n"key", 123\n"key"
    text = re.sub(
        r'((?:null|true|false|\d+|"[^"]*"))\s*\n(\s*"[\w.]+"\s*:)',
        r'\1,\n\2',
        text,
    )

    # Pass 7b: Insert missing closing brace before new array element
    # Detect: property line (6+ spaces) followed by { line (4 spaces)
    # without a closing } in between. This means an entry object was
    # not properly closed before the next entry starts.
    text = re.sub(
        r'("[\w.]+"\s*:\s*(?:null|true|false|\d+|"[^"]*"))\s*\n'
        r'(\s{2,4})\{',
        r'\1\n\2},\n\2{',
        text,
    )

    # Pass 8: Fix specific unescaped internal quotes in string values.
    # Line-by-line: if a line has "key": "value" structure but the value
    # portion contains unescaped quotes that break JSON, escape them.
    lines = text.split("\n")
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Match "key": "value" lines
        m = re.match(r'^(\s*"[\w.]+"\s*:\s*)"(.+)"(\s*,?\s*)$', line)
        if not m:
            continue
        prefix = m.group(1)
        inner = m.group(2)
        suffix = m.group(3)
        # Check if inner part contains unescaped quotes
        if '"' in inner and '\\"' not in inner:
            escaped_inner = inner.replace('"', '\\"')
            lines[i] = f'{prefix}"{escaped_inner}"{suffix}'
    text = "\n".join(lines)

    return text


def parse_chunks_from_text(content: str) -> List[Tuple[int, str]]:
    """
    Parse chunk markers and their JSON content from editable text format.

    Returns list of (chunk_index, json_string) tuples.
    """
    chunk_pattern = re.compile(r"===\s*chunk\s+(\d+)\s*===", re.IGNORECASE)
    markers = list(chunk_pattern.finditer(content))

    chunks = []
    for i, match in enumerate(markers):
        chunk_index = int(match.group(1))
        start = match.end()
        end = markers[i + 1].start() if i + 1 < len(markers) else len(content)

        chunk_text = content[start:end].strip()

        # Remove comment lines
        lines = []
        for line in chunk_text.split("\n"):
            if not line.strip().startswith("#"):
                lines.append(line)
        json_str = "\n".join(lines).strip()

        chunks.append((chunk_index, json_str))

    return chunks


def convert_file(
    filename: str,
    content: str,
    category: str,
    source_name: str,
    chunk_size: int,
) -> Optional[DocumentExtractions]:
    """
    Convert a single RA editable text file to a DocumentExtractions object.

    Applies JSON repair and validates each chunk.
    """
    chunks = parse_chunks_from_text(content)
    if not chunks:
        print(f"  [WARN] No chunks found in {filename}")
        return None

    doc = DocumentExtractions(source_name=source_name)
    errors = []

    for chunk_index, raw_json in chunks:
        # Apply repair
        repaired = repair_json(raw_json)

        try:
            data = json.loads(repaired)
        except json.JSONDecodeError as e:
            errors.append((chunk_index, e, repaired))
            # Store with parse_error flag so we know it needs manual fix
            data = {"parse_error": str(e), "raw_content": repaired}

        chunk = ChunkExtraction(
            chunk_index=chunk_index,
            custom_id=f"{source_name}-chunk-{chunk_index}",
            extraction_data=data,
        )
        doc.chunks.append(chunk)

    doc.chunks.sort(key=lambda c: c.chunk_index)

    if errors:
        print(f"  [ERROR] {len(errors)} chunk(s) still have JSON errors:")
        for cidx, err, text in errors:
            err_lines = text.split("\n")
            err_line = (
                err_lines[err.lineno - 1].strip()[:100]
                if err.lineno <= len(err_lines)
                else "???"
            )
            print(f"    chunk {cidx:03d} L{err.lineno}: {err.msg}")
            print(f"      >> {err_line.encode('ascii', 'replace').decode()}")

    return doc


def count_entries(doc: DocumentExtractions) -> int:
    """Count total entries across all chunks."""
    total = 0
    for chunk in doc.chunks:
        entries = chunk.extraction_data.get("entries", [])
        if isinstance(entries, list):
            total += len(entries)
    return total


def main() -> int:
    if not ZIP_PATH.exists():
        print(f"[ERROR] Zip file not found: {ZIP_PATH}")
        return 1

    print(f"[INFO] Reading zip: {ZIP_PATH.name}")
    z = zipfile.ZipFile(ZIP_PATH)

    GT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_files = 0
    total_errors = 0

    for ra_filename, (category, source_name, chunk_size) in sorted(
        RA_FILE_MAP.items()
    ):
        if ra_filename not in z.namelist():
            print(f"[WARN] File not found in zip: {ra_filename}")
            continue

        content = z.read(ra_filename).decode("utf-8")
        print(f"\n[INFO] Processing: {ra_filename}")
        print(f"  Category: {category}, Source: {source_name}, Chunk size: {chunk_size}")

        doc = convert_file(
            ra_filename, content, category, source_name, chunk_size
        )
        if doc is None:
            continue

        # Check for parse errors
        has_errors = any(
            "parse_error" in c.extraction_data for c in doc.chunks
        )
        if has_errors:
            total_errors += 1
            print(f"  [FAIL] Skipping write due to parse errors")
            continue

        # Write ground truth JSONL
        gt_dir = GT_OUTPUT_DIR / category
        gt_path = gt_dir / f"{source_name}_c{chunk_size}.jsonl"
        write_ground_truth_jsonl(doc, gt_path)

        entry_count = count_entries(doc)
        print(
            f"  [OK] Written: {gt_path.relative_to(SCRIPT_DIR)}"
            f" ({doc.chunk_count()} chunks, {entry_count} entries)"
        )
        total_files += 1

    print(f"\n{'=' * 60}")
    print(f"Conversion complete: {total_files} files written, {total_errors} failed")

    if total_errors > 0:
        print("[WARN] Some files had unresolvable JSON errors.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
