from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


CHUNK_HEADER_RE = re.compile(r"^===\s*chunk\s+(\d+)\s*===\s*$", re.IGNORECASE)


@dataclass
class ChunkInput:
    chunk_index: int
    input_text: str


def read_chunk_inputs_txt(path: Path) -> List[ChunkInput]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    chunks: List[ChunkInput] = []
    current_chunk: Optional[int] = None
    buf: List[str] = []

    def _flush() -> None:
        nonlocal current_chunk, buf
        if current_chunk is None:
            return
        chunks.append(
            ChunkInput(
                chunk_index=current_chunk,
                input_text="\n".join(buf).strip("\n"),
            )
        )
        current_chunk = None
        buf = []

    for line_no, line in enumerate(lines, 1):
        m = CHUNK_HEADER_RE.match(line.strip())
        if m:
            _flush()
            current_chunk = int(m.group(1))
            continue

        if current_chunk is None:
            if not line.strip():
                continue
            raise ValueError(
                f"Line {line_no}: content found before first chunk header. Expected '=== chunk N ==='"
            )

        buf.append(line)

    _flush()

    if not chunks:
        raise ValueError("No chunks found")

    chunks.sort(key=lambda c: c.chunk_index)
    seen = set()
    duplicates = [c.chunk_index for c in chunks if (c.chunk_index in seen) or seen.add(c.chunk_index)]
    if duplicates:
        dup_str = ", ".join(str(x) for x in sorted(set(duplicates)))
        raise ValueError(f"Duplicate chunk indices found: {dup_str}")

    return chunks
