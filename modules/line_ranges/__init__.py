"""ChronoMiner line-ranges workflow package.

Two user-visible workflows: (a) token-based line-range generation, and
(b) LLM-assisted semantic boundary readjustment on existing
``_line_ranges.txt`` files. The readjustment pipeline (large and
self-contained) lives in :mod:`modules.line_ranges.readjuster`; the
generator helpers currently remain in ``main/generate_line_ranges.py``
and will be extracted in a follow-up step.
"""

from modules.line_ranges.generator import (
    generate_line_ranges_for_file,
    write_line_ranges_file,
)
from modules.line_ranges.readjuster import LineRangeReadjuster

__all__ = [
    "LineRangeReadjuster",
    "generate_line_ranges_for_file",
    "write_line_ranges_file",
]
