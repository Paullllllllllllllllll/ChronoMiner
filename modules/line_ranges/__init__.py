"""ChronoMiner line-ranges workflow package.

Two user-visible workflows: (a) token-based line-range generation in
:mod:`modules.line_ranges.generator` (range computation, sidecar
writing, interactive file selection, and the per-file processing loop),
and (b) LLM-assisted semantic boundary readjustment on existing
``_line_ranges.txt`` files in :mod:`modules.line_ranges.readjuster`.
"""

from modules.line_ranges.generator import (
    generate_line_ranges_for_file,
    process_files,
    select_input_files,
    write_line_ranges_file,
)
from modules.line_ranges.readjuster import (
    LineRangeReadjuster,
    ReadjustmentInterrupted,
)

__all__ = [
    "LineRangeReadjuster",
    "ReadjustmentInterrupted",
    "generate_line_ranges_for_file",
    "process_files",
    "select_input_files",
    "write_line_ranges_file",
]
