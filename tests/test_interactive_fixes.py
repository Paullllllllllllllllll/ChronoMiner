# tests/test_interactive_fixes.py

"""Regression tests for the interactive-mode and reporting fixes.

Covers:

* FIX 1 -- interactive completion summary aggregates real per-file statuses
  (a failed file must raise the failed count, not be silently reported as a
  success).
* FIX 2 -- ``display_completion_summary`` shows a per-status breakdown, names
  failed/partial files, prints the this-run token delta, and prints the
  success line only when nothing failed or partial-ed.
* FIX 3 -- back navigation does not bounce off pass-through steps (text input:
  'b' at chunk_slice reaches the context step, not chunk_slice again).
* FIX 4/5 -- interactive discovery includes ``.md`` files, excludes the tool's
  own ``_output.txt`` reports, and a mixed folder discovers text files.
* FIX 11 -- ``generate_line_ranges`` honors a page-range slice.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from modules.infra.chunking import ChunkSlice
from modules.ui.core import UserInterface

# ---------------------------------------------------------------------------
# FIX 2: display_completion_summary
# ---------------------------------------------------------------------------


def _capture_summary(**kwargs: Any) -> str:
    """Run display_completion_summary and return the captured console text."""
    ui = UserInterface(use_colors=False)
    lines: list[str] = []
    ui.console_print = lambda message, log_also=False: lines.append(message)  # type: ignore[method-assign]
    ui.print_success = lambda message: lines.append(message)  # type: ignore[method-assign]
    ui.print_warning = lambda message: lines.append(message)  # type: ignore[method-assign]
    ui.print_section_header = lambda title: lines.append(title)  # type: ignore[method-assign]
    ui.display_completion_summary(**kwargs)
    return "\n".join(lines)


@pytest.mark.unit
def test_summary_success_only_when_clean() -> None:
    text = _capture_summary(
        processed_count=3,
        failed_count=0,
        use_batch=False,
        complete_count=3,
        partial_count=0,
        skipped_count=0,
    )
    assert "All 3 file(s) processed successfully!" in text


@pytest.mark.unit
def test_summary_no_success_line_when_failure() -> None:
    text = _capture_summary(
        processed_count=1,
        failed_count=1,
        use_batch=False,
        complete_count=1,
        partial_count=0,
        skipped_count=0,
        failed_files=["broken.txt"],
    )
    assert "processed successfully" not in text
    assert "Failed:" in text
    assert "broken.txt" in text


@pytest.mark.unit
def test_summary_no_success_line_when_partial() -> None:
    text = _capture_summary(
        processed_count=1,
        failed_count=0,
        use_batch=False,
        complete_count=1,
        partial_count=1,
        skipped_count=0,
        partial_files=["half.txt"],
    )
    assert "processed successfully" not in text
    assert "Partial:" in text
    assert "half.txt" in text
    # A resume pointer must be present for partial files.
    assert "resume" in text.lower()


@pytest.mark.unit
def test_summary_shows_token_delta() -> None:
    text = _capture_summary(
        processed_count=2,
        failed_count=0,
        use_batch=False,
        complete_count=2,
        tokens_this_run=12_345,
        daily_tokens_used=50_000,
        daily_token_limit=1_000_000,
    )
    assert "Tokens consumed this run: 12,345" in text
    assert "50,000/1,000,000" in text


@pytest.mark.unit
def test_summary_token_section_absent_when_disabled() -> None:
    text = _capture_summary(
        processed_count=1,
        failed_count=0,
        use_batch=False,
        complete_count=1,
        tokens_this_run=None,
    )
    assert "Tokens consumed this run" not in text


# ---------------------------------------------------------------------------
# FIX 4/5: interactive file discovery
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_discover_includes_md_and_excludes_output_txt(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    (tmp_path / "b.md").write_text("x", encoding="utf-8")
    (tmp_path / "a_output.txt").write_text("x", encoding="utf-8")
    (tmp_path / "a_context.txt").write_text("x", encoding="utf-8")
    (tmp_path / "a_line_ranges.txt").write_text("x", encoding="utf-8")

    ui = UserInterface(use_colors=False)
    found = {p.name for p in ui._discover_files(tmp_path, {".txt"}, is_visual=False)}
    assert found == {"a.txt", "b.md"}


@pytest.mark.unit
def test_discover_mixed_includes_text_and_visual(tmp_path: Path) -> None:
    from modules.config.constants import SUPPORTED_VISUAL_EXTENSIONS

    (tmp_path / "page.txt").write_text("x", encoding="utf-8")
    (tmp_path / "notes.md").write_text("x", encoding="utf-8")
    (tmp_path / "scan.pdf").write_text("x", encoding="utf-8")
    (tmp_path / "scan_output.txt").write_text("x", encoding="utf-8")

    ui = UserInterface(use_colors=False)
    mixed_exts = SUPPORTED_VISUAL_EXTENSIONS | {".txt", ".md"}
    found = {p.name for p in ui._discover_files(tmp_path, mixed_exts, is_visual=True)}
    assert "page.txt" in found
    assert "notes.md" in found
    assert "scan.pdf" in found
    # The tool's own report must not be re-ingested.
    assert "scan_output.txt" not in found


# ---------------------------------------------------------------------------
# FIX 11: generate_line_ranges honors a page-range slice
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_generate_line_ranges_page_range_slice(tmp_path: Path, monkeypatch) -> None:
    import main.generate_line_ranges as glr

    ranges = [(1, 10), (11, 20), (21, 30), (31, 40)]
    monkeypatch.setattr(
        glr, "generate_line_ranges_for_file", lambda **_kw: list(ranges)
    )
    written: dict[str, Any] = {}

    def _fake_write(file_path: Path, line_ranges: list[tuple[int, int]]) -> Path:
        written["ranges"] = line_ranges
        return file_path.with_name(f"{file_path.stem}_line_ranges.txt")

    monkeypatch.setattr(glr, "write_line_ranges_file", _fake_write)

    script = glr.GenerateLineRangesScript()
    script.tokens_per_chunk = 500
    script.model_name = "gpt-4o"
    script.ui = None

    input_file = tmp_path / "doc.txt"
    input_file.write_text("data", encoding="utf-8")

    success, fail = script._process_files(
        [input_file], verbose=False, chunk_slice=ChunkSlice(page_range=(2, 3))
    )
    assert (success, fail) == (1, 0)
    assert written["ranges"] == [(11, 20), (21, 30)]


# ---------------------------------------------------------------------------
# FIX 1 / FIX 3: interactive state machine (aggregation + back navigation)
# ---------------------------------------------------------------------------


class _ScriptedUI:
    """A UserInterface stand-in that replays scripted prompt answers.

    Each prompt method pops its next answer from a per-method list and records
    the call name so tests can assert navigation order.
    """

    HORIZONTAL_LINE = "-" * 10

    def __init__(self, answers: dict[str, list[Any]]) -> None:
        self._answers = {k: list(v) for k, v in answers.items()}
        self.calls: list[str] = []
        self.completion_kwargs: dict[str, Any] | None = None

    def _pop(self, name: str) -> Any:
        self.calls.append(name)
        return self._answers[name].pop(0)

    # Passive output helpers -------------------------------------------------
    def print_info(self, *a: Any, **k: Any) -> None: ...
    def print_error(self, *a: Any, **k: Any) -> None: ...
    def print_warning(self, *a: Any, **k: Any) -> None: ...
    def print_success(self, *a: Any, **k: Any) -> None: ...
    def print_section_header(self, *a: Any, **k: Any) -> None: ...
    def log(self, *a: Any, **k: Any) -> None: ...

    # Prompt methods ---------------------------------------------------------
    def select_schema(self, _mgr: Any, allow_back: bool = False) -> Any:
        return self._pop("select_schema")

    def ask_global_chunking_mode(self, allow_back: bool = False) -> Any:
        return self._pop("ask_global_chunking_mode")

    def ask_batch_processing(self, allow_back: bool = False) -> Any:
        return self._pop("ask_batch_processing")

    def select_option(self, *a: Any, **k: Any) -> Any:
        return self._pop("select_option")

    def ask_context_selection(self, allow_back: bool = False) -> Any:
        return self._pop("ask_context_selection")

    def ask_context_image(self, allow_back: bool = False) -> Any:
        return self._pop("ask_context_image")

    def ask_image_detail(self, allow_back: bool = False) -> Any:
        return self._pop("ask_image_detail")

    def ask_chunk_slice(self, allow_back: bool = False) -> Any:
        return self._pop("ask_chunk_slice")

    def select_input_source(self, *a: Any, **k: Any) -> Any:
        return self._pop("select_input_source")

    def display_processing_summary(self, *a: Any, **k: Any) -> bool:
        self.calls.append("display_processing_summary")
        return True

    def display_completion_summary(self, **kwargs: Any) -> None:
        self.calls.append("display_completion_summary")
        self.completion_kwargs = kwargs


class _FakeFileProcessor:
    """Returns a scripted status per file name."""

    def __init__(self, status_map: dict[str, str], **_kw: Any) -> None:
        self._status_map = status_map

    async def process_file(self, *, file_path: Path, **_kw: Any) -> str:
        return self._status_map.get(file_path.name, "complete")


class _FakeStdin:
    def isatty(self) -> bool:
        return True


def _install_common_patches(monkeypatch, status_map: dict[str, str]) -> None:
    import main.process_text_files as ptf

    monkeypatch.setattr(ptf.sys, "stdin", _FakeStdin())
    monkeypatch.setattr(ptf, "check_token_limit_enabled", lambda: False)
    monkeypatch.setattr(ptf, "detect_input_type", lambda _p: "text")

    class _FakeConfigManager:
        def __init__(self, *_a: Any, **_k: Any) -> None: ...

        def validate_paths(self, _paths: Any) -> None: ...

    monkeypatch.setattr(ptf, "ConfigManager", _FakeConfigManager)

    class _FakeSchemaManager:
        def get_available_schemas(self) -> dict[str, Any]:
            return {"TestSchema": {"schema": True}}

    monkeypatch.setattr(ptf, "load_schema_manager", lambda: _FakeSchemaManager())
    monkeypatch.setattr(
        ptf,
        "FileProcessor",
        lambda **kw: _FakeFileProcessor(status_map, **kw),
    )


def _run(monkeypatch, ui: _ScriptedUI, tmp_path: Path) -> None:
    import asyncio

    import main.process_text_files as ptf

    monkeypatch.setattr(ptf, "UserInterface", lambda *_a, **_k: ui)

    class _FakeConfigLoader:
        def get_concurrency_config(self) -> dict[str, Any]:
            return {}

    input_dir = tmp_path / "in"
    input_dir.mkdir()
    schemas_paths = {
        "TestSchema": {"input": str(input_dir), "output": str(tmp_path / "out")}
    }
    paths_config = {
        "general": {"input_paths_is_output_path": True},
        "schemas_paths": schemas_paths,
    }
    asyncio.run(
        ptf._run_interactive_mode(
            _FakeConfigLoader(),
            paths_config,
            {"extraction_model": {"name": "gpt-4o"}},
            {"chunking": {}},
            schemas_paths,
        )
    )


@pytest.mark.unit
def test_interactive_summary_counts_failed_file(monkeypatch, tmp_path: Path) -> None:
    """A failed file must be reflected in the completion summary (FIX 1)."""
    _install_common_patches(monkeypatch, {"a.txt": "failed", "b.txt": "complete"})
    ui = _ScriptedUI(
        {
            "select_schema": [({"schema": True}, "TestSchema")],
            "ask_global_chunking_mode": ["auto"],
            "ask_batch_processing": [False],
            "select_option": ["resume"],
            "ask_context_selection": [{"mode": "auto", "path": None}],
            "ask_chunk_slice": [ChunkSlice()],
            "select_input_source": [[Path("a.txt"), Path("b.txt")]],
        }
    )
    _run(monkeypatch, ui, tmp_path)

    assert ui.completion_kwargs is not None
    assert ui.completion_kwargs["failed_count"] == 1
    assert ui.completion_kwargs["complete_count"] == 1
    assert ui.completion_kwargs["failed_files"] == ["a.txt"]


@pytest.mark.unit
def test_interactive_back_from_chunk_slice_reaches_context(
    monkeypatch, tmp_path: Path
) -> None:
    """'b' at chunk_slice (text input) must reach the context step, not bounce
    straight back to chunk_slice (FIX 3)."""
    _install_common_patches(monkeypatch, {"a.txt": "complete"})
    ui = _ScriptedUI(
        {
            "select_schema": [({"schema": True}, "TestSchema")],
            "ask_global_chunking_mode": ["auto"],
            "ask_batch_processing": [False],
            "select_option": ["resume"],
            # context is asked twice: once forward, once after back navigation.
            "ask_context_selection": [
                {"mode": "auto", "path": None},
                {"mode": "auto", "path": None},
            ],
            # chunk_slice: first call returns None (back), second returns a slice.
            "ask_chunk_slice": [None, ChunkSlice()],
            "select_input_source": [[Path("a.txt")]],
        }
    )
    _run(monkeypatch, ui, tmp_path)

    # If the fix works, context is re-entered (asked twice) after the back.
    assert ui.calls.count("ask_context_selection") == 2
    # And the two chunk_slice prompts are not adjacent (context sits between).
    first = ui.calls.index("ask_chunk_slice")
    second = ui.calls.index("ask_chunk_slice", first + 1)
    assert "ask_context_selection" in ui.calls[first + 1 : second]
