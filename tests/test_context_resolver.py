"""Tests for the unified context resolver.

Tests the 3-level hierarchy:
1. File-specific:   {input_stem}_{suffix}.txt   next to the input file
2. Folder-specific: {parent_folder}_{suffix}.txt next to the input's parent folder
3. General fallback: context/{suffix}.txt        in the project context directory

Suffixes: extract_context (extraction), adjust_context (line-range readjustment)
"""

from pathlib import Path
import pytest
from modules.core.context_resolver import (
    _resolve_context,
    resolve_context_for_extraction,
    resolve_context_for_readjustment,
    _read_and_validate_context,
)


# ---------------------------------------------------------------------------
# _resolve_context (generic)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_resolve_context_file_specific(tmp_path):
    """File-specific context is found and returned."""
    text_file = tmp_path / "myfile.txt"
    text_file.write_text("body", encoding="utf-8")
    ctx = tmp_path / "myfile_extract_context.txt"
    ctx.write_text("file-level ctx", encoding="utf-8")

    content, path = _resolve_context("extract_context", text_file=text_file, context_dir=tmp_path)
    assert content == "file-level ctx"
    assert path == ctx


@pytest.mark.unit
def test_resolve_context_folder_specific(tmp_path):
    """Folder-specific context is found when no file-specific exists."""
    subfolder = tmp_path / "archive"
    subfolder.mkdir()
    text_file = subfolder / "myfile.txt"
    text_file.write_text("body", encoding="utf-8")
    ctx = tmp_path / "archive_extract_context.txt"
    ctx.write_text("folder-level ctx", encoding="utf-8")

    content, path = _resolve_context("extract_context", text_file=text_file, context_dir=tmp_path)
    assert content == "folder-level ctx"
    assert path == ctx


@pytest.mark.unit
def test_resolve_context_general_fallback(tmp_path):
    """General fallback is used when no file/folder context exists."""
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    general = context_dir / "extract_context.txt"
    general.write_text("general ctx", encoding="utf-8")

    text_file = tmp_path / "some" / "deep" / "file.txt"
    text_file.parent.mkdir(parents=True)
    text_file.write_text("body", encoding="utf-8")

    content, path = _resolve_context("extract_context", text_file=text_file, context_dir=context_dir)
    assert content == "general ctx"
    assert path == general


@pytest.mark.unit
def test_resolve_context_file_wins_over_folder(tmp_path):
    """File-specific context takes precedence over folder-specific."""
    subfolder = tmp_path / "archive"
    subfolder.mkdir()
    text_file = subfolder / "myfile.txt"
    text_file.write_text("body", encoding="utf-8")

    file_ctx = subfolder / "myfile_adjust_context.txt"
    file_ctx.write_text("file wins", encoding="utf-8")
    folder_ctx = tmp_path / "archive_adjust_context.txt"
    folder_ctx.write_text("folder loses", encoding="utf-8")

    content, path = _resolve_context("adjust_context", text_file=text_file, context_dir=tmp_path)
    assert content == "file wins"
    assert path == file_ctx


@pytest.mark.unit
def test_resolve_context_folder_wins_over_general(tmp_path):
    """Folder-specific context takes precedence over general fallback."""
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    general = context_dir / "extract_context.txt"
    general.write_text("general ctx", encoding="utf-8")

    subfolder = tmp_path / "archive"
    subfolder.mkdir()
    text_file = subfolder / "myfile.txt"
    text_file.write_text("body", encoding="utf-8")
    folder_ctx = tmp_path / "archive_extract_context.txt"
    folder_ctx.write_text("folder wins", encoding="utf-8")

    content, path = _resolve_context("extract_context", text_file=text_file, context_dir=context_dir)
    assert content == "folder wins"
    assert path == folder_ctx


@pytest.mark.unit
def test_resolve_context_no_context(tmp_path):
    """Returns (None, None) when no context exists anywhere."""
    text_file = tmp_path / "myfile.txt"
    text_file.write_text("body", encoding="utf-8")

    content, path = _resolve_context("extract_context", text_file=text_file, context_dir=tmp_path)
    assert content is None
    assert path is None


@pytest.mark.unit
def test_resolve_context_no_text_file(tmp_path):
    """General fallback is used when no text_file is provided."""
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    general = context_dir / "adjust_context.txt"
    general.write_text("fallback only", encoding="utf-8")

    content, path = _resolve_context("adjust_context", text_file=None, context_dir=context_dir)
    assert content == "fallback only"
    assert path == general


@pytest.mark.unit
def test_resolve_context_empty_file_skipped(tmp_path):
    """Empty context files are skipped."""
    text_file = tmp_path / "myfile.txt"
    text_file.write_text("body", encoding="utf-8")
    empty_ctx = tmp_path / "myfile_extract_context.txt"
    empty_ctx.write_text("", encoding="utf-8")

    content, path = _resolve_context("extract_context", text_file=text_file, context_dir=tmp_path)
    assert content is None
    assert path is None


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_resolve_context_for_extraction(tmp_path):
    """resolve_context_for_extraction uses 'extract_context' suffix."""
    text_file = tmp_path / "data.txt"
    text_file.write_text("body", encoding="utf-8")
    ctx = tmp_path / "data_extract_context.txt"
    ctx.write_text("extraction ctx", encoding="utf-8")

    content, path = resolve_context_for_extraction(text_file=text_file, context_dir=tmp_path)
    assert content == "extraction ctx"
    assert path == ctx


@pytest.mark.unit
def test_resolve_context_for_readjustment(tmp_path):
    """resolve_context_for_readjustment uses 'adjust_context' suffix."""
    text_file = tmp_path / "data.txt"
    text_file.write_text("body", encoding="utf-8")
    ctx = tmp_path / "data_adjust_context.txt"
    ctx.write_text("adjust ctx", encoding="utf-8")

    content, path = resolve_context_for_readjustment(text_file=text_file, context_dir=tmp_path)
    assert content == "adjust ctx"
    assert path == ctx


# ---------------------------------------------------------------------------
# _read_and_validate_context
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_read_and_validate_context_valid(tmp_path):
    """Valid file returns stripped content."""
    ctx = tmp_path / "ctx.txt"
    ctx.write_text("  hello world  \n", encoding="utf-8")
    assert _read_and_validate_context(ctx) == "hello world"


@pytest.mark.unit
def test_read_and_validate_context_empty(tmp_path):
    """Empty file returns None."""
    ctx = tmp_path / "ctx.txt"
    ctx.write_text("", encoding="utf-8")
    assert _read_and_validate_context(ctx) is None


@pytest.mark.unit
def test_read_and_validate_context_whitespace_only(tmp_path):
    """Whitespace-only file returns None."""
    ctx = tmp_path / "ctx.txt"
    ctx.write_text("   \n\t  ", encoding="utf-8")
    assert _read_and_validate_context(ctx) is None


@pytest.mark.unit
def test_read_and_validate_context_missing_file(tmp_path):
    """Non-existent file returns None."""
    result = _read_and_validate_context(tmp_path / "missing.txt")
    assert result is None
