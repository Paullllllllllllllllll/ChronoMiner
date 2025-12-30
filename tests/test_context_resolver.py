"""Tests for the unified context resolver."""

from pathlib import Path
import pytest
from modules.core.context_resolver import (
    resolve_context_for_extraction,
    resolve_context_for_readjustment,
)


@pytest.mark.unit
def test_resolve_extraction_context_schema_specific(tmp_path):
    """Test schema-specific extraction context resolution."""
    # Create context structure
    context_dir = tmp_path / "context"
    extraction_dir = context_dir / "extraction"
    extraction_dir.mkdir(parents=True)
    
    # Create schema-specific context
    schema_file = extraction_dir / "TestSchema.txt"
    schema_file.write_text("Schema-specific extraction context", encoding="utf-8")
    
    # Resolve context
    context, path = resolve_context_for_extraction(
        schema_name="TestSchema",
        global_context_dir=context_dir,
    )
    
    assert context == "Schema-specific extraction context"
    assert path == schema_file


@pytest.mark.unit
def test_resolve_extraction_context_file_specific(tmp_path):
    """Test file-specific extraction context takes precedence."""
    # Create context structure
    context_dir = tmp_path / "context"
    extraction_dir = context_dir / "extraction"
    extraction_dir.mkdir(parents=True)
    
    # Create schema-specific context
    schema_file = extraction_dir / "TestSchema.txt"
    schema_file.write_text("Schema-specific extraction context", encoding="utf-8")
    
    # Create file-specific context
    text_file = tmp_path / "test_file.txt"
    text_file.write_text("Some text", encoding="utf-8")
    file_context = tmp_path / "test_file_extraction.txt"
    file_context.write_text("File-specific extraction context", encoding="utf-8")
    
    # Resolve context
    context, path = resolve_context_for_extraction(
        schema_name="TestSchema",
        text_file=text_file,
        global_context_dir=context_dir,
    )
    
    assert context == "File-specific extraction context"
    assert path == file_context


@pytest.mark.unit
def test_resolve_readjustment_context_boundary_specific(tmp_path):
    """Test boundary-type-specific readjustment context resolution."""
    # Create context structure
    context_dir = tmp_path / "context"
    line_ranges_dir = context_dir / "line_ranges"
    line_ranges_dir.mkdir(parents=True)
    
    # Create boundary-type-specific context
    boundary_file = line_ranges_dir / "TestBoundary.txt"
    boundary_file.write_text("Boundary-specific readjustment context", encoding="utf-8")
    
    # Resolve context
    context, path = resolve_context_for_readjustment(
        boundary_type="TestBoundary",
        global_context_dir=context_dir,
    )
    
    assert context == "Boundary-specific readjustment context"
    assert path == boundary_file


@pytest.mark.unit
def test_resolve_extraction_context_no_context(tmp_path):
    """Test extraction context when no context exists."""
    context_dir = tmp_path / "context"
    
    context, path = resolve_context_for_extraction(
        schema_name="NonExistent",
        global_context_dir=context_dir,
    )
    
    assert context is None
    assert path is None


@pytest.mark.unit
def test_resolve_extraction_context_global_fallback(tmp_path):
    """Test extraction context falls back to general.txt."""
    # Create context structure
    context_dir = tmp_path / "context"
    extraction_dir = context_dir / "extraction"
    extraction_dir.mkdir(parents=True)
    
    # Create only general context
    general_file = extraction_dir / "general.txt"
    general_file.write_text("General extraction context", encoding="utf-8")
    
    # Resolve context for non-existent schema
    context, path = resolve_context_for_extraction(
        schema_name="NonExistent",
        global_context_dir=context_dir,
    )
    
    assert context == "General extraction context"
    assert path == general_file
