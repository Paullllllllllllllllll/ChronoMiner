"""Extended tests for workflow utilities."""

from pathlib import Path
import json

import pytest

from modules.core.workflow_utils import (
    filter_text_files,
    collect_text_files,
    validate_schema_paths,
    load_schema_manager,
    load_core_resources,
)


class TestFilterTextFiles:
    """Tests for filter_text_files function."""
    
    @pytest.mark.unit
    def test_filters_txt_files(self, tmp_path: Path):
        """Test that .txt files are included."""
        txt_file = tmp_path / "document.txt"
        txt_file.write_text("content", encoding="utf-8")
        
        result = filter_text_files([txt_file])
        
        assert len(result) == 1
        assert result[0] == txt_file
    
    @pytest.mark.unit
    def test_excludes_non_txt_files(self, tmp_path: Path):
        """Test that non-.txt files are excluded."""
        json_file = tmp_path / "data.json"
        json_file.write_text("{}", encoding="utf-8")
        
        result = filter_text_files([json_file])
        
        assert len(result) == 0
    
    @pytest.mark.unit
    def test_excludes_line_ranges_files(self, tmp_path: Path):
        """Test that _line_ranges.txt files are excluded."""
        ranges_file = tmp_path / "document_line_ranges.txt"
        ranges_file.write_text("1, 10", encoding="utf-8")
        
        result = filter_text_files([ranges_file])
        
        assert len(result) == 0
    
    @pytest.mark.unit
    def test_excludes_context_files(self, tmp_path: Path):
        """Test that _context.txt files are excluded."""
        context_file = tmp_path / "document_context.txt"
        context_file.write_text("context", encoding="utf-8")
        
        result = filter_text_files([context_file])
        
        assert len(result) == 0
    
    @pytest.mark.unit
    def test_excludes_extraction_files(self, tmp_path: Path):
        """Test that _extraction.txt files are excluded."""
        extraction_file = tmp_path / "document_extraction.txt"
        extraction_file.write_text("extraction", encoding="utf-8")
        
        result = filter_text_files([extraction_file])
        
        assert len(result) == 0
    
    @pytest.mark.unit
    def test_excludes_directories(self, tmp_path: Path):
        """Test that directories are excluded."""
        subdir = tmp_path / "subdir.txt"  # Even if named .txt
        subdir.mkdir()
        
        result = filter_text_files([subdir])
        
        assert len(result) == 0
    
    @pytest.mark.unit
    def test_mixed_files(self, tmp_path: Path):
        """Test filtering with mixed file types."""
        valid = tmp_path / "valid.txt"
        valid.write_text("content", encoding="utf-8")
        
        ranges = tmp_path / "valid_line_ranges.txt"
        ranges.write_text("1, 10", encoding="utf-8")
        
        json_file = tmp_path / "data.json"
        json_file.write_text("{}", encoding="utf-8")
        
        result = filter_text_files([valid, ranges, json_file])
        
        assert len(result) == 1
        assert result[0] == valid


class TestCollectTextFiles:
    """Tests for collect_text_files function."""
    
    @pytest.mark.unit
    def test_collects_from_directory(self, tmp_path: Path):
        """Test collecting files from a directory."""
        file1 = tmp_path / "doc1.txt"
        file1.write_text("content1", encoding="utf-8")
        
        file2 = tmp_path / "doc2.txt"
        file2.write_text("content2", encoding="utf-8")
        
        result = collect_text_files(tmp_path)
        
        assert len(result) == 2
    
    @pytest.mark.unit
    def test_collects_from_subdirectories(self, tmp_path: Path):
        """Test recursive collection from subdirectories."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        
        file1 = tmp_path / "root.txt"
        file1.write_text("content", encoding="utf-8")
        
        file2 = subdir / "nested.txt"
        file2.write_text("content", encoding="utf-8")
        
        result = collect_text_files(tmp_path)
        
        assert len(result) == 2
    
    @pytest.mark.unit
    def test_single_file_input(self, tmp_path: Path):
        """Test with single file as input."""
        single = tmp_path / "single.txt"
        single.write_text("content", encoding="utf-8")
        
        result = collect_text_files(single)
        
        assert len(result) == 1
        assert result[0] == single
    
    @pytest.mark.unit
    def test_empty_directory(self, tmp_path: Path):
        """Test with empty directory."""
        empty = tmp_path / "empty"
        empty.mkdir()
        
        result = collect_text_files(empty)
        
        assert len(result) == 0
    
    @pytest.mark.unit
    def test_sorted_results(self, tmp_path: Path):
        """Test that results are sorted."""
        file_c = tmp_path / "c.txt"
        file_c.write_text("c", encoding="utf-8")
        
        file_a = tmp_path / "a.txt"
        file_a.write_text("a", encoding="utf-8")
        
        file_b = tmp_path / "b.txt"
        file_b.write_text("b", encoding="utf-8")
        
        result = collect_text_files(tmp_path)
        
        assert len(result) == 3
        # Should be sorted alphabetically
        assert result[0].name == "a.txt"
        assert result[1].name == "b.txt"
        assert result[2].name == "c.txt"


class TestValidateSchemaPaths:
    """Tests for validate_schema_paths function."""
    
    @pytest.mark.unit
    def test_valid_schema_paths(self):
        """Test validation passes with valid paths."""
        schemas_paths = {
            "TestSchema": {
                "input": "/path/to/input",
                "output": "/path/to/output"
            }
        }
        
        result = validate_schema_paths("TestSchema", schemas_paths)
        
        assert result is True
    
    @pytest.mark.unit
    def test_missing_schema_returns_false(self):
        """Test validation fails for missing schema."""
        schemas_paths = {
            "OtherSchema": {
                "input": "/path/to/input",
                "output": "/path/to/output"
            }
        }
        
        result = validate_schema_paths("MissingSchema", schemas_paths)
        
        assert result is False
    
    @pytest.mark.unit
    def test_missing_input_path_returns_false(self):
        """Test validation fails when input path is missing."""
        schemas_paths = {
            "TestSchema": {
                "output": "/path/to/output"
            }
        }
        
        result = validate_schema_paths("TestSchema", schemas_paths)
        
        assert result is False
    
    @pytest.mark.unit
    def test_missing_output_path_returns_false(self):
        """Test validation fails when output path is missing."""
        schemas_paths = {
            "TestSchema": {
                "input": "/path/to/input"
            }
        }
        
        result = validate_schema_paths("TestSchema", schemas_paths)
        
        assert result is False
    
    @pytest.mark.unit
    def test_empty_paths_returns_false(self):
        """Test validation fails with empty paths."""
        schemas_paths = {
            "TestSchema": {
                "input": "",
                "output": ""
            }
        }
        
        result = validate_schema_paths("TestSchema", schemas_paths)
        
        assert result is False
