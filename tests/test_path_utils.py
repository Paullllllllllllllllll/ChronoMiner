from __future__ import annotations

from pathlib import Path
import pytest

from modules.core.path_utils import (
    HASH_LENGTH,
    MAX_SAFE_NAME_LENGTH,
    create_safe_directory_name,
    create_safe_log_filename,
    ensure_path_safe,
)


class TestCreateSafeDirectoryName:
    """Tests for create_safe_directory_name function."""
    
    @pytest.mark.unit
    def test_bounded_and_stable(self):
        """Test that long names are bounded and hash is stable."""
        name = "x" * 500
        suffix = "_working_files"
        safe = create_safe_directory_name(name, suffix=suffix)

        assert safe.endswith(suffix)
        assert len(safe) <= MAX_SAFE_NAME_LENGTH

        dash = safe.rfind("-")
        assert dash != -1
        assert safe[dash + 1 : dash + 1 + HASH_LENGTH].isalnum()
    
    @pytest.mark.unit
    def test_short_name_preserved(self):
        """Test that short names are preserved."""
        name = "short_name"
        safe = create_safe_directory_name(name, suffix="")
        
        assert name in safe
        assert len(safe) <= MAX_SAFE_NAME_LENGTH
    
    @pytest.mark.unit
    def test_hash_uniqueness(self):
        """Test that different names produce different hashes."""
        safe1 = create_safe_directory_name("name_one")
        safe2 = create_safe_directory_name("name_two")
        
        assert safe1 != safe2
    
    @pytest.mark.unit
    def test_hash_consistency(self):
        """Test that same name produces same hash."""
        name = "consistent_name"
        safe1 = create_safe_directory_name(name)
        safe2 = create_safe_directory_name(name)
        
        assert safe1 == safe2
    
    @pytest.mark.unit
    def test_truncation_at_word_boundary(self):
        """Test that truncation removes trailing punctuation."""
        name = "This is a test name with punctuation. " + "x" * 200
        safe = create_safe_directory_name(name)
        
        # Should not end with period or space before the hash
        dash_idx = safe.rfind("-")
        truncated_part = safe[:dash_idx]
        assert not truncated_part.endswith(".")
        assert not truncated_part.endswith(" ")
    
    @pytest.mark.unit
    def test_empty_suffix(self):
        """Test with empty suffix."""
        name = "test_name"
        safe = create_safe_directory_name(name, suffix="")
        
        assert len(safe) <= MAX_SAFE_NAME_LENGTH
        assert "-" in safe  # Hash separator present
    
    @pytest.mark.unit
    def test_special_characters(self):
        """Test name with special characters."""
        name = "Test (Special) [Characters] {Brackets}"
        safe = create_safe_directory_name(name)
        
        assert len(safe) <= MAX_SAFE_NAME_LENGTH


class TestCreateSafeLogFilename:
    """Tests for create_safe_log_filename function."""
    
    @pytest.mark.unit
    def test_suffix_and_length(self):
        """Test that log filename has correct suffix and length."""
        name = "y" * 500
        safe = create_safe_log_filename(name, "transcription")

        assert safe.endswith("_transcription_log.json")
        assert len(safe) <= MAX_SAFE_NAME_LENGTH
    
    @pytest.mark.unit
    def test_different_log_types(self):
        """Test different log types produce different filenames."""
        name = "test_document"
        trans_log = create_safe_log_filename(name, "transcription")
        summary_log = create_safe_log_filename(name, "summary")
        
        assert trans_log != summary_log
        assert trans_log.endswith("_transcription_log.json")
        assert summary_log.endswith("_summary_log.json")
    
    @pytest.mark.unit
    def test_short_name_preserved(self):
        """Test short names are preserved in log filename."""
        name = "short"
        safe = create_safe_log_filename(name, "test")
        
        assert name in safe
        assert safe.endswith("_test_log.json")
    
    @pytest.mark.unit
    def test_hash_in_filename(self):
        """Test that hash is included in filename."""
        name = "document_name"
        safe = create_safe_log_filename(name, "log")
        
        dash_idx = safe.rfind("-")
        assert dash_idx != -1
        # Hash should be between dash and underscore
        hash_part = safe[dash_idx + 1:].split("_")[0]
        assert len(hash_part) == HASH_LENGTH


class TestEnsurePathSafe:
    """Tests for ensure_path_safe function."""
    
    @pytest.mark.unit
    def test_returns_path_object(self, tmp_path):
        """Test that function returns a Path object."""
        test_path = tmp_path / "test_file.txt"
        result = ensure_path_safe(test_path)
        
        assert isinstance(result, Path)
    
    @pytest.mark.unit
    def test_resolves_relative_path(self, tmp_path):
        """Test that relative paths are resolved."""
        test_path = tmp_path / "subdir" / ".." / "test.txt"
        result = ensure_path_safe(test_path)
        
        assert result.is_absolute()
    
    @pytest.mark.unit
    def test_existing_path(self, tmp_path):
        """Test with existing path."""
        test_file = tmp_path / "existing.txt"
        test_file.write_text("content", encoding="utf-8")
        
        result = ensure_path_safe(test_file)
        assert result.exists()
    
    @pytest.mark.unit
    def test_nonexistent_path(self, tmp_path):
        """Test with non-existent path."""
        test_path = tmp_path / "nonexistent.txt"
        result = ensure_path_safe(test_path)
        
        # Should still return a valid path
        assert isinstance(result, Path)
