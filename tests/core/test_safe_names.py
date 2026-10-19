"""Safe directory and file names in chronominer.discovery.

Names are shortened and hashed so that paths stay under the Windows path limit.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chronominer.discovery import (
    HASH_LENGTH,
    MAX_SAFE_NAME_LENGTH,
    WINDOWS_MAX_PATH,
    _compute_name_hash,
    _truncate_name,
    create_safe_directory_name,
    create_safe_filename,
)


class TestCreateSafeDirectoryName:
    """Tests for create_safe_directory_name function."""

    @pytest.mark.unit
    def test_short_name_unchanged_with_hash(self) -> None:
        """Test that short names get hash appended."""
        result = create_safe_directory_name("short_name")
        assert result.startswith("short_name-")
        assert len(result) == len("short_name") + 1 + HASH_LENGTH  # name + dash + hash

    @pytest.mark.unit
    def test_long_name_truncated(self) -> None:
        """Test that long names are truncated."""
        long_name = "A" * 200
        result = create_safe_directory_name(long_name)
        assert len(result) <= MAX_SAFE_NAME_LENGTH
        assert "-" in result  # Should have hash separator

    @pytest.mark.unit
    def test_suffix_included(self) -> None:
        """Test that suffix is appended correctly."""
        result = create_safe_directory_name("document", suffix="_working_files")
        assert result.endswith("_working_files")

    @pytest.mark.unit
    def test_long_name_with_suffix(self) -> None:
        """Test long name with suffix stays within limits."""
        long_name = "Very long document name that exceeds normal limits" * 3
        suffix = "_working_files"
        result = create_safe_directory_name(long_name, suffix=suffix)
        assert len(result) <= MAX_SAFE_NAME_LENGTH
        assert result.endswith(suffix)

    @pytest.mark.unit
    def test_hash_provides_uniqueness(self) -> None:
        """Test that different names produce different hashes."""
        result1 = create_safe_directory_name("document_a")
        result2 = create_safe_directory_name("document_b")
        # Extract hashes (after last dash, before suffix)
        hash1 = result1.split("-")[-1]
        hash2 = result2.split("-")[-1]
        assert hash1 != hash2

    @pytest.mark.unit
    def test_same_name_produces_same_hash(self) -> None:
        """Test that same name always produces same hash."""
        result1 = create_safe_directory_name("consistent_name")
        result2 = create_safe_directory_name("consistent_name")
        assert result1 == result2

    @pytest.mark.unit
    def test_trailing_punctuation_stripped(self) -> None:
        """Test that trailing punctuation is stripped from truncated names."""
        # Create a name that will be truncated at a punctuation point
        long_name = "Document name with trailing... " + "x" * 100
        result = create_safe_directory_name(long_name)
        # Should not end with punctuation before the hash
        name_part = result.rsplit("-", 1)[0]
        assert not name_part.endswith((".", "-", "_", " "))


class TestCreateSafeFilename:
    """Tests for create_safe_filename function."""

    @pytest.mark.unit
    @pytest.mark.parametrize("ext", [".txt", ".jsonl"])
    def test_long_name_truncated_with_hash_and_extension(self, ext: str) -> None:
        """A long name is shortened with a hash; the extension is preserved."""
        result = create_safe_filename("A" * 200, ext)
        assert len(result) <= MAX_SAFE_NAME_LENGTH
        assert result.endswith(ext)
        assert "-" in result

    @pytest.mark.unit
    def test_parent_path_considered(self, temp_dir: Path) -> None:
        """Test that parent path length is considered for truncation."""
        parent_len = 190
        budget = parent_len - len(str(temp_dir))
        if budget < 2:
            pytest.skip("temp root leaves no room below the directory limit")
        segments: list[str] = []
        while budget > 60:
            segments.append("d" * 50)
            budget -= 51
        segments.append("d" * (budget - 1))
        deep_dir = temp_dir.joinpath(*segments)
        deep_dir.mkdir(parents=True, exist_ok=True)
        assert len(str(deep_dir)) == parent_len

        original = "document_" * 20
        result = create_safe_filename(original, ".txt", parent_path=deep_dir)

        assert len(str(deep_dir / result)) < WINDOWS_MAX_PATH
        assert len(result) < MAX_SAFE_NAME_LENGTH
        assert result.endswith(f"-{_compute_name_hash(original)}.txt")

    @pytest.mark.unit
    @pytest.mark.parametrize("ext", [".txt", ".md"])
    def test_short_name_keeps_its_extension(self, ext: str) -> None:
        """A short name keeps any extension without a hash."""
        assert create_safe_filename("test_file", ext) == f"test_file{ext}"


class TestTruncateName:
    """Truncation caps the length and drops trailing punctuation."""

    @pytest.mark.unit
    @pytest.mark.parametrize(("name", "limit"), [("short", 10), ("12345", 5)])
    def test_name_within_the_limit_is_unchanged(self, name: str, limit: int) -> None:
        assert _truncate_name(name, limit) == name

    @pytest.mark.unit
    def test_strips_trailing_punctuation(self) -> None:
        # Cutting at 12 characters leaves "hello world." with a trailing dot.
        result = _truncate_name("hello world. more text", 12)
        assert not result.endswith(".")
