from __future__ import annotations

import pytest

from modules.core.path_utils import (
    HASH_LENGTH,
    MAX_SAFE_NAME_LENGTH,
    create_safe_directory_name,
    create_safe_log_filename,
)


@pytest.mark.unit
def test_create_safe_directory_name_is_bounded_and_stable():
    name = "x" * 500
    suffix = "_working_files"
    safe = create_safe_directory_name(name, suffix=suffix)

    assert safe.endswith(suffix)
    assert len(safe) <= MAX_SAFE_NAME_LENGTH

    dash = safe.rfind("-")
    assert dash != -1
    assert safe[dash + 1 : dash + 1 + HASH_LENGTH].isalnum()


@pytest.mark.unit
def test_create_safe_log_filename_suffix_and_length():
    name = "y" * 500
    safe = create_safe_log_filename(name, "transcription")

    assert safe.endswith("_transcription_log.json")
    assert len(safe) <= MAX_SAFE_NAME_LENGTH
