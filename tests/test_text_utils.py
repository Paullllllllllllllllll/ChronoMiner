import pytest
from pathlib import Path
from modules.core.text_utils import (
    TextProcessor, TokenBasedChunking, ChunkHandler, load_line_ranges
)


@pytest.mark.unit
def test_text_processor_detect_encoding(tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("Hello World", encoding="utf-8")
    
    encoding = TextProcessor.detect_encoding(test_file)
    assert encoding is not None
    assert isinstance(encoding, str)


@pytest.mark.unit
def test_text_processor_normalize_text():
    text = "  hello world  \n\t"
    normalized = TextProcessor.normalize_text(text)
    assert normalized == "hello world"


@pytest.mark.unit
def test_text_processor_estimate_tokens():
    text = "This is a test sentence."
    token_count = TextProcessor.estimate_tokens(text)
    assert token_count > 0
    assert isinstance(token_count, int)


@pytest.mark.unit
def test_text_processor_estimate_tokens_empty_string():
    token_count = TextProcessor.estimate_tokens("")
    assert token_count == 0


@pytest.mark.unit
def test_token_based_chunking_single_chunk():
    processor = TextProcessor()
    chunking = TokenBasedChunking(
        tokens_per_chunk=1000,
        model_name="gpt-4o",
        text_processor=processor
    )
    
    lines = ["Line 1\n", "Line 2\n", "Line 3\n"]
    ranges = chunking.get_line_ranges(lines)
    
    assert len(ranges) == 1
    assert ranges[0] == (1, 3)


@pytest.mark.unit
def test_token_based_chunking_multiple_chunks():
    processor = TextProcessor()
    chunking = TokenBasedChunking(
        tokens_per_chunk=5,
        model_name="gpt-4o",
        text_processor=processor
    )
    
    lines = [
        "This is a long line with many tokens\n",
        "Another long line with many tokens\n",
        "Third line\n"
    ]
    ranges = chunking.get_line_ranges(lines)
    
    assert len(ranges) > 0
    assert all(isinstance(r, tuple) and len(r) == 2 for r in ranges)


@pytest.mark.unit
def test_chunk_handler_split_text_into_chunks():
    processor = TextProcessor()
    handler = ChunkHandler(
        model_name="gpt-4o",
        default_tokens_per_chunk=100,
        text_processor=processor
    )
    
    lines = ["Line 1\n", "Line 2\n", "Line 3\n", "Line 4\n"]
    ranges = [(1, 2), (3, 4)]
    
    chunks = handler.split_text_into_chunks(lines, ranges)
    
    assert len(chunks) == 2
    assert chunks[0] == "Line 1\nLine 2\n"
    assert chunks[1] == "Line 3\nLine 4\n"


@pytest.mark.unit
def test_chunk_handler_split_text_single_line_chunks():
    processor = TextProcessor()
    handler = ChunkHandler(
        model_name="gpt-4o",
        default_tokens_per_chunk=100,
        text_processor=processor
    )
    
    lines = ["Line 1\n", "Line 2\n"]
    ranges = [(1, 1), (2, 2)]
    
    chunks = handler.split_text_into_chunks(lines, ranges)
    
    assert len(chunks) == 2
    assert chunks[0] == "Line 1\n"
    assert chunks[1] == "Line 2\n"


@pytest.mark.unit
def test_load_line_ranges_basic(tmp_path):
    ranges_file = tmp_path / "line_ranges.txt"
    ranges_file.write_text("1, 10\n11, 20\n21, 30\n", encoding="utf-8")
    
    ranges = load_line_ranges(ranges_file)
    
    assert len(ranges) == 3
    assert ranges[0] == (1, 10)
    assert ranges[1] == (11, 20)
    assert ranges[2] == (21, 30)


@pytest.mark.unit
def test_load_line_ranges_with_parentheses(tmp_path):
    ranges_file = tmp_path / "line_ranges.txt"
    ranges_file.write_text("(1, 10)\n(11, 20)\n(21, 30)\n", encoding="utf-8")
    
    ranges = load_line_ranges(ranges_file)
    
    assert len(ranges) == 3
    assert ranges[0] == (1, 10)
    assert ranges[1] == (11, 20)
    assert ranges[2] == (21, 30)


@pytest.mark.unit
def test_load_line_ranges_with_whitespace(tmp_path):
    ranges_file = tmp_path / "line_ranges.txt"
    ranges_file.write_text("  1 ,  10  \n  11 , 20  \n", encoding="utf-8")
    
    ranges = load_line_ranges(ranges_file)
    
    assert len(ranges) == 2
    assert ranges[0] == (1, 10)
    assert ranges[1] == (11, 20)


@pytest.mark.unit
def test_load_line_ranges_empty_file(tmp_path):
    ranges_file = tmp_path / "empty_ranges.txt"
    ranges_file.write_text("", encoding="utf-8")
    
    ranges = load_line_ranges(ranges_file)
    
    assert len(ranges) == 0


@pytest.mark.unit
def test_load_line_ranges_with_empty_lines(tmp_path):
    ranges_file = tmp_path / "ranges_with_empty.txt"
    ranges_file.write_text("1, 10\n\n11, 20\n\n", encoding="utf-8")
    
    ranges = load_line_ranges(ranges_file)
    
    assert len(ranges) == 2
    assert ranges[0] == (1, 10)
    assert ranges[1] == (11, 20)


@pytest.mark.unit
def test_load_line_ranges_invalid_format_skipped(tmp_path):
    ranges_file = tmp_path / "invalid_ranges.txt"
    ranges_file.write_text("1, 10\ninvalid line\n20, 30\n", encoding="utf-8")
    
    ranges = load_line_ranges(ranges_file)
    
    assert len(ranges) == 2
    assert ranges[0] == (1, 10)
    assert ranges[1] == (20, 30)


@pytest.mark.unit
def test_chunk_handler_get_line_ranges():
    processor = TextProcessor()
    handler = ChunkHandler(
        model_name="gpt-4o",
        default_tokens_per_chunk=100,
        text_processor=processor
    )
    
    strategy = TokenBasedChunking(100, "gpt-4o", processor)
    lines = ["Line 1\n", "Line 2\n", "Line 3\n"]
    
    ranges = handler.get_line_ranges(strategy, lines)
    
    assert len(ranges) > 0
    assert all(isinstance(r, tuple) for r in ranges)
