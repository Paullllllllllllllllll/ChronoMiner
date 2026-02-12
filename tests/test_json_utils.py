import json
import pytest
from pathlib import Path
from modules.core.json_utils import (
    extract_entries_from_json,
    _extract_text_from_api_body,
    _parse_entries_from_text,
    _extract_entries_from_record,
)


@pytest.mark.unit
def test_extract_entries_direct_format(tmp_path):
    json_file = tmp_path / "direct.json"
    json_file.write_text(json.dumps({
        "entries": [
            {"id": 1, "name": "entry1"},
            {"id": 2, "name": "entry2"}
        ]
    }), encoding="utf-8")
    
    entries = extract_entries_from_json(json_file)
    assert len(entries) == 2
    assert entries[0]["id"] == 1
    assert entries[1]["name"] == "entry2"


@pytest.mark.unit
def test_extract_entries_with_no_content_flag(tmp_path):
    json_file = tmp_path / "no_content.json"
    json_file.write_text(json.dumps({
        "contains_no_content_of_requested_type": True,
        "entries": []
    }), encoding="utf-8")
    
    entries = extract_entries_from_json(json_file)
    assert len(entries) == 0


@pytest.mark.unit
def test_extract_entries_batch_responses_string_format(tmp_path):
    json_file = tmp_path / "batch_responses.json"
    json_file.write_text(json.dumps({
        "responses": [
            json.dumps({"entries": [{"id": 1}]}),
            json.dumps({"entries": [{"id": 2}]})
        ]
    }), encoding="utf-8")
    
    entries = extract_entries_from_json(json_file)
    assert len(entries) == 2
    assert entries[0]["id"] == 1
    assert entries[1]["id"] == 2


@pytest.mark.unit
def test_extract_entries_chat_completions_format(tmp_path):
    json_file = tmp_path / "chat_completions.json"
    json_file.write_text(json.dumps({
        "responses": [
            {
                "body": {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps({"entries": [{"id": 1, "text": "test"}]})
                            }
                        }
                    ]
                }
            }
        ]
    }), encoding="utf-8")
    
    entries = extract_entries_from_json(json_file)
    assert len(entries) == 1
    assert entries[0]["id"] == 1


@pytest.mark.unit
def test_extract_entries_responses_api_format(tmp_path):
    json_file = tmp_path / "responses_api.json"
    json_file.write_text(json.dumps({
        "responses": [
            {
                "body": {
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {"text": json.dumps({"entries": [{"id": 1}]})}
                            ]
                        }
                    ]
                }
            }
        ]
    }), encoding="utf-8")
    
    entries = extract_entries_from_json(json_file)
    assert len(entries) == 1
    assert entries[0]["id"] == 1


@pytest.mark.unit
def test_extract_entries_chunk_format(tmp_path):
    json_file = tmp_path / "chunks.json"
    json_file.write_text(json.dumps([
        {"response": {"entries": [{"id": 1}]}},
        {"response": {"entries": [{"id": 2}]}}
    ]), encoding="utf-8")
    
    entries = extract_entries_from_json(json_file)
    assert len(entries) == 2


@pytest.mark.unit
def test_extract_entries_filters_none_values(tmp_path):
    json_file = tmp_path / "with_nones.json"
    json_file.write_text(json.dumps({
        "entries": [
            {"id": 1},
            None,
            {"id": 2},
            None
        ]
    }), encoding="utf-8")
    
    entries = extract_entries_from_json(json_file)
    assert len(entries) == 2
    assert entries[0]["id"] == 1
    assert entries[1]["id"] == 2


@pytest.mark.unit
def test_extract_entries_invalid_json(tmp_path):
    json_file = tmp_path / "invalid.json"
    json_file.write_text("not valid json", encoding="utf-8")
    
    entries = extract_entries_from_json(json_file)
    assert len(entries) == 0


@pytest.mark.unit
def test_extract_entries_raw_response_format(tmp_path):
    json_file = tmp_path / "raw_response.json"
    json_file.write_text(json.dumps({
        "responses": [
            {
                "raw_response": {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps({"entries": [{"id": 99}]})
                            }
                        }
                    ]
                }
            }
        ]
    }), encoding="utf-8")
    
    entries = extract_entries_from_json(json_file)
    assert len(entries) == 1
    assert entries[0]["id"] == 99


@pytest.mark.unit
def test_extract_entries_skips_no_content_responses(tmp_path):
    json_file = tmp_path / "mixed_responses.json"
    json_file.write_text(json.dumps({
        "responses": [
            json.dumps({"entries": [{"id": 1}]}),
            json.dumps({"contains_no_content_of_requested_type": True}),
            json.dumps({"entries": [{"id": 2}]})
        ]
    }), encoding="utf-8")
    
    entries = extract_entries_from_json(json_file)
    assert len(entries) == 2
    assert entries[0]["id"] == 1
    assert entries[1]["id"] == 2


# ---------------------------------------------------------------------------
# Records format tests (from process_text_files.py / _generate_output_files)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_extract_entries_records_chat_completions(tmp_path):
    """Records format where response is a Chat Completions API body."""
    json_file = tmp_path / "records_chat.json"
    json_file.write_text(json.dumps({
        "_metadata": {"schema": "test"},
        "records": [
            {
                "custom_id": "chunk-1",
                "chunk_index": 0,
                "response": {
                    "choices": [
                        {"message": {"content": json.dumps({"entries": [{"id": 1}]})}}
                    ]
                }
            },
            {
                "custom_id": "chunk-2",
                "chunk_index": 1,
                "response": {
                    "choices": [
                        {"message": {"content": json.dumps({"entries": [{"id": 2}, {"id": 3}]})}}
                    ]
                }
            }
        ]
    }), encoding="utf-8")
    entries = extract_entries_from_json(json_file)
    assert len(entries) == 3
    assert [e["id"] for e in entries] == [1, 2, 3]


@pytest.mark.unit
def test_extract_entries_records_output_text(tmp_path):
    """Records format where response contains output_text shorthand."""
    json_file = tmp_path / "records_output_text.json"
    json_file.write_text(json.dumps({
        "records": [
            {
                "custom_id": "chunk-1",
                "response": {
                    "output_text": json.dumps({"entries": [{"id": 10}]})
                }
            }
        ]
    }), encoding="utf-8")
    entries = extract_entries_from_json(json_file)
    assert len(entries) == 1
    assert entries[0]["id"] == 10


@pytest.mark.unit
def test_extract_entries_records_responses_api(tmp_path):
    """Records format where response is a Responses API body."""
    json_file = tmp_path / "records_resp_api.json"
    json_file.write_text(json.dumps({
        "records": [
            {
                "custom_id": "chunk-1",
                "response": {
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {"text": json.dumps({"entries": [{"id": 42}]})}
                            ]
                        }
                    ]
                }
            }
        ]
    }), encoding="utf-8")
    entries = extract_entries_from_json(json_file)
    assert len(entries) == 1
    assert entries[0]["id"] == 42


@pytest.mark.unit
def test_extract_entries_records_string_response(tmp_path):
    """Records format where response is an already-serialised JSON string."""
    json_file = tmp_path / "records_str.json"
    json_file.write_text(json.dumps({
        "records": [
            {
                "custom_id": "chunk-1",
                "response": json.dumps({"entries": [{"id": 7}]})
            }
        ]
    }), encoding="utf-8")
    entries = extract_entries_from_json(json_file)
    assert len(entries) == 1
    assert entries[0]["id"] == 7


@pytest.mark.unit
def test_extract_entries_records_mixed_content_no_content(tmp_path):
    """Records format with a mix of content and no-content chunks."""
    json_file = tmp_path / "records_mixed.json"
    json_file.write_text(json.dumps({
        "records": [
            {
                "custom_id": "chunk-1",
                "response": {
                    "choices": [{"message": {"content": json.dumps({"entries": [{"id": 1}]})}}]
                }
            },
            {
                "custom_id": "chunk-2",
                "response": {
                    "choices": [{"message": {"content": json.dumps({"contains_no_content_of_requested_type": True})}}]
                }
            },
            {
                "custom_id": "chunk-3",
                "response": {
                    "choices": [{"message": {"content": json.dumps({"entries": [{"id": 2}]})}}]
                }
            }
        ]
    }), encoding="utf-8")
    entries = extract_entries_from_json(json_file)
    assert len(entries) == 2
    assert [e["id"] for e in entries] == [1, 2]


@pytest.mark.unit
def test_extract_entries_records_with_none_response(tmp_path):
    """Records format where some records have a None response."""
    json_file = tmp_path / "records_none.json"
    json_file.write_text(json.dumps({
        "records": [
            {"custom_id": "chunk-1", "response": None},
            {
                "custom_id": "chunk-2",
                "response": {"entries": [{"id": 5}]}
            }
        ]
    }), encoding="utf-8")
    entries = extract_entries_from_json(json_file)
    assert len(entries) == 1
    assert entries[0]["id"] == 5


@pytest.mark.unit
def test_extract_entries_records_direct_entries_in_response(tmp_path):
    """Records format where response directly contains entries dict."""
    json_file = tmp_path / "records_direct.json"
    json_file.write_text(json.dumps({
        "records": [
            {"custom_id": "c-1", "response": {"entries": [{"id": 1}, None, {"id": 2}]}}
        ]
    }), encoding="utf-8")
    entries = extract_entries_from_json(json_file)
    assert len(entries) == 2
    assert [e["id"] for e in entries] == [1, 2]


# ---------------------------------------------------------------------------
# Helper function unit tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_extract_text_from_api_body_chat_completions():
    body = {"choices": [{"message": {"content": "hello"}}]}
    assert _extract_text_from_api_body(body) == "hello"


@pytest.mark.unit
def test_extract_text_from_api_body_output_text():
    body = {"output_text": "hello world"}
    assert _extract_text_from_api_body(body) == "hello world"


@pytest.mark.unit
def test_extract_text_from_api_body_responses_api():
    body = {
        "output": [
            {"type": "message", "content": [{"text": "part1"}, {"text": "part2"}]}
        ]
    }
    assert _extract_text_from_api_body(body) == "part1part2"


@pytest.mark.unit
def test_extract_text_from_api_body_non_dict():
    assert _extract_text_from_api_body("string") == ""
    assert _extract_text_from_api_body(None) == ""


@pytest.mark.unit
def test_parse_entries_from_text_valid():
    text = json.dumps({"entries": [{"id": 1}]})
    result = _parse_entries_from_text(text)
    assert result == [{"id": 1}]


@pytest.mark.unit
def test_parse_entries_from_text_no_content():
    text = json.dumps({"contains_no_content_of_requested_type": True})
    assert _parse_entries_from_text(text) is None


@pytest.mark.unit
def test_parse_entries_from_text_invalid_json():
    assert _parse_entries_from_text("not json") is None


@pytest.mark.unit
def test_parse_entries_from_text_empty():
    assert _parse_entries_from_text("") is None


@pytest.mark.unit
def test_extract_entries_from_record_string_response():
    record = {"response": json.dumps({"entries": [{"id": 1}]})}
    assert _extract_entries_from_record(record) == [{"id": 1}]


@pytest.mark.unit
def test_extract_entries_from_record_api_body():
    record = {
        "response": {
            "choices": [{"message": {"content": json.dumps({"entries": [{"id": 2}]})}}]
        }
    }
    assert _extract_entries_from_record(record) == [{"id": 2}]


@pytest.mark.unit
def test_extract_entries_from_record_none_response():
    assert _extract_entries_from_record({"response": None}) == []
    assert _extract_entries_from_record({}) == []
    assert _extract_entries_from_record("not a dict") == []


# ---------------------------------------------------------------------------
# End-to-end integration: converter + records format
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_csv_converter_with_records_format_json(tmp_path):
    """CSVConverter should produce a CSV when given records-format JSON."""
    from modules.core.data_processing import CSVConverter

    json_file = tmp_path / "records.json"
    json_file.write_text(json.dumps({
        "_metadata": {"schema": "BibliographicEntries"},
        "records": [
            {
                "custom_id": "chunk-1",
                "response": {
                    "choices": [{
                        "message": {
                            "content": json.dumps({
                                "entries": [{
                                    "full_title": "Test Book",
                                    "short_title": "TB",
                                    "main_author": "Author",
                                    "edition_info": []
                                }]
                            })
                        }
                    }]
                }
            }
        ]
    }), encoding="utf-8")

    csv_file = tmp_path / "output.csv"
    converter = CSVConverter("BibliographicEntries")
    converter.convert_to_csv(json_file, csv_file)

    assert csv_file.exists()
    content = csv_file.read_text(encoding="utf-8")
    assert "Test Book" in content
    assert "Author" in content


@pytest.mark.unit
def test_txt_converter_with_records_format_json(tmp_path):
    """DocumentConverter should produce a TXT when given records-format JSON."""
    from modules.core.text_processing import DocumentConverter

    json_file = tmp_path / "records.json"
    json_file.write_text(json.dumps({
        "records": [
            {
                "custom_id": "chunk-1",
                "response": {
                    "output_text": json.dumps({
                        "entries": [{
                            "full_title": "Another Book",
                            "short_title": "AB",
                            "edition_info": []
                        }]
                    })
                }
            }
        ]
    }), encoding="utf-8")

    txt_file = tmp_path / "output.txt"
    converter = DocumentConverter("BibliographicEntries")
    converter.convert_to_txt(json_file, txt_file)

    assert txt_file.exists()
    content = txt_file.read_text(encoding="utf-8")
    assert "Another Book" in content
