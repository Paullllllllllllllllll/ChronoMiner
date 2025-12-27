import json
import pytest
from pathlib import Path
from modules.core.json_utils import extract_entries_from_json


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
