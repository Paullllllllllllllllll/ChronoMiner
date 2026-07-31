import json

import pytest

from modules.batch.diagnostics import extract_custom_id_mapping


@pytest.mark.unit
def test_extract_custom_id_mapping_batch_request_format(tmp_path):
    temp_file = tmp_path / "batch_requests.jsonl"

    lines = [
        json.dumps(
            {
                "batch_request": {
                    "custom_id": "req_1",
                    "image_info": {"order_index": 0, "filename": "image1.jpg"},
                }
            }
        ),
        json.dumps(
            {
                "batch_request": {
                    "custom_id": "req_2",
                    "image_info": {"order_index": 1, "filename": "image2.jpg"},
                }
            }
        ),
    ]
    temp_file.write_text("\n".join(lines), encoding="utf-8")

    custom_id_map, order_map = extract_custom_id_mapping(temp_file)

    assert len(custom_id_map) == 2
    assert "req_1" in custom_id_map
    assert "req_2" in custom_id_map
    assert custom_id_map["req_1"]["filename"] == "image1.jpg"
    assert order_map["req_1"] == 0
    assert order_map["req_2"] == 1


@pytest.mark.unit
def test_extract_custom_id_mapping_image_metadata_format(tmp_path):
    temp_file = tmp_path / "image_metadata.jsonl"

    lines = [
        json.dumps(
            {
                "image_metadata": {
                    "custom_id": "img_1",
                    "order_index": 0,
                    "path": "/path/to/image1.jpg",
                }
            }
        ),
        json.dumps(
            {
                "image_metadata": {
                    "custom_id": "img_2",
                    "order_index": 1,
                    "path": "/path/to/image2.jpg",
                }
            }
        ),
    ]
    temp_file.write_text("\n".join(lines), encoding="utf-8")

    custom_id_map, order_map = extract_custom_id_mapping(temp_file)

    assert len(custom_id_map) == 2
    assert "img_1" in custom_id_map
    assert "img_2" in custom_id_map
    assert order_map["img_1"] == 0
    assert order_map["img_2"] == 1


@pytest.mark.unit
def test_extract_custom_id_mapping_mixed_valid_invalid_lines(tmp_path):
    temp_file = tmp_path / "mixed.jsonl"

    lines = [
        json.dumps(
            {
                "batch_request": {
                    "custom_id": "valid_1",
                    "image_info": {"order_index": 0},
                }
            }
        ),
        "invalid json line",
        json.dumps(
            {
                "batch_request": {
                    "custom_id": "valid_2",
                    "image_info": {"order_index": 1},
                }
            }
        ),
        json.dumps({"other_format": "ignored"}),
    ]
    temp_file.write_text("\n".join(lines), encoding="utf-8")

    custom_id_map, order_map = extract_custom_id_mapping(temp_file)

    assert len(custom_id_map) == 2
    assert "valid_1" in custom_id_map
    assert "valid_2" in custom_id_map


@pytest.mark.unit
def test_extract_custom_id_mapping_no_custom_id(tmp_path):
    temp_file = tmp_path / "no_id.jsonl"

    lines = [
        json.dumps({"batch_request": {"image_info": {"filename": "image.jpg"}}}),
        json.dumps({"image_metadata": {"path": "/path/to/image.jpg"}}),
    ]
    temp_file.write_text("\n".join(lines), encoding="utf-8")

    custom_id_map, order_map = extract_custom_id_mapping(temp_file)

    assert len(custom_id_map) == 0
    assert len(order_map) == 0


@pytest.mark.unit
def test_extract_custom_id_mapping_empty_file(tmp_path):
    temp_file = tmp_path / "empty.jsonl"
    temp_file.write_text("", encoding="utf-8")

    custom_id_map, order_map = extract_custom_id_mapping(temp_file)

    assert len(custom_id_map) == 0
    assert len(order_map) == 0


@pytest.mark.unit
def test_extract_custom_id_mapping_without_order_index(tmp_path):
    temp_file = tmp_path / "no_order.jsonl"

    lines = [
        json.dumps(
            {
                "batch_request": {
                    "custom_id": "req_1",
                    "image_info": {"filename": "image1.jpg"},
                }
            }
        )
    ]
    temp_file.write_text("\n".join(lines), encoding="utf-8")

    custom_id_map, order_map = extract_custom_id_mapping(temp_file)

    assert len(custom_id_map) == 1
    assert "req_1" in custom_id_map
    assert len(order_map) == 0
