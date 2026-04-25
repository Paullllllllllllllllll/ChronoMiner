import json
import pytest
from unittest.mock import Mock, MagicMock
from modules.llm.openai_sdk_utils import sdk_to_dict, list_all_batches, coerce_file_id


@pytest.mark.unit
def test_sdk_to_dict_with_plain_dict():
    obj = {"key": "value", "number": 42}
    result = sdk_to_dict(obj)
    assert result == obj


@pytest.mark.unit
def test_sdk_to_dict_with_model_dump():
    mock_obj = Mock()
    mock_obj.model_dump = Mock(return_value={"field": "value"})
    
    result = sdk_to_dict(mock_obj)
    assert result == {"field": "value"}
    mock_obj.model_dump.assert_called_once()


@pytest.mark.unit
def test_sdk_to_dict_with_to_dict():
    mock_obj = Mock()
    mock_obj.to_dict = Mock(return_value={"field": "value"})
    delattr(mock_obj, "model_dump")
    
    result = sdk_to_dict(mock_obj)
    assert result == {"field": "value"}
    mock_obj.to_dict.assert_called_once()


@pytest.mark.unit
def test_sdk_to_dict_with_json_method():
    mock_obj = Mock()
    mock_obj.json = Mock(return_value='{"field": "value"}')
    delattr(mock_obj, "model_dump")
    delattr(mock_obj, "to_dict")
    
    result = sdk_to_dict(mock_obj)
    assert result == {"field": "value"}


@pytest.mark.unit
def test_sdk_to_dict_with_attributes():
    class SimpleObject:
        def __init__(self):
            self.field1 = "value1"
            self.field2 = 42
            self._private = "hidden"
    
    obj = SimpleObject()
    result = sdk_to_dict(obj)
    
    assert "field1" in result
    assert "field2" in result
    assert "_private" not in result


@pytest.mark.unit
def test_list_all_batches_single_page():
    mock_client = Mock()
    mock_batch = Mock()
    mock_batch.model_dump = Mock(return_value={"id": "batch_1", "status": "completed"})
    
    mock_page = Mock()
    mock_page.data = [mock_batch]
    mock_page.has_more = False
    mock_page.last_id = None
    
    mock_client.batches.list = Mock(return_value=mock_page)
    
    result = list_all_batches(mock_client, limit=100)
    
    assert len(result) == 1
    assert result[0]["id"] == "batch_1"
    mock_client.batches.list.assert_called_once()


@pytest.mark.unit
def test_list_all_batches_multiple_pages():
    mock_client = Mock()
    
    mock_batch1 = Mock()
    mock_batch1.model_dump = Mock(return_value={"id": "batch_1"})
    
    mock_batch2 = Mock()
    mock_batch2.model_dump = Mock(return_value={"id": "batch_2"})
    
    mock_page1 = Mock()
    mock_page1.data = [mock_batch1]
    mock_page1.has_more = True
    mock_page1.last_id = "batch_1"
    
    mock_page2 = Mock()
    mock_page2.data = [mock_batch2]
    mock_page2.has_more = False
    mock_page2.last_id = None
    
    mock_client.batches.list = Mock(side_effect=[mock_page1, mock_page2])
    
    result = list_all_batches(mock_client, limit=100)
    
    assert len(result) == 2
    assert result[0]["id"] == "batch_1"
    assert result[1]["id"] == "batch_2"
    assert mock_client.batches.list.call_count == 2


@pytest.mark.unit
def test_coerce_file_id_with_string():
    result = coerce_file_id("file-123")
    assert result == "file-123"


@pytest.mark.unit
def test_coerce_file_id_with_empty_string():
    result = coerce_file_id("")
    assert result is None


@pytest.mark.unit
def test_coerce_file_id_with_dict_id():
    result = coerce_file_id({"id": "file-123", "other": "data"})
    assert result == "file-123"


@pytest.mark.unit
def test_coerce_file_id_with_dict_file_id():
    result = coerce_file_id({"file_id": "file-456"})
    assert result == "file-456"


@pytest.mark.unit
def test_coerce_file_id_with_list_of_strings():
    result = coerce_file_id(["file-789", "file-012"])
    assert result == "file-789"


@pytest.mark.unit
def test_coerce_file_id_with_list_of_dicts():
    result = coerce_file_id([{"id": "file-111"}, {"id": "file-222"}])
    assert result == "file-111"


@pytest.mark.unit
def test_coerce_file_id_with_empty_list():
    result = coerce_file_id([])
    assert result is None


@pytest.mark.unit
def test_coerce_file_id_with_none():
    result = coerce_file_id(None)
    assert result is None


@pytest.mark.unit
def test_coerce_file_id_with_invalid_dict():
    result = coerce_file_id({"no_id": "here"})
    assert result is None


@pytest.mark.unit
def test_list_all_batches_empty_result():
    mock_client = Mock()
    
    mock_page = MagicMock()
    mock_page.data = []
    mock_page.has_more = False
    mock_page.last_id = None
    mock_page.__iter__ = Mock(return_value=iter([]))
    
    mock_client.batches.list = Mock(return_value=mock_page)
    
    result = list_all_batches(mock_client)
    
    assert len(result) == 0
