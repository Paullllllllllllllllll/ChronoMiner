from unittest.mock import Mock

import pytest

from modules.llm.openai_sdk_utils import sdk_to_dict
from modules.llm.openai_utils import _normalize_structured_schema


class _Caps:
    supports_structured_outputs = True


@pytest.mark.unit
def test_normalize_structured_schema_default_name_is_extraction():
    """ChronoMiner extracts; the fallback name must not say 'Transcription'."""
    result = _normalize_structured_schema({"type": "object"}, _Caps())
    assert result == {
        "name": "ExtractionSchema",
        "schema": {"type": "object"},
        "strict": True,
    }


@pytest.mark.unit
def test_normalize_structured_schema_wrapped_default_name():
    result = _normalize_structured_schema({"schema": {"type": "object"}}, _Caps())
    assert result is not None
    assert result["name"] == "ExtractionSchema"


@pytest.mark.unit
def test_normalize_structured_schema_keeps_explicit_name():
    result = _normalize_structured_schema(
        {"name": "BibliographicEntries", "schema": {"type": "object"}}, _Caps()
    )
    assert result is not None
    assert result["name"] == "BibliographicEntries"


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
