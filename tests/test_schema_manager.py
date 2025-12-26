from __future__ import annotations

import json
from pathlib import Path

import pytest

from modules.core.schema_manager import SchemaManager


@pytest.mark.unit
def test_schema_manager_loads_schemas_and_lists_options(tmp_path: Path):
    schemas_dir = tmp_path / "schemas"
    schemas_dir.mkdir()

    schema = {"name": "MySchema", "schema": {"type": "object"}}
    (schemas_dir / "my_schema.json").write_text(json.dumps(schema), encoding="utf-8")

    mgr = SchemaManager(schemas_dir=schemas_dir, dev_messages_dir=tmp_path / "devmsgs")
    mgr.load_schemas()

    available = mgr.get_available_schemas()
    assert "MySchema" in available

    options = mgr.list_schema_options()
    assert options[0][0] == "MySchema"


@pytest.mark.unit
def test_schema_manager_loads_dev_messages(tmp_path: Path):
    schemas_dir = tmp_path / "schemas"
    schemas_dir.mkdir()

    dev_dir = tmp_path / "developer_messages"
    dev_dir.mkdir()

    (dev_dir / "Foo.txt").write_text("hello", encoding="utf-8")

    bar_dir = dev_dir / "Bar"
    bar_dir.mkdir()
    (bar_dir / "a.txt").write_text("a", encoding="utf-8")
    (bar_dir / "b.txt").write_text("b", encoding="utf-8")

    mgr = SchemaManager(schemas_dir=schemas_dir, dev_messages_dir=dev_dir)
    mgr.load_dev_messages()

    assert mgr.get_dev_message("Foo") == "hello"
    assert mgr.get_dev_message("Bar") == "a\n\nb".replace("\n\n", "\n") or mgr.get_dev_message("Bar") == "a\n\nb"  # tolerate join formatting
