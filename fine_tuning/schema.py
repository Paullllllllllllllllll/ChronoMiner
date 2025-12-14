from __future__ import annotations

from typing import Any, Dict

from modules.core.schema_manager import SchemaManager


def load_schema(schema_name: str) -> Dict[str, Any]:
    manager = SchemaManager()
    manager.load_schemas()
    schemas = manager.get_available_schemas()
    if schema_name not in schemas:
        available = ", ".join(sorted(schemas.keys()))
        raise ValueError(f"Unknown schema '{schema_name}'. Available: {available}")
    return schemas[schema_name]
