from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fine_tuning.annotation_txt import ChunkAnnotation
from fine_tuning.jsonl_io import write_jsonl


def build_annotation_records(
    *,
    schema_wrapper: Dict[str, Any],
    annotations: List[ChunkAnnotation],
    source_id: Optional[str] = None,
    annotator_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Build annotation records from chunk annotations.
    
    Args:
        schema_wrapper: The schema wrapper containing schema name and version.
        annotations: List of ChunkAnnotation objects.
        source_id: Optional identifier for the source document.
        annotator_id: Optional identifier for the annotator.
        
    Returns:
        List of annotation records ready for JSONL serialization.
    """
    schema_name = schema_wrapper.get("name")
    schema_version = schema_wrapper.get("schema_version")

    records: List[Dict[str, Any]] = []
    for ann in sorted(annotations, key=lambda a: a.chunk_index):
        if ann.output is None:
            raise ValueError(f"Chunk {ann.chunk_index}: missing output JSON")

        record: Dict[str, Any] = {
            "schema_name": schema_name,
            "schema_version": schema_version,
            "chunk_index": ann.chunk_index,
            "input_text": ann.input_text,
            "output": ann.output,
            "annotated_at": datetime.now(timezone.utc).isoformat(),
        }
        
        # Include source file information for traceability
        if ann.source_file is not None:
            record["source_file"] = ann.source_file
        if ann.source_path is not None:
            record["source_path"] = ann.source_path
        
        if source_id is not None:
            record["source_id"] = source_id
        if annotator_id is not None:
            record["annotator_id"] = annotator_id

        records.append(record)

    return records


def write_annotations_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    write_jsonl(path, records)
