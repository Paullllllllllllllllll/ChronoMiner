"""
JSONL utilities for chunk-level extraction evaluation.

This module provides functions for parsing, exporting, and importing chunk-level
extraction data from temporary JSONL files produced during extraction runs.
This enables evaluation at the chunk level, avoiding formatting penalties from
post-processing and enabling accurate per-chunk error attribution.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ChunkExtraction:
    """Represents a single chunk's extraction result."""
    
    chunk_index: int
    custom_id: str
    extraction_data: Dict[str, Any] = field(default_factory=dict)
    chunk_text: Optional[str] = None
    chunk_range: Optional[Tuple[int, int]] = None
    
    def has_entries(self) -> bool:
        """Check if this chunk has extracted entries."""
        entries = self.extraction_data.get("entries", [])
        return bool(entries)
    
    def get_entries(self) -> List[Dict[str, Any]]:
        """Get the entries list from extraction data."""
        return self.extraction_data.get("entries", [])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "chunk_index": self.chunk_index,
            "custom_id": self.custom_id,
            "extraction_data": self.extraction_data,
            "chunk_text": self.chunk_text,
            "chunk_range": self.chunk_range,
        }


@dataclass
class DocumentExtractions:
    """Container for all chunk extractions from a single document."""
    
    source_name: str
    chunks: List[ChunkExtraction] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def chunk_count(self) -> int:
        """Return number of chunks."""
        return len(self.chunks)
    
    def get_chunk_by_index(self, index: int) -> Optional[ChunkExtraction]:
        """Get chunk by its index."""
        for chunk in self.chunks:
            if chunk.chunk_index == index:
                return chunk
        return None
    
    def get_chunk_by_custom_id(self, custom_id: str) -> Optional[ChunkExtraction]:
        """Get chunk by custom_id."""
        for chunk in self.chunks:
            if chunk.custom_id == custom_id:
                return chunk
        return None
    
    def get_all_entries(self) -> List[Dict[str, Any]]:
        """Get all entries from all chunks, preserving order."""
        all_entries = []
        for chunk in sorted(self.chunks, key=lambda c: c.chunk_index):
            all_entries.extend(chunk.get_entries())
        return all_entries
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source_name": self.source_name,
            "chunks": [c.to_dict() for c in self.chunks],
            "metadata": self.metadata,
        }


def parse_extraction_jsonl(jsonl_path: Path) -> DocumentExtractions:
    """
    Parse a temporary JSONL file containing chunk extraction results.
    
    Handles both sync processing format and batch API response format.
    
    Args:
        jsonl_path: Path to the JSONL file
        
    Returns:
        DocumentExtractions object with all chunks
    """
    doc = DocumentExtractions(source_name=jsonl_path.stem.replace("_temp", ""))
    
    if not jsonl_path.exists():
        return doc
    
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            # Skip batch tracking records
            if "batch_tracking" in record:
                continue
            
            # Extract chunk data
            chunk = _parse_chunk_record(record, line_num)
            if chunk:
                doc.chunks.append(chunk)
    
    # Sort chunks by index
    doc.chunks.sort(key=lambda c: c.chunk_index)
    
    return doc


def _parse_chunk_record(record: Dict[str, Any], line_num: int) -> Optional[ChunkExtraction]:
    """
    Parse a single JSONL record into a ChunkExtraction.
    
    Handles multiple formats:
    - Sync processing: {"custom_id": "...", "response": {"body": {...}}}
    - Batch API: {"custom_id": "...", "response": {"body": {"choices": [...]}}}
    - Ground truth: {"chunk_index": N, "custom_id": "...", "extraction_data": {...}}
    """
    custom_id = record.get("custom_id", "")
    
    # Try to extract chunk index from custom_id (format: "{stem}-chunk-{idx}")
    chunk_index = _extract_chunk_index(custom_id, line_num)
    
    # Get extraction data from various locations
    extraction_data = {}
    chunk_text = None
    chunk_range = record.get("chunk_range")
    
    # Ground truth format
    if "extraction_data" in record:
        extraction_data = record.get("extraction_data", {})
        chunk_index = record.get("chunk_index", chunk_index)
        chunk_text = record.get("chunk_text")
    
    # Sync processing format: response.body contains the result
    elif "response" in record:
        body = record.get("response", {}).get("body", {})
        
        if isinstance(body, dict):
            # Check for output_text (parsed JSON string)
            output_text = body.get("output_text", "")
            if output_text:
                try:
                    extraction_data = json.loads(output_text) if isinstance(output_text, str) else output_text
                except (json.JSONDecodeError, TypeError):
                    extraction_data = {"raw_output": output_text}
            
            # Also check response_data for structured output
            response_data = body.get("response_data", {})
            if response_data and not extraction_data:
                # Try to get from choices
                choices = response_data.get("choices", [])
                if choices:
                    message = choices[0].get("message", {})
                    content = message.get("content", "")
                    if content:
                        try:
                            extraction_data = json.loads(content) if isinstance(content, str) else content
                        except (json.JSONDecodeError, TypeError):
                            extraction_data = {"raw_output": content}
    
    # Direct format (some batch responses)
    elif "choices" in record:
        choices = record.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            content = message.get("content", "")
            if content:
                try:
                    extraction_data = json.loads(content) if isinstance(content, str) else content
                except (json.JSONDecodeError, TypeError):
                    extraction_data = {"raw_output": content}
    
    if not custom_id and chunk_index == 0:
        return None
    
    return ChunkExtraction(
        chunk_index=chunk_index,
        custom_id=custom_id,
        extraction_data=extraction_data,
        chunk_text=chunk_text,
        chunk_range=chunk_range,
    )


def _extract_chunk_index(custom_id: str, fallback: int = 0) -> int:
    """Extract chunk index from custom_id string."""
    if not custom_id:
        return fallback
    
    # Pattern: {stem}-chunk-{idx}
    match = re.search(r'-chunk-(\d+)$', custom_id)
    if match:
        return int(match.group(1))
    
    return fallback


def find_jsonl_file(
    base_path: Path,
    category: str,
    model_name: str,
    source_name: str,
) -> Optional[Path]:
    """
    Find a JSONL file for a specific source/model combination.
    
    Args:
        base_path: Base output directory
        category: Dataset category
        model_name: Model identifier
        source_name: Source file name (without extension)
        
    Returns:
        Path to JSONL file or None if not found
    """
    # Try various naming patterns
    patterns = [
        base_path / category / model_name / f"{source_name}_temp.jsonl",
        base_path / category / model_name / source_name / f"{source_name}_temp.jsonl",
        base_path / category / model_name / f"{source_name}.jsonl",
    ]
    
    for pattern in patterns:
        if pattern.exists():
            return pattern
    
    # Try glob search
    search_dir = base_path / category / model_name
    if search_dir.exists():
        matches = list(search_dir.rglob(f"{source_name}*.jsonl"))
        # Filter out batch tracking files
        matches = [m for m in matches if "_batch_" not in m.name]
        if matches:
            return matches[0]
    
    return None


def load_chunk_extractions(
    output_path: Path,
    category: str,
    model_name: str,
    source_name: str,
) -> Optional[DocumentExtractions]:
    """
    Load chunk extractions for a specific source/model combination.
    
    Args:
        output_path: Base output directory
        category: Dataset category
        model_name: Model identifier
        source_name: Source file name
        
    Returns:
        DocumentExtractions or None if not found
    """
    jsonl_path = find_jsonl_file(output_path, category, model_name, source_name)
    if jsonl_path is None:
        return None
    
    return parse_extraction_jsonl(jsonl_path)


def load_ground_truth_chunks(
    ground_truth_path: Path,
    category: str,
    source_name: str,
) -> Optional[DocumentExtractions]:
    """
    Load ground truth chunk extractions.
    
    Args:
        ground_truth_path: Base ground truth directory
        category: Dataset category
        source_name: Source file name
        
    Returns:
        DocumentExtractions or None if not found
    """
    # Try JSONL format first
    jsonl_path = ground_truth_path / category / f"{source_name}.jsonl"
    if jsonl_path.exists():
        return parse_extraction_jsonl(jsonl_path)
    
    # Fall back to legacy JSON format (single merged file)
    json_path = ground_truth_path / category / f"{source_name}.json"
    if json_path.exists():
        return _load_legacy_json_as_chunks(json_path, source_name)
    
    return None


def _load_legacy_json_as_chunks(json_path: Path, source_name: str) -> DocumentExtractions:
    """
    Load a legacy merged JSON file as a single-chunk DocumentExtractions.
    
    This provides backward compatibility with the old evaluation format.
    """
    doc = DocumentExtractions(source_name=source_name)
    
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Create a single chunk containing all entries
        chunk = ChunkExtraction(
            chunk_index=1,
            custom_id=f"{source_name}-chunk-1",
            extraction_data=data if isinstance(data, dict) else {"entries": data},
        )
        doc.chunks.append(chunk)
    except (json.JSONDecodeError, IOError):
        pass
    
    return doc


def align_chunks(
    hyp_doc: DocumentExtractions,
    gt_doc: DocumentExtractions,
) -> List[Tuple[Optional[ChunkExtraction], Optional[ChunkExtraction]]]:
    """
    Align hypothesis chunks with ground truth chunks.
    
    Uses chunk_index for alignment. Returns pairs of (hyp_chunk, gt_chunk)
    where either may be None if unmatched.
    
    Args:
        hyp_doc: Hypothesis (model output) document
        gt_doc: Ground truth document
        
    Returns:
        List of aligned chunk pairs
    """
    aligned = []
    
    # Get all unique indices
    hyp_indices = {c.chunk_index for c in hyp_doc.chunks}
    gt_indices = {c.chunk_index for c in gt_doc.chunks}
    all_indices = sorted(hyp_indices | gt_indices)
    
    for idx in all_indices:
        hyp_chunk = hyp_doc.get_chunk_by_index(idx)
        gt_chunk = gt_doc.get_chunk_by_index(idx)
        aligned.append((hyp_chunk, gt_chunk))
    
    return aligned


def export_chunks_to_editable_txt(
    doc: DocumentExtractions,
    output_path: Path,
    include_chunk_text: bool = False,
) -> None:
    """
    Export chunk extractions to an editable text file with chunk markers.
    
    Format:
    === chunk 001 ===
    {
      "entries": [...]
    }
    
    === chunk 002 ===
    ...
    
    Args:
        doc: DocumentExtractions to export
        output_path: Path for output text file
        include_chunk_text: Whether to include original chunk text as comments
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open("w", encoding="utf-8") as f:
        f.write(f"# Source: {doc.source_name}\n")
        f.write(f"# Chunks: {doc.chunk_count()}\n")
        f.write("#\n")
        f.write("# Edit the JSON content below each chunk marker.\n")
        f.write("# Do not modify the chunk markers (=== chunk NNN ===).\n")
        f.write("#\n\n")
        
        for chunk in sorted(doc.chunks, key=lambda c: c.chunk_index):
            f.write(f"=== chunk {chunk.chunk_index:03d} ===\n")
            
            if include_chunk_text and chunk.chunk_text:
                f.write("# Original text (for reference):\n")
                for line in chunk.chunk_text.split("\n")[:10]:  # First 10 lines
                    f.write(f"# {line}\n")
                f.write("#\n")
            
            # Pretty-print the extraction data
            json_str = json.dumps(chunk.extraction_data, indent=2, ensure_ascii=False)
            f.write(json_str)
            f.write("\n\n")


def import_chunks_from_editable_txt(
    txt_path: Path,
    original_doc: Optional[DocumentExtractions] = None,
) -> DocumentExtractions:
    """
    Import corrected chunk extractions from an edited text file.
    
    Args:
        txt_path: Path to edited text file
        original_doc: Optional original document for metadata
        
    Returns:
        DocumentExtractions with corrected data
    """
    source_name = txt_path.stem.replace("_editable", "")
    doc = DocumentExtractions(source_name=source_name)
    
    if original_doc:
        doc.metadata = original_doc.metadata.copy()
    
    content = txt_path.read_text(encoding="utf-8")
    
    # Split by chunk markers
    chunk_pattern = re.compile(r'===\s*chunk\s+(\d+)\s*===', re.IGNORECASE)
    
    # Find all chunk markers and their positions
    markers = list(chunk_pattern.finditer(content))
    
    for i, match in enumerate(markers):
        chunk_index = int(match.group(1))
        
        # Get content between this marker and the next (or end)
        start = match.end()
        end = markers[i + 1].start() if i + 1 < len(markers) else len(content)
        chunk_content = content[start:end].strip()
        
        # Remove comment lines
        lines = []
        for line in chunk_content.split("\n"):
            stripped = line.strip()
            if not stripped.startswith("#"):
                lines.append(line)
        chunk_json_str = "\n".join(lines).strip()
        
        # Parse JSON
        extraction_data = {}
        if chunk_json_str:
            try:
                extraction_data = json.loads(chunk_json_str)
            except json.JSONDecodeError as e:
                # Try to recover by finding JSON object boundaries
                extraction_data = {"parse_error": str(e), "raw_content": chunk_json_str}
        
        # Get original chunk text if available
        chunk_text = None
        if original_doc:
            orig_chunk = original_doc.get_chunk_by_index(chunk_index)
            if orig_chunk:
                chunk_text = orig_chunk.chunk_text
        
        chunk = ChunkExtraction(
            chunk_index=chunk_index,
            custom_id=f"{source_name}-chunk-{chunk_index}",
            extraction_data=extraction_data,
            chunk_text=chunk_text,
        )
        doc.chunks.append(chunk)
    
    # Sort by index
    doc.chunks.sort(key=lambda c: c.chunk_index)
    
    return doc


def write_ground_truth_jsonl(
    doc: DocumentExtractions,
    output_path: Path,
) -> None:
    """
    Write corrected extractions to ground truth JSONL format.
    
    Args:
        doc: DocumentExtractions to write
        output_path: Path for output JSONL file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open("w", encoding="utf-8") as f:
        # Write metadata record
        metadata_record = {
            "metadata": {
                "source_name": doc.source_name,
                "chunk_count": doc.chunk_count(),
                "is_ground_truth": True,
            }
        }
        f.write(json.dumps(metadata_record) + "\n")
        
        # Write chunk records
        for chunk in sorted(doc.chunks, key=lambda c: c.chunk_index):
            record = {
                "chunk_index": chunk.chunk_index,
                "custom_id": chunk.custom_id,
                "extraction_data": chunk.extraction_data,
            }
            if chunk.chunk_text:
                record["chunk_text"] = chunk.chunk_text
            if chunk.chunk_range:
                record["chunk_range"] = chunk.chunk_range
            
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def convert_legacy_json_to_jsonl(
    json_path: Path,
    jsonl_path: Path,
    source_name: Optional[str] = None,
) -> DocumentExtractions:
    """
    Convert a legacy merged JSON file to chunk-based JSONL format.
    
    This creates a single-chunk JSONL file from the merged output.
    
    Args:
        json_path: Path to legacy JSON file
        jsonl_path: Path for output JSONL file
        source_name: Optional source name override
        
    Returns:
        DocumentExtractions created from the JSON file
    """
    if source_name is None:
        source_name = json_path.stem
    
    doc = _load_legacy_json_as_chunks(json_path, source_name)
    write_ground_truth_jsonl(doc, jsonl_path)
    
    return doc
