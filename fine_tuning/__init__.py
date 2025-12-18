"""
ChronoMiner Fine-Tuning Module.

Provides tools for creating fine-tuning datasets from structured data extraction:
- create-editable: Generate human-editable annotation files with source identification
- import-annotations: Import corrected annotations to JSONL
- build-sft: Build text-based SFT datasets with train/val split

Usage:
    python -m fine_tuning.cli create-editable --schema historical_recipes --chunks input.txt
    python -m fine_tuning.cli import-annotations --schema historical_recipes --editable annotations_editable.txt
    python -m fine_tuning.cli build-sft --schema historical_recipes --annotations annotations.jsonl --dataset-id my_dataset
"""

from fine_tuning.annotation_txt import ChunkAnnotation, read_annotations_txt, write_annotations_txt
from fine_tuning.annotations_jsonl import build_annotation_records, write_annotations_jsonl
from fine_tuning.chunk_input import ChunkInput, read_chunk_inputs_txt
from fine_tuning.jsonl_io import iter_jsonl, read_jsonl, write_jsonl
from fine_tuning.paths import annotations_root, artifacts_root, datasets_root, editable_root
from fine_tuning.sft_dataset import build_sft_dataset, build_sft_examples, train_val_split
from fine_tuning.validation import validate_top_level_output

__all__ = [
    "ChunkAnnotation",
    "ChunkInput",
    "read_annotations_txt",
    "write_annotations_txt",
    "read_chunk_inputs_txt",
    "build_annotation_records",
    "write_annotations_jsonl",
    "iter_jsonl",
    "read_jsonl",
    "write_jsonl",
    "annotations_root",
    "artifacts_root",
    "datasets_root",
    "editable_root",
    "build_sft_dataset",
    "build_sft_examples",
    "train_val_split",
    "validate_top_level_output",
]
