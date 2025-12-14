from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from fine_tuning.jsonl_io import iter_jsonl, write_jsonl


DEFAULT_SYSTEM_PROMPT = (
    "You are a structured data extraction expert. "
    "Extract structured data from the user's text. "
    "Return only valid JSON. "
    "Use null for missing values."
)


def _canonical_json(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def build_sft_examples(
    annotation_records: Iterable[Dict[str, Any]],
    *,
    system_prompt: str,
) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    for rec in annotation_records:
        input_text = rec.get("input_text")
        output_obj = rec.get("output")
        if not isinstance(input_text, str):
            raise ValueError("Annotation record missing 'input_text' string")
        if not isinstance(output_obj, dict):
            raise ValueError("Annotation record missing 'output' JSON object")

        examples.append(
            {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_text},
                    {"role": "assistant", "content": _canonical_json(output_obj)},
                ]
            }
        )
    return examples


def train_val_split(
    examples: Sequence[Dict[str, Any]],
    *,
    val_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if val_ratio < 0 or val_ratio >= 1:
        raise ValueError("val_ratio must be in [0, 1)")

    rng = random.Random(seed)
    indices = list(range(len(examples)))
    rng.shuffle(indices)

    val_count = int(len(examples) * val_ratio)
    val_idx = set(indices[:val_count])

    train: List[Dict[str, Any]] = []
    val: List[Dict[str, Any]] = []
    for i, ex in enumerate(examples):
        if i in val_idx:
            val.append(ex)
        else:
            train.append(ex)

    return train, val


def build_sft_dataset(
    *,
    annotations_paths: List[Path],
    out_dir: Path,
    system_prompt: Optional[str] = None,
    val_ratio: float = 0.1,
    seed: int = 0,
) -> Tuple[Path, Optional[Path]]:
    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    records: List[Dict[str, Any]] = []
    for p in annotations_paths:
        for rec in iter_jsonl(p):
            records.append(rec)

    examples = build_sft_examples(records, system_prompt=system_prompt)
    train, val = train_val_split(examples, val_ratio=val_ratio, seed=seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    write_jsonl(train_path, train)

    val_path: Optional[Path]
    if val:
        val_path = out_dir / "val.jsonl"
        write_jsonl(val_path, val)
    else:
        val_path = None

    return train_path, val_path
