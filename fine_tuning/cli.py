from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from fine_tuning.annotation_txt import ChunkAnnotation, read_annotations_txt, write_annotations_txt
from fine_tuning.annotations_jsonl import build_annotation_records, write_annotations_jsonl
from fine_tuning.chunk_input import read_chunk_inputs_txt
from fine_tuning.extractor import extract_chunks_sync
from fine_tuning.paths import annotations_root, datasets_root, editable_root
from fine_tuning.prompt_builder import build_system_prompt
from fine_tuning.schema import load_schema
from fine_tuning.sft_dataset import build_sft_dataset
from fine_tuning.validation import validate_top_level_output


def _default_prompt_path() -> Path:
    return Path("prompts") / "structured_output_prompt.txt"


def _blank_output(schema_wrapper: Dict[str, Any]) -> Dict[str, Any]:
    root = schema_wrapper.get("schema")
    if not isinstance(root, dict):
        return {
            "contains_no_content_of_requested_type": False,
            "entries": [],
        }

    required = root.get("required")
    if not isinstance(required, list) or not required:
        return {
            "contains_no_content_of_requested_type": False,
            "entries": [],
        }

    out: Dict[str, Any] = {}
    for key in required:
        if key == "contains_no_content_of_requested_type":
            out[key] = False
        elif key == "entries":
            out[key] = []
        else:
            out[key] = None
    return out


def cmd_create_editable(args: argparse.Namespace) -> None:
    schema_wrapper = load_schema(args.schema)

    chunk_inputs = read_chunk_inputs_txt(Path(args.chunks))
    chunk_texts = [c.input_text for c in chunk_inputs]

    annotations: List[ChunkAnnotation] = []

    if args.blank:
        for c in chunk_inputs:
            annotations.append(
                ChunkAnnotation(
                    chunk_index=c.chunk_index,
                    input_text=c.input_text,
                    output=_blank_output(schema_wrapper),
                )
            )
    else:
        system_prompt = build_system_prompt(
            schema_name=args.schema,
            schema_obj=schema_wrapper.get("schema", {}),
            prompt_path=Path(args.prompt_path),
            inject_schema=not args.no_inject_schema,
            additional_context=None,
        )

        results = extract_chunks_sync(
            chunk_texts=chunk_texts,
            system_prompt=system_prompt,
            schema_definition=schema_wrapper,
            model_name=args.model,
        )

        if len(results) != len(chunk_inputs):
            raise RuntimeError(
                f"Expected {len(chunk_inputs)} results but got {len(results)}"
            )

        for c, res in zip(chunk_inputs, results):
            out_obj = res.get("output")
            if not isinstance(out_obj, dict):
                out_obj = {"raw_model_output": res.get("raw")}
            annotations.append(
                ChunkAnnotation(
                    chunk_index=c.chunk_index,
                    input_text=c.input_text,
                    output=out_obj,
                )
            )

    out_path = Path(args.out) if args.out else editable_root() / f"{Path(args.chunks).stem}_editable.txt"
    write_annotations_txt(annotations, out_path)

    print(f"[OK] Wrote editable file: {out_path}")


def cmd_import_annotations(args: argparse.Namespace) -> None:
    schema_wrapper = load_schema(args.schema)

    editable_path = Path(args.editable)
    annotations = read_annotations_txt(editable_path)

    for ann in annotations:
        if ann.output is None:
            raise ValueError(f"Chunk {ann.chunk_index}: missing output JSON")
        validate_top_level_output(ann.output, schema_wrapper)

    records = build_annotation_records(
        schema_wrapper=schema_wrapper,
        annotations=annotations,
        source_id=args.source_id,
        annotator_id=args.annotator_id,
    )

    default_out_name = editable_path.stem.replace("_editable", "") + ".jsonl"
    out_path = Path(args.out) if args.out else annotations_root() / default_out_name
    write_annotations_jsonl(out_path, records)

    print(f"[OK] Wrote annotations JSONL: {out_path}")


def cmd_build_sft(args: argparse.Namespace) -> None:
    schema_wrapper = load_schema(args.schema)

    system_prompt = build_system_prompt(
        schema_name=args.schema,
        schema_obj=schema_wrapper.get("schema", {}),
        prompt_path=Path(args.prompt_path),
        inject_schema=not args.no_inject_schema,
        additional_context=None,
    )

    annotations_paths = [Path(p) for p in args.annotations]
    out_dir = Path(args.out_dir) if args.out_dir else (datasets_root() / args.dataset_id)

    train_path, val_path = build_sft_dataset(
        annotations_paths=annotations_paths,
        out_dir=out_dir,
        system_prompt=system_prompt,
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
    )

    print(f"[OK] Wrote SFT train JSONL: {train_path}")
    if val_path is not None:
        print(f"[OK] Wrote SFT val JSONL: {val_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fine_tuning")
    sub = parser.add_subparsers(dest="command", required=True)

    p_editable = sub.add_parser("create-editable")
    p_editable.add_argument("--schema", required=True)
    p_editable.add_argument("--chunks", required=True)
    p_editable.add_argument("--out")
    p_editable.add_argument("--model")
    p_editable.add_argument("--prompt-path", default=str(_default_prompt_path()))
    p_editable.add_argument("--no-inject-schema", action="store_true")
    p_editable.add_argument("--blank", action="store_true")
    p_editable.set_defaults(func=cmd_create_editable)

    p_import = sub.add_parser("import-annotations")
    p_import.add_argument("--schema", required=True)
    p_import.add_argument("--editable", required=True)
    p_import.add_argument("--out")
    p_import.add_argument("--source-id")
    p_import.add_argument("--annotator-id")
    p_import.set_defaults(func=cmd_import_annotations)

    p_sft = sub.add_parser("build-sft")
    p_sft.add_argument("--schema", required=True)
    p_sft.add_argument("--annotations", required=True, nargs="+")
    p_sft.add_argument("--dataset-id", required=True)
    p_sft.add_argument("--out-dir")
    p_sft.add_argument("--prompt-path", default=str(_default_prompt_path()))
    p_sft.add_argument("--no-inject-schema", action="store_true")
    p_sft.add_argument("--val-ratio", default="0.1")
    p_sft.add_argument("--seed", default="0")
    p_sft.set_defaults(func=cmd_build_sft)

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
