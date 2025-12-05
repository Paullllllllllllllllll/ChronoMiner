#!/usr/bin/env python3
"""
Ground Truth Preparation Helper for ChronoMiner Extraction Evaluation.

This script facilitates creating and editing ground truth extractions for
the evaluation framework. It provides two main operations:

1. EXTRACT: Export chunk-by-chunk extractions from model output JSONL files
   to an editable text file with chunk markers and pretty-printed JSON.

2. APPLY: Import corrected extractions from the edited text file back into
   ground truth JSONL format.

Usage:
    # Extract extractions for editing
    python main/prepare_extraction_ground_truth.py --extract --input eval/test_data/output/bibliography/gpt_5.1_medium
    
    # Apply corrections back to ground truth
    python main/prepare_extraction_ground_truth.py --apply --input eval/test_data/output/bibliography/gpt_5.1_medium/source_editable.txt
    
    # Specify custom ground truth output directory
    python main/prepare_extraction_ground_truth.py --apply --input source_editable.txt --output eval/test_data/ground_truth/bibliography
    
    # Show status of ground truth coverage
    python main/prepare_extraction_ground_truth.py --status
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.jsonl_eval import (
    DocumentExtractions,
    export_chunks_to_editable_txt,
    find_jsonl_file,
    import_chunks_from_editable_txt,
    parse_extraction_jsonl,
    write_ground_truth_jsonl,
    convert_legacy_json_to_jsonl,
)


def print_info(msg: str) -> None:
    """Print info message."""
    print(f"[INFO] {msg}")


def print_success(msg: str) -> None:
    """Print success message."""
    print(f"[SUCCESS] {msg}")


def print_error(msg: str) -> None:
    """Print error message."""
    print(f"[ERROR] {msg}", file=sys.stderr)


def print_warning(msg: str) -> None:
    """Print warning message."""
    print(f"[WARN] {msg}")


def extract_for_editing(
    input_path: Path,
    output_dir: Optional[Path] = None,
    include_text: bool = False,
) -> int:
    """
    Extract chunk extractions from JSONL files to editable text format.
    
    Args:
        input_path: Path to JSONL file or directory containing JSONL files
        output_dir: Optional output directory (defaults to same as input)
        include_text: Whether to include original chunk text as comments
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    print_info(f"Extracting extractions from: {input_path}")
    
    # Find JSONL files
    jsonl_files: list[Path] = []
    
    if input_path.is_file() and input_path.suffix == ".jsonl":
        jsonl_files.append(input_path)
    elif input_path.is_dir():
        # Search for JSONL files in directory and subdirectories
        jsonl_files.extend(input_path.rglob("*.jsonl"))
    else:
        print_error(f"Input path does not exist or is not a valid JSONL file/directory: {input_path}")
        return 1
    
    if not jsonl_files:
        print_error(f"No JSONL files found in: {input_path}")
        return 1
    
    print_info(f"Found {len(jsonl_files)} JSONL file(s)")
    
    # Process each JSONL file
    extracted_count = 0
    for jsonl_path in sorted(jsonl_files):
        # Skip batch tracking files and other non-extraction JSONL
        if "_batch_" in jsonl_path.name or "debug" in jsonl_path.name.lower():
            continue
        
        doc = parse_extraction_jsonl(jsonl_path)
        
        if not doc.chunks:
            print_warning(f"No chunks found in: {jsonl_path.name}")
            continue
        
        # Determine output path
        if output_dir:
            out_dir = output_dir
        else:
            out_dir = jsonl_path.parent
        
        out_dir.mkdir(parents=True, exist_ok=True)
        editable_path = out_dir / f"{jsonl_path.stem.replace('_temp', '')}_editable.txt"
        
        export_chunks_to_editable_txt(doc, editable_path, include_chunk_text=include_text)
        
        print_success(f"Exported {doc.chunk_count()} chunks -> {editable_path.name}")
        extracted_count += 1
    
    if extracted_count == 0:
        print_error("No extractions were extracted")
        return 1
    
    print_info(f"Extraction complete. {extracted_count} file(s) created.")
    print_info("Edit the _editable.txt files to correct extractions, then run with --apply")
    
    return 0


def apply_corrections(
    input_path: Path,
    output_dir: Optional[Path] = None,
    category: Optional[str] = None,
    backup: bool = True,
    dry_run: bool = False,
) -> int:
    """
    Apply corrected extractions from edited text file to ground truth JSONL.
    
    Args:
        input_path: Path to edited text file or directory containing edited files
        output_dir: Output directory for ground truth JSONL files
        category: Category name (inferred from path if not provided)
        backup: Whether to create backup of existing ground truth
        dry_run: If True, only show what would be done without writing
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    print_info(f"Applying corrections from: {input_path}")
    
    if dry_run:
        print_info("DRY RUN - no files will be modified")
    
    # Find editable text files
    txt_files: list[Path] = []
    
    if input_path.is_file() and input_path.suffix == ".txt":
        txt_files.append(input_path)
    elif input_path.is_dir():
        txt_files.extend(input_path.rglob("*_editable.txt"))
    else:
        print_error(f"Input path does not exist: {input_path}")
        return 1
    
    if not txt_files:
        print_error(f"No _editable.txt files found in: {input_path}")
        return 1
    
    print_info(f"Found {len(txt_files)} editable file(s)")
    
    # Determine default output directory
    if output_dir is None:
        # Try to infer from path structure
        # Expected: .../output/{category}/{model}/source_editable.txt
        # Output to: .../ground_truth/{category}/source.jsonl
        eval_dir = PROJECT_ROOT / "eval"
        output_dir = eval_dir / "test_data" / "ground_truth"
        print_info(f"Using default ground truth directory: {output_dir}")
    
    # Process each edited file
    applied_count = 0
    for txt_path in sorted(txt_files):
        # Try to find original JSONL for metadata
        original_jsonl_name = txt_path.stem.replace("_editable", "") + "_temp.jsonl"
        original_jsonl = txt_path.parent / original_jsonl_name
        
        # Also try without _temp suffix
        if not original_jsonl.exists():
            original_jsonl = txt_path.parent / (txt_path.stem.replace("_editable", "") + ".jsonl")
        
        original_doc = None
        if original_jsonl.exists():
            original_doc = parse_extraction_jsonl(original_jsonl)
        
        # Import corrected extractions
        corrected_doc = import_chunks_from_editable_txt(txt_path, original_doc)
        
        if not corrected_doc.chunks:
            print_warning(f"No chunks parsed from: {txt_path.name}")
            continue
        
        # Determine category from path if not provided
        actual_category = category
        if actual_category is None:
            # Try to extract from path: .../output/{category}/{model}/...
            parts = txt_path.parts
            for i, part in enumerate(parts):
                if part == "output" and i + 1 < len(parts):
                    actual_category = parts[i + 1]
                    break
        
        if actual_category is None:
            actual_category = "uncategorized"
            print_warning(f"Could not determine category, using: {actual_category}")
        
        # Build output path
        source_name = txt_path.stem.replace("_editable", "").replace("_temp", "")
        gt_dir = output_dir / actual_category
        gt_path = gt_dir / f"{source_name}.jsonl"
        
        print_info(f"Processing: {txt_path.name}")
        print_info(f"  Chunks: {corrected_doc.chunk_count()}")
        print_info(f"  Category: {actual_category}")
        print_info(f"  Output: {gt_path}")
        
        if dry_run:
            print_info("  [DRY RUN] Would write ground truth JSONL")
            applied_count += 1
            continue
        
        # Backup existing ground truth
        if backup and gt_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = gt_path.with_suffix(f".jsonl.bak_{timestamp}")
            shutil.copy2(gt_path, backup_path)
            print_info(f"  Backed up existing file to: {backup_path.name}")
        
        # Write ground truth JSONL
        write_ground_truth_jsonl(corrected_doc, gt_path)
        print_success(f"  Written: {gt_path.name}")
        applied_count += 1
    
    if applied_count == 0:
        print_error("No corrections were applied")
        return 1
    
    action = "would be created" if dry_run else "created"
    print_info(f"Complete. {applied_count} ground truth file(s) {action}.")
    
    return 0


def convert_legacy(
    input_path: Path,
    output_dir: Optional[Path] = None,
    category: Optional[str] = None,
) -> int:
    """
    Convert legacy JSON ground truth files to JSONL format.
    
    Args:
        input_path: Path to JSON file or directory containing JSON files
        output_dir: Output directory for JSONL files
        category: Category name
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    print_info(f"Converting legacy JSON files from: {input_path}")
    
    # Find JSON files
    json_files: list[Path] = []
    
    if input_path.is_file() and input_path.suffix == ".json":
        json_files.append(input_path)
    elif input_path.is_dir():
        json_files.extend(input_path.glob("*.json"))
    else:
        print_error(f"Input path does not exist: {input_path}")
        return 1
    
    if not json_files:
        print_error(f"No JSON files found in: {input_path}")
        return 1
    
    # Determine output directory
    if output_dir is None:
        output_dir = input_path if input_path.is_dir() else input_path.parent
    
    converted_count = 0
    for json_path in sorted(json_files):
        jsonl_path = output_dir / f"{json_path.stem}.jsonl"
        
        convert_legacy_json_to_jsonl(json_path, jsonl_path)
        print_success(f"Converted: {json_path.name} -> {jsonl_path.name}")
        converted_count += 1
    
    print_info(f"Complete. {converted_count} file(s) converted.")
    return 0


def show_status(eval_dir: Path) -> int:
    """
    Show status of ground truth files vs available model outputs.
    
    Args:
        eval_dir: Path to eval directory
        
    Returns:
        Exit code
    """
    test_data = eval_dir / "test_data"
    output_dir = test_data / "output"
    gt_dir = test_data / "ground_truth"
    
    if not output_dir.exists():
        print_error(f"Output directory not found: {output_dir}")
        return 1
    
    print_info("Ground Truth Status")
    print("=" * 60)
    
    # Find all categories
    categories = sorted([d.name for d in output_dir.iterdir() if d.is_dir()])
    
    for category in categories:
        print(f"\n{category.upper()}")
        print("-" * 40)
        
        cat_output = output_dir / category
        cat_gt = gt_dir / category
        
        # Find all model outputs
        models = sorted([d.name for d in cat_output.iterdir() if d.is_dir()])
        
        # Find all sources (JSONL files)
        sources: set[str] = set()
        for model_dir in cat_output.iterdir():
            if model_dir.is_dir():
                for jsonl in model_dir.rglob("*.jsonl"):
                    if "_batch_" not in jsonl.name and "debug" not in jsonl.name.lower():
                        # Remove _temp suffix for source name
                        source_name = jsonl.stem.replace("_temp", "")
                        sources.add(source_name)
        
        print(f"  Models with output: {len(models)}")
        for m in models:
            print(f"    - {m}")
        
        print(f"  Sources found: {len(sources)}")
        
        # Check ground truth status
        gt_count = 0
        gt_sources: set[str] = set()
        if cat_gt.exists():
            gt_files = list(cat_gt.glob("*.jsonl")) + list(cat_gt.glob("*.json"))
            gt_count = len(gt_files)
            gt_sources = {f.stem for f in gt_files}
            missing = sources - gt_sources
        else:
            missing = sources
        
        print(f"  Ground truth files: {gt_count}")
        
        if missing:
            print(f"  Missing ground truth ({len(missing)}):")
            for s in sorted(missing)[:5]:
                print(f"    - {s}")
            if len(missing) > 5:
                print(f"    ... and {len(missing) - 5} more")
        else:
            print("  All sources have ground truth ✓")
    
    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Ground truth preparation helper for extraction evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract extractions for manual editing
  python main/prepare_extraction_ground_truth.py --extract --input eval/test_data/output/bibliography/gpt_5.1_medium
  
  # Include original chunk text as comments
  python main/prepare_extraction_ground_truth.py --extract --input folder --include-text
  
  # Apply corrections to create ground truth
  python main/prepare_extraction_ground_truth.py --apply --input eval/test_data/output/bibliography/gpt_5.1_medium
  
  # Dry run to see what would be written
  python main/prepare_extraction_ground_truth.py --apply --input folder_editable.txt --dry-run
  
  # Convert legacy JSON ground truth to JSONL
  python main/prepare_extraction_ground_truth.py --convert --input eval/test_data/ground_truth/bibliography
  
  # Show status of ground truth coverage
  python main/prepare_extraction_ground_truth.py --status
        """,
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--extract",
        action="store_true",
        help="Extract extractions from JSONL to editable text format",
    )
    mode_group.add_argument(
        "--apply",
        action="store_true",
        help="Apply corrections from edited text to ground truth JSONL",
    )
    mode_group.add_argument(
        "--convert",
        action="store_true",
        help="Convert legacy JSON ground truth to JSONL format",
    )
    mode_group.add_argument(
        "--status",
        action="store_true",
        help="Show status of ground truth files",
    )
    
    # Input/output options
    parser.add_argument(
        "--input", "-i",
        type=Path,
        help="Input path (JSONL file/directory for extract, edited txt for apply)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output directory (optional, defaults based on mode)",
    )
    parser.add_argument(
        "--category", "-c",
        type=str,
        help="Category name (auto-detected if not provided)",
    )
    
    # Options
    parser.add_argument(
        "--include-text",
        action="store_true",
        help="Include original chunk text as comments in editable file",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create backup of existing ground truth files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without writing files",
    )
    
    args = parser.parse_args()
    
    # Handle status mode
    if args.status:
        eval_dir = PROJECT_ROOT / "eval"
        return show_status(eval_dir)
    
    # Validate input for other modes
    if args.input is None:
        print_error("--input is required for --extract, --apply, and --convert modes")
        return 1
    
    # Resolve relative paths
    input_path = args.input
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path
    
    output_dir = args.output
    if output_dir and not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    
    # Execute mode
    if args.extract:
        return extract_for_editing(input_path, output_dir, include_text=args.include_text)
    elif args.apply:
        return apply_corrections(
            input_path,
            output_dir,
            args.category,
            backup=not args.no_backup,
            dry_run=args.dry_run,
        )
    elif args.convert:
        return convert_legacy(input_path, output_dir, args.category)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
