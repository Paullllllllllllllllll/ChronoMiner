# main/cost_analysis.py

"""
Script for analyzing token costs from temporary .jsonl files.

Supports two execution modes:
1. Interactive Mode: User-friendly prompts and selections via UI
2. CLI Mode: Command-line arguments for automation and scripting

The mode is controlled by the 'interactive_mode' setting in config/paths_config.yaml
or by providing command-line arguments.

Workflow:
 1. Load configuration and determine input/output paths
 2. Scan for temporary .jsonl files
 3. Extract token usage data from each file
 4. Calculate costs based on model pricing
 5. Display statistics with standard and 50% discounted pricing
 6. Optionally save results as CSV
"""

import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict

from modules.cli.mode_detector import should_use_interactive_mode
from modules.core.logger import setup_logger
from modules.core.workflow_utils import load_core_resources
from modules.operations.cost_analysis import (
    find_jsonl_files,
    perform_cost_analysis,
    save_analysis_to_csv,
)
from modules.ui.core import UserInterface
from modules.ui.cost_display import display_analysis

# Initialize logger
logger = setup_logger(__name__)


def _run_interactive_mode(
    paths_config: Dict,
    schemas_paths: Dict,
) -> None:
    """Run cost analysis in interactive mode."""
    ui = UserInterface(logger, use_colors=True)
    ui.display_banner()
    
    ui.print_info("Loading configuration...")
    logger.info("Starting cost analysis (Interactive Mode)")
    
    # Find .jsonl files
    ui.print_info("Scanning for temporary .jsonl files...")
    jsonl_files = find_jsonl_files(paths_config, schemas_paths)
    
    if not jsonl_files:
        ui.print_warning("No temporary .jsonl files found.")
        logger.warning("No .jsonl files found for analysis")
        return
    
    ui.print_success(f"Found {len(jsonl_files)} file(s) to analyze")
    
    # Perform analysis
    ui.print_info("Analyzing token usage...")
    analysis = perform_cost_analysis(jsonl_files)
    
    # Display results
    display_analysis(analysis, ui)
    
    # Ask to save CSV
    if analysis.file_stats:
        save_csv = ui.confirm("Save results as CSV?", default=True)
        
        if save_csv:
            # Determine output directory (use first file's directory)
            output_dir = analysis.file_stats[0].file_path.parent
            output_path = output_dir / "cost_analysis.csv"
            
            try:
                save_analysis_to_csv(analysis, output_path)
                ui.print_success(f"Saved to: {output_path}")
            except Exception as e:
                ui.print_error(f"Failed to save CSV: {e}")
    
    ui.console_print(f"\n{ui.BOLD}Analysis complete!{ui.RESET}\n")


def _run_cli_mode(
    paths_config: Dict,
    schemas_paths: Dict,
) -> None:
    """Run cost analysis in CLI mode."""
    parser = ArgumentParser(
        description="Analyze token costs from temporary .jsonl files"
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Save results as CSV file"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for CSV file (default: cost_analysis.csv in first file's directory)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output"
    )
    
    args = parser.parse_args()
    
    logger.info("Starting cost analysis (CLI Mode)")
    
    # Find .jsonl files
    if not args.quiet:
        print("[INFO] Scanning for temporary .jsonl files...")
    
    jsonl_files = find_jsonl_files(paths_config, schemas_paths)
    
    if not jsonl_files:
        print("[WARNING] No temporary .jsonl files found.")
        logger.warning("No .jsonl files found for analysis")
        sys.exit(0)
    
    if not args.quiet:
        print(f"[INFO] Found {len(jsonl_files)} file(s) to analyze")
    
    # Perform analysis
    if not args.quiet:
        print("[INFO] Analyzing token usage...")
    
    analysis = perform_cost_analysis(jsonl_files)
    
    # Display results
    if not args.quiet:
        display_analysis(analysis, ui=None)
    
    # Save CSV if requested
    if args.save_csv and analysis.file_stats:
        if args.output:
            output_path = Path(args.output)
        else:
            output_dir = analysis.file_stats[0].file_path.parent
            output_path = output_dir / "cost_analysis.csv"
        
        try:
            save_analysis_to_csv(analysis, output_path)
            print(f"[SUCCESS] Saved to: {output_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save CSV: {e}")
            sys.exit(1)
    
    logger.info("Cost analysis complete")


def main() -> None:
    """Main entry point."""
    try:
        # Load configuration
        (
            config_loader,
            paths_config,
            model_config,
            chunking_and_context_config,
            schemas_paths,
        ) = load_core_resources()
        
        if should_use_interactive_mode(config_loader):
            _run_interactive_mode(paths_config, schemas_paths)
        else:
            _run_cli_mode(paths_config, schemas_paths)
    
    except KeyboardInterrupt:
        logger.info("Cost analysis interrupted by user")
        print("\n[INFO] Operation cancelled by user")
        sys.exit(0)
    except Exception as exc:
        logger.exception("Unexpected error in cost analysis", exc_info=exc)
        print(f"[ERROR] Unexpected error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
