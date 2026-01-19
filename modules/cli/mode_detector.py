# modules/cli/mode_detector.py

"""
Utility to detect execution mode (interactive UI vs CLI) based on configuration and arguments.
"""

import sys
from typing import Any, Tuple, Optional
from pathlib import Path


def detect_execution_mode(config_loader: Any) -> Tuple[bool, bool]:
    """
    Detect the execution mode based on configuration and command-line arguments.
    
    Returns:
        Tuple of (is_interactive, has_cli_args)
        - is_interactive: True if should use interactive UI prompts
        - has_cli_args: True if command-line arguments were provided
    """
    # Check if CLI arguments were provided (beyond just script name)
    has_cli_args = len(sys.argv) > 1
    
    # Get interactive_mode from config (defaults to True if not specified)
    paths_config = config_loader.get_paths_config()
    is_interactive = paths_config.get("general", {}).get("interactive_mode", True)
    
    # If CLI args are provided, override config and use CLI mode
    if has_cli_args:
        # Check if it's a help flag
        if any(arg in ['-h', '--help'] for arg in sys.argv):
            return False, True  # CLI mode for help
        is_interactive = False
    
    return is_interactive, has_cli_args


def should_use_interactive_mode(config_loader: Any) -> bool:
    """
    Determine if interactive mode should be used.
    
    Returns True if interactive UI should be used, False for CLI mode.
    """
    is_interactive, _ = detect_execution_mode(config_loader)
    return is_interactive


def get_mode_description(is_interactive: bool) -> str:
    """Get a human-readable description of the current mode."""
    return "Interactive UI Mode" if is_interactive else "CLI Mode"
