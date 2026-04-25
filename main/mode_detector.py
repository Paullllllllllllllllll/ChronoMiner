# modules/cli/mode_detector.py

"""
Utility to detect execution mode (interactive UI vs CLI) based on configuration and arguments.
"""

import sys
from typing import Any


def detect_execution_mode(config_loader: Any) -> bool:
    """
    Detect whether interactive mode should be used.

    Returns True if interactive UI prompts should be used,
    False for CLI mode.
    """
    # Check if CLI arguments were provided (beyond just script name)
    has_cli_args = len(sys.argv) > 1

    # Get interactive_mode from config (defaults to True if not specified)
    paths_config = config_loader.get_paths_config()
    is_interactive = paths_config.get("general", {}).get("interactive_mode", True)

    # If CLI args are provided, override config and use CLI mode
    if has_cli_args:
        is_interactive = False

    return is_interactive


def get_mode_description(is_interactive: bool) -> str:
    """Get a human-readable description of the current mode."""
    return "Interactive UI Mode" if is_interactive else "CLI Mode"
