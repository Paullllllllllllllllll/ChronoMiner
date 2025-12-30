"""
ChronoMiner UI Module.

Provides user interface components for interactive mode.

Includes modular prompt utilities synchronized with ChronoTranscriber
for consistent UX across projects.

Structure:
- core.py: UserInterface class (backward-compatible wrapper)
- prompts.py: Core prompt utilities with navigation support
- workflows.py: WorkflowUI class for multi-step workflows
- cost_display.py: Cost analysis display utilities
"""

from modules.ui.core import UserInterface
from modules.ui.prompts import (
    NavigationAction,
    PromptResult,
    PromptStyle,
    ui_print,
    ui_input,
    print_header,
    print_separator,
    print_info,
    print_success,
    print_warning,
    print_error,
    print_navigation_help,
    prompt_select,
    prompt_yes_no,
    prompt_text,
    prompt_multiselect,
    confirm_action,
)
from modules.ui.workflows import WorkflowUI

__all__ = [
    # Core UI class (backward-compatible)
    "UserInterface",
    # Navigation
    "NavigationAction",
    "PromptResult",
    "PromptStyle",
    # Print utilities
    "ui_print",
    "ui_input",
    "print_header",
    "print_separator",
    "print_info",
    "print_success",
    "print_warning",
    "print_error",
    "print_navigation_help",
    # Prompt functions
    "prompt_select",
    "prompt_yes_no",
    "prompt_text",
    "prompt_multiselect",
    "confirm_action",
    # Workflow UI
    "WorkflowUI",
]
