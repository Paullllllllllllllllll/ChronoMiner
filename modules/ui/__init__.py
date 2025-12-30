"""
ChronoMiner UI Module.

Provides user interface components for interactive mode.

Includes modular prompt utilities synchronized with ChronoTranscriber
for consistent UX across projects.
"""

from modules.ui.core import UserInterface
from modules.ui.messaging import MessagingAdapter, create_messaging_adapter
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
    # Core UI class
    "UserInterface",
    # Messaging
    "MessagingAdapter",
    "create_messaging_adapter",
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
