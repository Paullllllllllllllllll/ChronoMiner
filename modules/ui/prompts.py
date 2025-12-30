"""Enhanced user prompting utilities with navigation support.

This module provides a clean separation between user interaction and logging,
with consistent visual formatting and navigation options (back/quit).

Synchronized with ChronoTranscriber's UI system for consistent UX across projects.
"""

from __future__ import annotations

import sys
from enum import Enum
from typing import Any, Callable, List, Optional, Tuple
from dataclasses import dataclass

# Initialize colorama for Windows color support
try:
    import colorama
    colorama.just_fix_windows_console()
except ImportError:
    pass  # colorama not available, colors may not work on Windows

from modules.core.logger import setup_logger

logger = setup_logger(__name__)


class NavigationAction(Enum):
    """User navigation choices in interactive prompts."""
    CONTINUE = "continue"
    BACK = "back"
    QUIT = "quit"


@dataclass
class PromptResult:
    """Result from a prompt with navigation support."""
    action: NavigationAction
    value: Any = None


class PromptStyle:
    """Visual styling constants for consistent UI."""
    
    # Box drawing characters - ASCII-safe for Windows compatibility
    DOUBLE_LINE = "="
    SINGLE_LINE = "-"
    LIGHT_LINE = "."
    
    # ANSI Color codes - now work on Windows thanks to colorama
    HEADER = "\033[1;36m"      # Cyan bold (headers, titles)
    INFO = "\033[0;36m"        # Cyan (informational messages)
    SUCCESS = "\033[1;32m"     # Green bold (success messages)
    WARNING = "\033[1;33m"     # Yellow bold (warnings)
    ERROR = "\033[1;31m"       # Red bold (errors)
    PROMPT = "\033[1;37m"      # White bold (user prompts)
    DIM = "\033[2;37m"         # Dimmed white (secondary text)
    HIGHLIGHT = "\033[1;35m"   # Magenta bold (highlights)
    RESET = "\033[0m"          # Reset to default
    
    @staticmethod
    def supports_color() -> bool:
        """Check if terminal supports color."""
        # With colorama, we can safely assume color support
        return True
    
    @classmethod
    def colorize(cls, text: str, color: str) -> str:
        """Apply color to text if terminal supports it."""
        if cls.supports_color():
            return f"{color}{text}{cls.RESET}"
        return text


def ui_print(message: str, style: str = "", end: str = "\n") -> None:
    """Print a UI message (distinct from logging).
    
    Args:
        message: The message to display
        style: Optional style/color code
        end: String appended after the message (default: newline)
    """
    try:
        if style and PromptStyle.supports_color():
            # Encode with UTF-8 and handle errors gracefully
            output = f"{style}{message}{PromptStyle.RESET}"
            print(output, end=end, flush=True)
        else:
            print(message, end=end, flush=True)
    except UnicodeEncodeError:
        # Fallback to ASCII if encoding fails
        safe_message = message.encode('ascii', 'replace').decode('ascii')
        if style and PromptStyle.supports_color():
            print(f"{style}{safe_message}{PromptStyle.RESET}", end=end, flush=True)
        else:
            print(safe_message, end=end, flush=True)


def ui_input(prompt: str, style: str = PromptStyle.PROMPT) -> str:
    """Get user input with consistent styling.
    
    Args:
        prompt: The prompt message
        style: Optional style/color code
        
    Returns:
        User input stripped of whitespace
    """
    styled_prompt = PromptStyle.colorize(prompt, style) if style else prompt
    try:
        return input(styled_prompt).strip()
    except (EOFError, KeyboardInterrupt):
        ui_print("\n[INFO] Operation cancelled by user.", PromptStyle.INFO)
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error reading input: {e}")
        ui_print("[ERROR] Unable to read input.", PromptStyle.ERROR)
        sys.exit(1)


def print_header(title: str, subtitle: str = "") -> None:
    """Print a formatted header for a section.
    
    Args:
        title: Main title text
        subtitle: Optional subtitle text
    """
    width = 80
    ui_print("\n" + PromptStyle.DOUBLE_LINE * width, PromptStyle.HEADER)
    ui_print(f"  {title}", PromptStyle.HEADER)
    if subtitle:
        ui_print(f"  {subtitle}", PromptStyle.INFO)
    ui_print(PromptStyle.DOUBLE_LINE * width, PromptStyle.HEADER)
    ui_print("")  # Extra spacing


def print_separator(char: str = PromptStyle.SINGLE_LINE, width: int = 80) -> None:
    """Print a separator line."""
    ui_print(char * width, PromptStyle.DIM)


def print_info(message: str, prefix: str = "[INFO]") -> None:
    """Print an informational message."""
    ui_print(f"{prefix} {message}", PromptStyle.INFO)


def print_success(message: str, prefix: str = "[SUCCESS]") -> None:
    """Print a success message."""
    ui_print(f"{prefix} {message}", PromptStyle.SUCCESS)


def print_warning(message: str, prefix: str = "[WARNING]") -> None:
    """Print a warning message."""
    ui_print(f"{prefix} {message}", PromptStyle.WARNING)


def print_error(message: str, prefix: str = "[ERROR]") -> None:
    """Print an error message."""
    ui_print(f"{prefix} {message}", PromptStyle.ERROR)


def print_navigation_help(allow_back: bool = False) -> None:
    """Print navigation options help text.
    
    Args:
        allow_back: Whether to show the 'back' option
    """
    options = []
    if allow_back:
        options.append("'b' to go back")
    options.append("'q' to quit")
    
    if options:
        help_text = " | ".join(options)
        ui_print(f"  {PromptStyle.LIGHT_LINE * 3} {help_text}", PromptStyle.DIM)


def handle_navigation_input(user_input: str, allow_back: bool = False) -> Optional[NavigationAction]:
    """Check if user input is a navigation command.
    
    Args:
        user_input: The user's input
        allow_back: Whether back navigation is allowed
        
    Returns:
        NavigationAction if input is a navigation command, None otherwise
    """
    input_lower = user_input.lower()
    
    if input_lower in ["q", "quit", "exit"]:
        print_info("Exiting as requested.")
        return NavigationAction.QUIT
    
    if allow_back and input_lower in ["b", "back"]:
        print_info("Going back to previous step...")
        return NavigationAction.BACK
    
    return None


def prompt_select(
    question: str,
    options: List[Tuple[str, str]],
    allow_back: bool = False,
    show_help: bool = True
) -> PromptResult:
    """Prompt user to select from a list of options.
    
    Args:
        question: The question to ask
        options: List of (value, description) tuples
        allow_back: Whether to allow back navigation
        show_help: Whether to show navigation help
        
    Returns:
        PromptResult with the selected value or navigation action
    """
    ui_print(f"\n{question}", PromptStyle.PROMPT)
    print_separator()
    
    for idx, (value, description) in enumerate(options, 1):
        ui_print(f"  {idx}. {description}")
    
    if show_help:
        print_navigation_help(allow_back)
    
    while True:
        choice = ui_input("\nEnter your choice: ")
        
        # Check for navigation
        nav_action = handle_navigation_input(choice, allow_back)
        if nav_action == NavigationAction.QUIT:
            sys.exit(0)
        if nav_action == NavigationAction.BACK:
            return PromptResult(NavigationAction.BACK)
        
        # Check for valid selection
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(options):
                return PromptResult(NavigationAction.CONTINUE, options[idx - 1][0])
        
        print_error("Invalid selection. Please try again.")


def prompt_yes_no(
    question: str,
    default: Optional[bool] = None,
    allow_back: bool = False
) -> PromptResult:
    """Prompt user for a yes/no answer.
    
    Args:
        question: The question to ask
        default: Default answer if user presses Enter (None for no default)
        allow_back: Whether to allow back navigation
        
    Returns:
        PromptResult with boolean value or navigation action
    """
    # Build prompt suffix
    if default is True:
        suffix = " (Y/n)"
    elif default is False:
        suffix = " (y/N)"
    else:
        suffix = " (y/n)"
    
    ui_print(f"\n{question}{suffix}", PromptStyle.PROMPT)
    
    if allow_back:
        print_navigation_help(allow_back)
    
    while True:
        choice = ui_input("> ").lower()
        
        # Check for navigation
        nav_action = handle_navigation_input(choice, allow_back)
        if nav_action == NavigationAction.QUIT:
            sys.exit(0)
        if nav_action == NavigationAction.BACK:
            return PromptResult(NavigationAction.BACK)
        
        # Handle default
        if choice == "" and default is not None:
            return PromptResult(NavigationAction.CONTINUE, default)
        
        # Check for valid yes/no
        if choice in ["y", "yes"]:
            return PromptResult(NavigationAction.CONTINUE, True)
        if choice in ["n", "no"]:
            return PromptResult(NavigationAction.CONTINUE, False)
        
        print_error("Please enter 'y' for yes or 'n' for no.")


def prompt_text(
    question: str,
    allow_empty: bool = False,
    allow_back: bool = False,
    validator: Optional[Callable[[str], bool]] = None,
    error_message: str = "Invalid input. Please try again.",
    default: str = ""
) -> PromptResult:
    """Prompt user for text input.
    
    Args:
        question: The question to ask
        allow_empty: Whether to allow empty input
        allow_back: Whether to allow back navigation
        validator: Optional validation function
        error_message: Message to show on validation failure
        default: Default value if user presses Enter
        
    Returns:
        PromptResult with text value or navigation action
    """
    prompt_text_display = question
    if default:
        prompt_text_display = f"{question} [{default}]"
    
    ui_print(f"\n{prompt_text_display}", PromptStyle.PROMPT)
    
    if allow_back:
        print_navigation_help(allow_back)
    
    while True:
        value = ui_input("> ")
        
        # Use default if empty and default provided
        if not value and default:
            value = default
        
        # Check for navigation
        nav_action = handle_navigation_input(value, allow_back)
        if nav_action == NavigationAction.QUIT:
            sys.exit(0)
        if nav_action == NavigationAction.BACK:
            return PromptResult(NavigationAction.BACK)
        
        # Check for empty
        if not value and not allow_empty:
            print_error("Input cannot be empty.")
            continue
        
        # Validate
        if validator and not validator(value):
            print_error(error_message)
            continue
        
        return PromptResult(NavigationAction.CONTINUE, value)


def prompt_multiselect(
    question: str,
    items: List[Tuple[str, str]],
    allow_all: bool = True,
    allow_back: bool = False
) -> PromptResult:
    """Prompt user to select multiple items.
    
    Args:
        question: The question to ask
        items: List of (identifier, description) tuples
        allow_all: Whether to allow selecting all items
        allow_back: Whether to allow back navigation
        
    Returns:
        PromptResult with list of selected identifiers or navigation action
    """
    ui_print(f"\n{question}", PromptStyle.PROMPT)
    print_separator()
    
    for idx, (identifier, description) in enumerate(items, 1):
        # Truncate long descriptions
        if len(description) > 70:
            description = description[:67] + "..."
        ui_print(f"  {idx}. {description}")
    
    ui_print(f"\n  Selection options:", PromptStyle.INFO)
    ui_print("    - Enter numbers separated by commas (e.g., '1,3,5')", PromptStyle.DIM)
    ui_print("    - Enter a range with a dash (e.g., '1-5')", PromptStyle.DIM)
    if allow_all:
        ui_print("    - Enter 'all' to select everything", PromptStyle.DIM)
    
    if allow_back:
        ui_print("")  # Spacing
        print_navigation_help(allow_back)
    
    while True:
        choice = ui_input("\nYour selection: ").lower()
        
        # Check for navigation
        nav_action = handle_navigation_input(choice, allow_back)
        if nav_action == NavigationAction.QUIT:
            sys.exit(0)
        if nav_action == NavigationAction.BACK:
            return PromptResult(NavigationAction.BACK)
        
        # Check for 'all'
        if allow_all and choice == "all":
            selected = [identifier for identifier, _ in items]
            return PromptResult(NavigationAction.CONTINUE, selected)
        
        # Parse selection
        try:
            indices = set()
            parts = choice.split(",")
            for part in parts:
                part = part.strip()
                if "-" in part:
                    # Range
                    start, end = part.split("-", 1)
                    start_idx = int(start.strip())
                    end_idx = int(end.strip())
                    indices.update(range(start_idx, end_idx + 1))
                else:
                    # Single number
                    indices.add(int(part))
            
            # Validate indices
            valid_indices = [i for i in indices if 1 <= i <= len(items)]
            if not valid_indices:
                print_error("No valid selections made.")
                continue
            
            selected = [items[i - 1][0] for i in sorted(valid_indices)]
            return PromptResult(NavigationAction.CONTINUE, selected)
            
        except (ValueError, IndexError):
            print_error("Invalid format. Please try again.")


def confirm_action(message: str, default: bool = False) -> bool:
    """Simple confirmation prompt (no back navigation).
    
    Args:
        message: Confirmation message
        default: Default answer
        
    Returns:
        True if confirmed, False otherwise
    """
    result = prompt_yes_no(message, default=default, allow_back=False)
    return result.value if result.action == NavigationAction.CONTINUE else False
