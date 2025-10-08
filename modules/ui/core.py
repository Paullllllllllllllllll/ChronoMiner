# modules/ui/core.py

"""
User Interface Module for ChronoMiner

Provides a consistent, user-friendly interface for all interactive operations.
Separates user-facing prompts from detailed logging.
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class UserInterface:
    """
    Handles all user interaction for ChronoMiner with consistent formatting and navigation.
    
    Key Features:
    - Clear visual separation with box-drawing characters
    - Consistent exit/back navigation options
    - Separation of user prompts from detailed logging
    - Enhanced error handling and validation
    """

    # Visual styling constants
    HORIZONTAL_LINE = "─" * 80
    DOUBLE_LINE = "═" * 80
    SECTION_START = "┌" + "─" * 78 + "┐"
    SECTION_END = "└" + "─" * 78 + "┘"
    
    # Color codes for terminals that support them (optional)
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    SUCCESS = "\033[92m"
    WARNING = "\033[93m"
    ERROR = "\033[91m"
    INFO = "\033[94m"
    PROMPT = "\033[96m"

    def __init__(self, logger: Optional[logging.Logger] = None, use_colors: bool = False) -> None:
        """
        Initialize the user interface.

        :param logger: Optional logger instance for detailed logging separate from user output
        :param use_colors: Whether to use ANSI color codes (default False for compatibility)
        """
        self.logger = logger
        self.use_colors = use_colors
        if not use_colors:
            # Disable colors for better compatibility
            self.RESET = self.BOLD = self.DIM = ""
            self.SUCCESS = self.WARNING = self.ERROR = self.INFO = self.PROMPT = ""

    def log(self, message: str, level: str = "info") -> None:
        """
        Log a message to the logger without printing to console.
        Use this for detailed technical information.
        
        :param message: Message to log
        :param level: Log level (debug, info, warning, error)
        """
        if self.logger:
            log_method = getattr(self.logger, level.lower(), self.logger.info)
            log_method(message)

    def console_print(self, message: str, log_also: bool = False) -> None:
        """
        Print to console for user interaction.
        Optionally also log the message.
        
        :param message: Message to display
        :param log_also: Whether to also write to log file
        """
        print(message)
        if log_also and self.logger:
            self.logger.info(message)

    def print_success(self, message: str) -> None:
        """Print a success message with visual formatting."""
        self.console_print(f"{self.SUCCESS}✓ {message}{self.RESET}")
        self.log(f"SUCCESS: {message}", "info")

    def print_error(self, message: str) -> None:
        """Print an error message with visual formatting."""
        self.console_print(f"{self.ERROR}✗ {message}{self.RESET}")
        self.log(f"ERROR: {message}", "error")

    def print_warning(self, message: str) -> None:
        """Print a warning message with visual formatting."""
        self.console_print(f"{self.WARNING}⚠ {message}{self.RESET}")
        self.log(f"WARNING: {message}", "warning")

    def print_info(self, message: str) -> None:
        """Print an info message with visual formatting."""
        self.console_print(f"{self.INFO}ℹ {message}{self.RESET}")
        self.log(f"INFO: {message}", "info")

    def print_section_header(self, title: str) -> None:
        """Print a prominent section header."""
        self.console_print(f"\n{self.DOUBLE_LINE}")
        self.console_print(f"{self.BOLD}  {title.upper()}{self.RESET}")
        self.console_print(self.DOUBLE_LINE)

    def print_subsection_header(self, title: str) -> None:
        """Print a subsection header."""
        self.console_print(f"\n{self.HORIZONTAL_LINE}")
        self.console_print(f"  {title}")
        self.console_print(self.HORIZONTAL_LINE)

    def display_banner(self) -> None:
        """Display a welcome banner with information about the application."""
        self.console_print("\n" + self.DOUBLE_LINE)
        self.console_print(f"{self.BOLD}  ChronoMiner - Structured Data Extraction Tool{self.RESET}")
        self.console_print(self.DOUBLE_LINE)
        self.console_print("  Extract structured data from historical documents using")
        self.console_print("  advanced AI models and customizable schemas.")
        self.console_print(self.DOUBLE_LINE + "\n")

    def get_input(self, prompt: str, allow_back: bool = False, allow_quit: bool = True) -> Optional[str]:
        """
        Get user input with consistent navigation options.
        
        :param prompt: The prompt to display
        :param allow_back: Whether to allow 'b' to go back
        :param allow_quit: Whether to allow 'q' to quit
        :return: User input or None if back/quit selected
        """
        nav_options = []
        if allow_back:
            nav_options.append("'b' to go back")
        if allow_quit:
            nav_options.append("'q' to quit")
        
        if nav_options:
            nav_hint = f" ({', '.join(nav_options)})"
        else:
            nav_hint = ""
        
        self.console_print(f"\n{self.PROMPT}{prompt}{nav_hint}{self.RESET}")
        try:
            user_input = input("→ ").strip()
        except (EOFError, KeyboardInterrupt):
            self.print_info("Operation cancelled.")
            return None
        
        if allow_quit and user_input.lower() in ['q', 'quit', 'exit']:
            self.print_info("Exiting application.")
            sys.exit(0)
        
        if allow_back and user_input.lower() in ['b', 'back']:
            return None
        
        return user_input

    def confirm(self, message: str, default: bool = False) -> bool:
        """
        Ask for yes/no confirmation.
        
        :param message: Confirmation message
        :param default: Default value if user just presses Enter
        :return: True if confirmed, False otherwise
        """
        hint = "[Y/n]" if default else "[y/N]"
        response = self.get_input(f"{message} {hint}", allow_back=False, allow_quit=True)
        
        if response is None or response == "":
            return default
        
        return response.lower() in ['y', 'yes', 'true', '1']

    def select_option(
        self,
        prompt: str,
        options: List[Tuple[str, str]],
        allow_back: bool = False,
        allow_quit: bool = True,
        show_numbers: bool = True
    ) -> Optional[str]:
        """
        Display a menu with options and return the user's choice.

        :param prompt: The prompt to display
        :param options: List of (value, description) tuples
        :param allow_back: Whether to allow going back to previous step
        :param allow_quit: Whether to allow quitting
        :param show_numbers: Whether to show option numbers (True) or letters (False)
        :return: The selected option value or None if back selected
        """
        self.console_print(f"\n{self.BOLD}{prompt}{self.RESET}")
        self.console_print(self.HORIZONTAL_LINE)

        for idx, (value, description) in enumerate(options, 1):
            marker = f"{idx}." if show_numbers else f"{chr(96 + idx)})"
            self.console_print(f"  {self.BOLD}{marker}{self.RESET} {description}")

        self.console_print("")  # Empty line for spacing

        while True:
            choice = self.get_input("Enter your choice", allow_back=allow_back, allow_quit=allow_quit)

            if choice is None:  # User pressed back
                return None

            if choice.isdigit():
                choice_num = int(choice)
                if 1 <= choice_num <= len(options):
                    selected = options[choice_num - 1][0]
                    self.log(f"User selected option {choice_num}: {selected}")
                    return selected

            self.print_error(f"Invalid selection '{choice}'. Please enter a number between 1 and {len(options)}.")

    def select_schema(self, schema_manager, allow_back: bool = False) -> Optional[Tuple[Dict[str, Any], str]]:
        """
        Present available schemas and guide the user through selection.

        :param schema_manager: SchemaManager instance
        :param allow_back: Whether to allow going back to previous step
        :return: Tuple of (schema_dict, schema_name) or None if back selected
        """
        available_schemas = schema_manager.get_available_schemas()

        if not available_schemas:
            self.print_error("No schemas available. Please add schemas to the 'schemas/' folder.")
            sys.exit(1)

        self.print_section_header("Schema Selection")

        schema_options_with_paths = schema_manager.list_schema_options()
        if schema_options_with_paths:
            schema_options = [
                (name, f"{name} ({path.name})") for name, path in schema_options_with_paths
            ]
        else:
            schema_options = [(name, name) for name in available_schemas.keys()]

        selected_schema_name = self.select_option(
            "Select a schema to use for extraction:",
            schema_options,
            allow_back=allow_back,
            allow_quit=True
        )

        if selected_schema_name is None:
            return None

        self.log(f"Schema selected: {selected_schema_name}")
        return available_schemas[selected_schema_name], selected_schema_name

    def ask_global_chunking_mode(self, allow_back: bool = False) -> Optional[str]:
        """
        Present options for global chunking strategy.

        :param allow_back: Whether to allow going back to previous step
        :return: Selected chunking method or None if back selected
        """
        self.print_section_header("Chunking Strategy")

        chunking_options = [
            ("auto", "Automatic - System splits text into chunks based on token limits"),
            ("auto-adjust", "Auto-adjust - Automatic chunking with intelligent boundary refinement"),
            ("line_ranges.txt", "Use existing line ranges - Process with pre-defined line range files"),
            ("adjust-line-ranges", "Adjust & use - Refine line ranges with AI-detected boundaries, then process"),
            ("per-file", "Per-file selection - Choose chunking method for each file individually")
        ]

        return self.select_option(
            "How would you like to chunk the text for processing?",
            chunking_options,
            allow_back=allow_back,
            allow_quit=True
        )

    def ask_batch_processing(self, allow_back: bool = False) -> Optional[bool]:
        """
        Present options for batch processing selection.

        :param allow_back: Whether to allow going back to previous step
        :return: True if batch processing selected, False for sync, None if back
        """
        self.print_section_header("Processing Mode")

        batch_options = [
            ("sync", "Synchronous - Process in real-time with immediate results"),
            ("batch", "Batch - Submit as batch job (50% cost reduction, results within 24 hours)")
        ]

        mode = self.select_option(
            "Select how would like to process the data:",
            batch_options,
            allow_back=allow_back,
            allow_quit=True
        )

        if mode is None:
            return None
        return mode == "batch"

    def ask_additional_context_mode(self, allow_back: bool = False) -> Optional[Dict[str, Any]]:
        """
        Present options for additional context handling.

        :param allow_back: Whether to allow going back to previous step
        :return: Dictionary with context settings or None if back selected
        """
        self.print_section_header("Additional Context")

        use_context_options = [
            ("yes", "Yes - Provide additional context to improve extraction accuracy"),
            ("no", "No - Process text without additional context")
        ]

        use_context = self.select_option(
            "Would you like to provide additional context for extraction?",
            use_context_options,
            allow_back=allow_back,
            allow_quit=True
        )

        if use_context is None:
            return None

        context_settings = {"use_additional_context": use_context == "yes"}

        if context_settings["use_additional_context"]:
            context_source_options = [
                ("default", "Default - Use schema-specific context files from additional_context/"),
                ("file", "File-specific - Use individual context files (e.g., filename_context.txt)")
            ]

            context_source = self.select_option(
                "Which source of context would you like to use?",
                context_source_options,
                allow_back=True,
                allow_quit=True
            )

            if context_source is None:
                # User went back, loop again
                return self.ask_additional_context_mode(allow_back=allow_back)

            context_settings["use_default_context"] = context_source == "default"

        return context_settings

    def select_input_source(self, raw_text_dir: Path, allow_back: bool = False) -> Optional[List[Path]]:
        """
        Guide user through selecting input source (single file or folder).

        :param raw_text_dir: Base directory for input files
        :param allow_back: Whether to allow going back to previous step
        :return: List of selected file paths or None if back selected
        """
        self.print_section_header("Input Selection")

        mode_options = [
            ("single", "Process a single file"),
            ("folder", "Process all files in a folder")
        ]

        mode = self.select_option(
            "Select how you would like to specify input:",
            mode_options,
            allow_back=allow_back,
            allow_quit=True
        )

        if mode is None:
            return None

        files = []

        if mode == "single":
            self.print_info("Enter the filename to process (with or without .txt extension)")
            self.console_print("  • Enter the base text filename")
            self.console_print("  • Or enter the line range filename ending in '_line_ranges.txt'")
            
            file_input = self.get_input("Filename", allow_back=True, allow_quit=True)
            if not file_input:
                # User went back
                return self.select_input_source(raw_text_dir, allow_back=allow_back)
            
            normalized_input = (
                file_input if file_input.lower().endswith(".txt") else f"{file_input}.txt"
            )

            wants_line_range = normalized_input.lower().endswith(
                ("_line_ranges.txt", "_line_range.txt")
            )
            excluded_suffixes = ["_context.txt"]
            if not wants_line_range:
                excluded_suffixes.extend(["_line_ranges.txt", "_line_range.txt"])

            file_candidates = [
                f
                for f in raw_text_dir.rglob(normalized_input)
                if not any(f.name.endswith(suffix) for suffix in excluded_suffixes)
            ]

            if not file_candidates:
                self.print_error(f"File '{normalized_input}' not found in {raw_text_dir}")
                sys.exit(1)
            elif len(file_candidates) == 1:
                files.append(file_candidates[0])
                self.print_success(f"Selected: {files[0].name}")
            else:
                self.print_warning(f"Found {len(file_candidates)} matching files:")
                self.console_print(self.HORIZONTAL_LINE)

                file_options = [(str(i), str(f.relative_to(raw_text_dir))) for i, f in enumerate(file_candidates)]
                
                for idx, f in enumerate(file_candidates, 1):
                    self.console_print(f"  {idx}. {f.relative_to(raw_text_dir)}")

                while True:
                    selected_index = self.get_input("Select file by number", allow_back=True, allow_quit=True)
                    
                    if not selected_index:
                        # User went back to file name input
                        return self.select_input_source(raw_text_dir, allow_back=allow_back)

                    try:
                        idx = int(selected_index) - 1
                        if 0 <= idx < len(file_candidates):
                            files.append(file_candidates[idx])
                            self.print_success(f"Selected: {files[0].name}")
                            break
                        else:
                            self.print_error(f"Please enter a number between 1 and {len(file_candidates)}.")
                    except ValueError:
                        self.print_error("Invalid input. Please enter a number.")

        elif mode == "folder":
            # Get all .txt files, filtering out auxiliary files
            files = [f for f in raw_text_dir.rglob("*.txt")
                     if not (f.name.endswith("_line_ranges.txt") or
                             f.name.endswith("_context.txt"))]

            if not files:
                self.print_error(f"No .txt files found in {raw_text_dir}")
                sys.exit(1)

            self.print_success(f"Found {len(files)} text files to process")

        return files

    def display_processing_summary(
        self,
        files: List[Path],
        selected_schema_name: str,
        global_chunking_method: Optional[str],
        use_batch: bool,
        context_settings: Dict[str, Any]
    ) -> bool:
        """
        Display a summary of the selected processing options and ask for confirmation.

        :param files: List of selected file paths
        :param selected_schema_name: Name of the selected schema
        :param global_chunking_method: Selected chunking method
        :param use_batch: Whether batch processing is enabled
        :param context_settings: Context settings dictionary
        :return: True if user confirms, False otherwise
        """
        self.print_section_header("Processing Summary")

        file_type = "file" if len(files) == 1 else "files"
        self.console_print(f"\n{self.BOLD}Ready to process {len(files)} {file_type} with the following settings:{self.RESET}\n")
        
        self.console_print(f"  📋 Schema: {self.BOLD}{selected_schema_name}{self.RESET}")

        chunking_display = {
            "auto": "Automatic chunking",
            "auto-adjust": "Auto-adjusted chunking",
            "line_ranges.txt": "Manual chunking (using line_ranges.txt files)",
            "adjust-line-ranges": "Adjust & use line ranges with semantic boundary detection",
            "per-file": "Per-file chunking selection"
        }

        self.console_print(
            f"  ✂️  Chunking: {chunking_display.get(global_chunking_method, 'Per-file selection')}")

        processing_mode = "Batch (asynchronous)" if use_batch else "Synchronous (real-time)"
        self.console_print(f"  ⚙️  Processing: {processing_mode}")

        if context_settings.get("use_additional_context", False):
            context_source = "Default schema-specific" if context_settings.get(
                "use_default_context", False) else "File-specific"
            self.console_print(f"  📝 Context: {context_source}")
        else:
            self.console_print("  📝 Context: None")

        # Show files
        self.console_print(f"\n{self.BOLD}Selected files:{self.RESET}")
        for i, item in enumerate(files[:5], 1):
            self.console_print(f"  {i}. {item.name}")

        if len(files) > 5:
            self.console_print(f"  ... and {len(files) - 5} more")

        self.console_print("")  # Empty line
        return self.confirm("Proceed with processing?", default=False)

    def ask_file_chunking_method(self, file_name: str) -> str:
        """
        Prompt the user to select a chunking method for the given file.

        :param file_name: Name of the file to prompt for
        :return: Selected chunking method
        """
        self.print_subsection_header(f"Chunking Method for '{file_name}'")
        
        options = [
            ("auto", "Automatic - Split based on token limits"),
            ("auto-adjust", "Interactive - View and manually adjust chunk boundaries"),
            ("line_ranges.txt", "Predefined - Use saved boundaries from line range file")
        ]

        result = self.select_option(
            "Select chunking method:",
            options,
            allow_back=False,
            allow_quit=False
        )
        
        return result or "auto"

    def display_batch_summary(self, batches: List[Any]) -> None:
        """
        Display a summary of batch job statuses.

        :param batches: List of batch objects from OpenAI API
        """
        # Count batches by status
        status_counts = {}
        in_progress_batches = []

        for batch in batches:
            if isinstance(batch, dict):
                status = str(batch.get("status", "unknown")).lower()
                batch_id = batch.get("id", "unknown")
                created_time = batch.get("created_at") or batch.get("created", "Unknown")
            else:
                status = str(getattr(batch, "status", "unknown")).lower()
                batch_id = getattr(batch, "id", "unknown")
                created_time = getattr(batch, "created_at", getattr(batch, "created", "Unknown"))
            
            status_counts[status] = status_counts.get(status, 0) + 1

            # Keep track of non-terminal batches for detailed display
            if status not in {"completed", "expired", "cancelled", "failed"}:
                in_progress_batches.append((batch_id, status, created_time))

        # Display summary
        self.print_section_header("Batch Jobs Summary")
        self.console_print(f"\n{self.BOLD}Total batches: {len(batches)}{self.RESET}\n")

        # Display counts by status
        status_icons = {
            "completed": "✓",
            "failed": "✗",
            "cancelled": "⊗",
            "expired": "⏱",
            "validating": "◷",
            "in_progress": "▶",
            "finalizing": "◉"
        }
        
        for status, count in sorted(status_counts.items()):
            icon = status_icons.get(status, "•")
            self.console_print(f"  {icon} {status.capitalize()}: {count}")

        # Display in-progress batches if any
        if in_progress_batches:
            self.print_subsection_header("Batches In Progress")
            for batch_id, status, created_time in in_progress_batches:
                self.console_print(f"  • {batch_id} | {status} | Created: {created_time}")

    def display_batch_processing_progress(
        self,
        temp_file: Path,
        batch_ids: List[str],
        completed_batches: int,
        missing_batches: int,
        failed_batches: List[Tuple[Dict[str, Any], str]],
    ) -> None:
        """Print a progress summary for a temp batch file."""
        total_batches = len(batch_ids)
        self.print_info(f"{temp_file.name}: {completed_batches}/{total_batches} batches completed")
        
        if missing_batches:
            self.print_warning(f"{missing_batches} batch ID(s) missing from OpenAI responses")
        
        if failed_batches:
            self.print_warning("Failed or terminal batches detected:")
            for track, status in failed_batches:
                bid = track.get("batch_id", "unknown")
                self.console_print(f"    • {bid} | Status: {status}")

    def display_batch_operation_result(
        self,
        batch_id: str,
        operation: str,
        success: bool,
        message: str = None
    ) -> None:
        """
        Display the result of a batch operation.

        :param batch_id: The ID of the batch
        :param operation: The operation performed (e.g., "cancel", "process")
        :param success: Whether the operation was successful
        :param message: Optional message to display
        """
        if success:
            result = f"Successfully {operation}ed batch {batch_id}"
            if message:
                result += f": {message}"
            self.print_success(result)
        else:
            result = f"Failed to {operation} batch {batch_id}"
            if message:
                result += f": {message}"
            self.print_error(result)

        self.log(f"Batch operation: {operation} on {batch_id} - {'success' if success else 'failure'}")
