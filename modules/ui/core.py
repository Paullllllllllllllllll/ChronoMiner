# modules/ui/core.py

"""
User Interface Module for ChronoMiner

Provides a consistent, user-friendly interface for all interactive operations.
Separates user-facing prompts from detailed logging.

Refactored to use the modular prompts system for Windows compatibility
and consistency with ChronoTranscriber.
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import logging

# Import from modular prompts system
from modules.ui.prompts import (
    PromptStyle,
    ui_print,
    ui_input,
    print_header,
    print_separator,
    print_info as _print_info,
    print_success as _print_success,
    print_warning as _print_warning,
    print_error as _print_error,
    prompt_select,
    prompt_yes_no,
    prompt_text,
    prompt_multiselect,
    NavigationAction,
    PromptResult,
)

logger = logging.getLogger(__name__)


class UserInterface:
    """
    Handles all user interaction for ChronoMiner with consistent formatting and navigation.
    
    Key Features:
    - ASCII-safe characters for Windows compatibility
    - Delegates to modular prompts system
    - Consistent exit/back navigation options
    - Separation of user prompts from detailed logging
    """

    # ASCII-safe visual styling constants (Windows compatible)
    HORIZONTAL_LINE = "-" * 80
    DOUBLE_LINE = "=" * 80
    
    # Color codes - delegated to PromptStyle
    RESET = PromptStyle.RESET
    BOLD = "\033[1m"
    DIM = PromptStyle.DIM
    SUCCESS = PromptStyle.SUCCESS
    WARNING = PromptStyle.WARNING
    ERROR = PromptStyle.ERROR
    INFO = PromptStyle.INFO
    PROMPT = PromptStyle.PROMPT

    def __init__(self, logger: Optional[logging.Logger] = None, use_colors: bool = True) -> None:
        """
        Initialize the user interface.

        :param logger: Optional logger instance for detailed logging separate from user output
        :param use_colors: Whether to use ANSI color codes (default True with colorama)
        """
        self.logger = logger
        self.use_colors = use_colors
        if not use_colors:
            # Disable colors
            self.RESET = self.BOLD = self.DIM = ""
            self.SUCCESS = self.WARNING = self.ERROR = self.INFO = self.PROMPT = ""

    def log(self, message: str, level: str = "info") -> None:
        """
        Log a message to the logger without printing to console.
        
        :param message: Message to log
        :param level: Log level (debug, info, warning, error)
        """
        if self.logger:
            log_method = getattr(self.logger, level.lower(), self.logger.info)
            log_method(message)

    def console_print(self, message: str, log_also: bool = False) -> None:
        """
        Print to console for user interaction.
        
        :param message: Message to display
        :param log_also: Whether to also write to log file
        """
        ui_print(message)
        if log_also and self.logger:
            self.logger.info(message)

    def print_success(self, message: str) -> None:
        """Print a success message with visual formatting."""
        _print_success(message)
        self.log(f"SUCCESS: {message}", "info")

    def print_error(self, message: str) -> None:
        """Print an error message with visual formatting."""
        _print_error(message)
        self.log(f"ERROR: {message}", "error")

    def print_warning(self, message: str) -> None:
        """Print a warning message with visual formatting."""
        _print_warning(message)
        self.log(f"WARNING: {message}", "warning")

    def print_info(self, message: str) -> None:
        """Print an info message with visual formatting."""
        _print_info(message)
        self.log(f"INFO: {message}", "info")

    def print_section_header(self, title: str) -> None:
        """Print a prominent section header."""
        print_header(title, "")

    def print_subsection_header(self, title: str) -> None:
        """Print a subsection header."""
        ui_print(f"\n{self.HORIZONTAL_LINE}", PromptStyle.DIM)
        ui_print(f"  {title}", PromptStyle.HEADER)
        ui_print(self.HORIZONTAL_LINE, PromptStyle.DIM)

    def display_banner(self) -> None:
        """Display a welcome banner with information about the application."""
        print_header(
            "ChronoMiner - Structured Data Extraction Tool",
            "Extract structured data from historical documents using advanced AI models"
        )

    def get_input(self, prompt: str, allow_back: bool = False, allow_quit: bool = True) -> Optional[str]:
        """
        Get user input with consistent navigation options.
        
        :param prompt: The prompt to display
        :param allow_back: Whether to allow 'b' to go back
        :param allow_quit: Whether to allow 'q' to quit
        :return: User input or None if back/quit selected
        """
        # Use the modular prompt_text function
        result = prompt_text(
            prompt,
            allow_empty=True,
            allow_back=allow_back
        )
        
        if result.action == NavigationAction.BACK:
            return None
        if result.action == NavigationAction.QUIT:
            sys.exit(0)
        
        return result.value

    def confirm(self, message: str, default: bool = False) -> bool:
        """
        Ask for yes/no confirmation.
        
        :param message: Confirmation message
        :param default: Default value if user just presses Enter
        :return: True if confirmed, False otherwise
        """
        result = prompt_yes_no(message, default=default, allow_back=False)
        if result.action == NavigationAction.CONTINUE:
            return result.value
        return default

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
        result = prompt_select(prompt, options, allow_back=allow_back)
        
        if result.action == NavigationAction.BACK:
            return None
        if result.action == NavigationAction.CONTINUE:
            self.log(f"User selected: {result.value}")
            return result.value
        
        return None

    def select_schema(self, schema_manager: Any, allow_back: bool = False) -> Optional[Tuple[Dict[str, Any], str]]:
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
            ("auto-adjust", "Manual adjustment - Automatic chunking with real-time manual boundary refinement"),
            ("line_ranges.txt", "Use existing line ranges - Process with pre-defined line range files"),
            ("adjust-line-ranges", "AI-assisted adjustment - Auto-generate line ranges, refine with AI boundary detection, then process"),
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

    def ask_chunk_slice(self, allow_back: bool = False):
        """
        Prompt user to optionally limit processing to first/last N chunks.

        :param allow_back: Whether to allow going back to previous step
        :return: A ChunkSlice instance, or None on back navigation.
                 Returns ChunkSlice() (both fields None) for 'all chunks'.
        """
        from modules.core.chunking_service import ChunkSlice

        self.print_section_header("Chunk Range")

        range_options = [
            ("all", "Process all chunks (default)"),
            ("first", "Process only the first N chunks"),
            ("last", "Process only the last N chunks"),
        ]

        choice = self.select_option(
            "Would you like to limit which chunks are processed?",
            range_options,
            allow_back=allow_back,
            allow_quit=True,
        )

        if choice is None:
            return None  # back navigation

        if choice == "all":
            return ChunkSlice()

        label = "first" if choice == "first" else "last"
        while True:
            n_input = self.get_input(
                f"How many {label} chunks should be processed?",
                allow_back=True,
                allow_quit=True,
            )
            if n_input is None:
                # User went back — re-show the range options
                return self.ask_chunk_slice(allow_back=allow_back)
            try:
                n = int(n_input)
                if n < 1:
                    self.print_error("Please enter a positive integer (>= 1).")
                    continue
                if choice == "first":
                    return ChunkSlice(first_n=n)
                else:
                    return ChunkSlice(last_n=n)
            except ValueError:
                self.print_error("Invalid number. Please enter a positive integer.")

    def select_input_source(self, raw_text_dir: Path, allow_back: bool = False) -> Optional[List[Path]]:
        """
        Guide user through selecting input source (single file, multiple files, or folder).

        :param raw_text_dir: Base directory for input files
        :param allow_back: Whether to allow going back to previous step
        :return: List of selected file paths or None if back selected
        """
        while True:  # Allow retry on errors
            self.print_section_header("Input Selection")

            mode_options = [
                ("single", "Process a single file"),
                ("multi", "Process selected files from a folder"),
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
                while True:  # Inner loop for single file selection
                    self.print_info("Enter the filename to process (with or without .txt extension)")
                    self.console_print("  - Enter the base text filename")
                    self.console_print("  - Or enter the line range filename ending in '_line_ranges.txt'")
                    
                    file_input = self.get_input("Filename", allow_back=True, allow_quit=True)
                    if not file_input:
                        # User went back - break to mode selection
                        break
                    
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
                        self.print_info("Please try again or press 'b' to go back.")
                        continue  # Retry filename input
                    elif len(file_candidates) == 1:
                        files.append(file_candidates[0])
                        self.print_success(f"Selected: {files[0].name}")
                        return files
                    else:
                        self.print_warning(f"Found {len(file_candidates)} matching files:")
                        self.console_print(self.HORIZONTAL_LINE)

                        file_options = [(str(i), str(f.relative_to(raw_text_dir))) for i, f in enumerate(file_candidates)]
                        
                        for idx, f in enumerate(file_candidates, 1):
                            self.console_print(f"  {idx}. {f.relative_to(raw_text_dir)}")

                        while True:
                            selected_index = self.get_input("Select file by number", allow_back=True, allow_quit=True)
                            
                            if not selected_index:
                                # User went back - break to filename input
                                break

                            try:
                                idx = int(selected_index) - 1
                                if 0 <= idx < len(file_candidates):
                                    files.append(file_candidates[idx])
                                    self.print_success(f"Selected: {files[0].name}")
                                    return files
                                else:
                                    self.print_error(f"Please enter a number between 1 and {len(file_candidates)}.")
                            except ValueError:
                                self.print_error("Invalid input. Please enter a number.")
                        
                        # If we broke out of the number selection loop, retry filename input
                        continue
                
                # If we broke out of the filename loop, return to mode selection
                continue

            elif mode == "multi":
                # Get all .txt files, filtering out auxiliary files
                all_files = [f for f in raw_text_dir.rglob("*.txt")
                             if not (f.name.endswith("_line_ranges.txt") or
                                     f.name.endswith("_context.txt"))]
                
                if not all_files:
                    self.print_error(f"No .txt files found in {raw_text_dir}")
                    self.print_info("Please check the directory or go back to select a different option.")
                    continue
                
                # Sort files for consistent ordering
                all_files = sorted(all_files)
                
                self.print_info(f"Found {len(all_files)} text files in {raw_text_dir.name}")
                self.console_print(self.HORIZONTAL_LINE)
                
                # Display all files with numbers
                for idx, f in enumerate(all_files, 1):
                    self.console_print(f"  {idx}. {f.relative_to(raw_text_dir)}")
                
                self.console_print("")  # Empty line for spacing
                self.print_info("Enter file numbers to process (comma-separated, e.g., '1,3,5' or '1-3,5')")
                
                while True:
                    selection = self.get_input("File selection", allow_back=True, allow_quit=True)
                    
                    if not selection:
                        # User went back to mode selection
                        break
                    
                    try:
                        # Parse comma-separated indices and ranges
                        selected_indices: set[int] = set()
                        parts = selection.split(',')
                        
                        for part in parts:
                            part = part.strip()
                            if '-' in part:
                                # Range like "1-3"
                                start, end = part.split('-', 1)
                                start_idx = int(start.strip())
                                end_idx = int(end.strip())
                                if start_idx < 1 or end_idx > len(all_files) or start_idx > end_idx:
                                    raise ValueError(f"Invalid range: {part}")
                                selected_indices.update(range(start_idx, end_idx + 1))
                            else:
                                # Single index
                                idx = int(part)
                                if idx < 1 or idx > len(all_files):
                                    raise ValueError(f"Index {idx} out of range")
                                selected_indices.add(idx)
                        
                        if not selected_indices:
                            self.print_error("No files selected. Please enter at least one file number.")
                            continue
                        
                        # Convert to file paths
                        files = [all_files[idx - 1] for idx in sorted(selected_indices)]
                        
                        # Confirm selection
                        self.print_success(f"Selected {len(files)} file(s):")
                        for f in files:
                            self.console_print(f"  - {f.name}")
                        
                        return files
                        
                    except ValueError as e:
                        self.print_error(f"Invalid selection: {e}")
                        self.print_info(f"Please enter numbers between 1 and {len(all_files)}, comma-separated")
                
                # If we broke out of selection loop, return to mode selection
                continue

            elif mode == "folder":
                # Get all .txt files, filtering out auxiliary files
                files = [f for f in raw_text_dir.rglob("*.txt")
                         if not (f.name.endswith("_line_ranges.txt") or
                                 f.name.endswith("_context.txt"))]

                if not files:
                    self.print_error(f"No .txt files found in {raw_text_dir}")
                    self.print_info("Please check the directory or go back to select a different option.")
                    continue

                self.print_success(f"Found {len(files)} text files to process")
                return files

    def display_processing_summary(
        self,
        files: List[Path],
        selected_schema_name: str,
        global_chunking_method: Optional[str],
        use_batch: bool,
        model_config: Optional[Dict[str, Any]] = None,
        paths_config: Optional[Dict[str, Any]] = None,
        concurrency_config: Optional[Dict[str, Any]] = None,
        chunk_slice: Any = None,
        context_mode: Optional[str] = None,
        existing_output_count: Optional[int] = None,
    ) -> bool:
        """
        Display a detailed summary of the selected processing options and ask for confirmation.

        :param files: List of selected file paths
        :param selected_schema_name: Name of the selected schema
        :param global_chunking_method: Selected chunking method
        :param use_batch: Whether batch processing is enabled
        :param model_config: Model configuration dictionary
        :param paths_config: Paths configuration dictionary
        :param concurrency_config: Concurrency configuration dictionary
        :param chunk_slice: Optional ChunkSlice for first/last N chunks
        :param context_mode: Context mode string (e.g. 'auto', 'none', or a path)
        :param existing_output_count: Number of output files that already exist (CM-10)
        :return: True if user confirms, False otherwise
        """
        self.print_section_header("Processing Summary")
        self.console_print("  Review your selections before processing")
        self.console_print(self.HORIZONTAL_LINE)

        file_type = "file" if len(files) == 1 else "files"
        self.console_print(f"\n  Ready to process {self.BOLD}{len(files)}{self.RESET} text {file_type}\n")
        
        # === Processing Configuration ===
        self.console_print(f"  {self.BOLD}Processing Configuration:{self.RESET}")
        self.console_print(self.HORIZONTAL_LINE)
        
        self.console_print(f"    - Schema: {selected_schema_name}")
        
        chunking_display = {
            "auto": "Automatic chunking",
            "auto-adjust": "Manual real-time adjustment",
            "line_ranges.txt": "Manual chunking (using line_ranges.txt files)",
            "adjust-line-ranges": "AI-assisted line range adjustment",
            "per-file": "Per-file chunking selection"
        }
        self.console_print(f"    - Chunking method: {chunking_display.get(global_chunking_method or '', 'Per-file selection')}")

        processing_mode = "Batch (asynchronous)" if use_batch else "Synchronous (real-time)"
        self.console_print(f"    - Processing mode: {processing_mode}")

        # Chunk slice
        if chunk_slice is not None:
            first_n = getattr(chunk_slice, "first_n", None)
            last_n = getattr(chunk_slice, "last_n", None)
            if first_n is not None:
                self.console_print(f"    - Chunk range: First {first_n} chunks only")
            elif last_n is not None:
                self.console_print(f"    - Chunk range: Last {last_n} chunks only")

        # Context display (CM-8)
        if context_mode is None or context_mode == "auto":
            context_display = "Automatic (hierarchical resolution)"
        elif context_mode == "none":
            context_display = "Disabled"
        else:
            context_display = f"Manual: {context_mode}"
        self.console_print(f"    {self.DIM}- Context: {context_display}{self.RESET}")
        
        self.console_print(self.HORIZONTAL_LINE)
        
        # === Model Configuration ===
        self.console_print(f"\n  {self.BOLD}Model Configuration:{self.RESET}")
        self.console_print(self.HORIZONTAL_LINE)
        
        if model_config:
            tm = model_config.get("transcription_model", {})
            provider = tm.get("provider", "auto-detect")
            model_name = tm.get("name", "unknown")
            self.console_print(f"    - Provider: {provider.upper() if provider != 'auto-detect' else 'Auto-detect'}")
            self.console_print(f"    - Model: {model_name}")
            
            # Show key model parameters (dimmed)
            temperature = tm.get("temperature")
            max_tokens = tm.get("max_output_tokens") or tm.get("max_tokens", 32000)
            if temperature is not None:
                self.console_print(f"    {self.DIM}- Temperature: {temperature}{self.RESET}")
            self.console_print(f"    {self.DIM}- Max output tokens: {max_tokens:,}{self.RESET}")
            
            # Show reasoning effort if configured
            reasoning = tm.get("reasoning", {})
            if reasoning.get("effort"):
                self.console_print(f"    {self.DIM}- Reasoning effort: {reasoning['effort']}{self.RESET}")
            
            # Show text verbosity if present (GPT-5 specific)
            text_config = tm.get("text", {})
            if text_config.get("verbosity"):
                self.console_print(f"    {self.DIM}- Text verbosity: {text_config['verbosity']}{self.RESET}")
        
        self.console_print(self.HORIZONTAL_LINE)
        
        # === Concurrency Configuration ===
        if concurrency_config:
            self.console_print(f"\n  {self.BOLD}Concurrency Configuration:{self.RESET}")
            self.console_print(self.HORIZONTAL_LINE)
            
            # API request concurrency (CM-6: correct key path)
            extraction_cfg = concurrency_config.get("concurrency", {}).get("extraction", {})
            trans_concurrency = extraction_cfg.get("concurrency_limit", 5)
            trans_service_tier = extraction_cfg.get("service_tier", "default")
            self.console_print(f"    - API requests: {trans_concurrency} concurrent")
            self.console_print(f"    {self.DIM}- Service tier: {trans_service_tier}{self.RESET}")

            # Retry configuration
            retry_config = extraction_cfg.get("retry", {})
            max_attempts = retry_config.get("attempts", 5)
            self.console_print(f"    {self.DIM}- Max retry attempts: {max_attempts}{self.RESET}")
            
            self.console_print(self.HORIZONTAL_LINE)
        
        # === Output Location ===
        self.console_print(f"\n  {self.BOLD}Output Location:{self.RESET}")
        self.console_print(self.HORIZONTAL_LINE)
        
        if paths_config:
            use_input_as_output = paths_config.get('general', {}).get('input_paths_is_output_path', False)
            if use_input_as_output:
                self.console_print("    - Output: Same directory as input files")
            else:
                # Show configured output directory for this schema
                schemas_paths = paths_config.get('schemas_paths', {})
                schema_config = schemas_paths.get(selected_schema_name, {})
                output_dir = schema_config.get('output', 'configured output directory')
                self.console_print(f"    - Output directory: {output_dir}")
                
                # Show output formats if configured
                output_formats = []
                if schema_config.get('csv_output', False):
                    output_formats.append('CSV')
                if schema_config.get('docx_output', False):
                    output_formats.append('DOCX')
                if schema_config.get('txt_output', False):
                    output_formats.append('TXT')
                if output_formats:
                    self.console_print(f"    - Output formats: {', '.join(output_formats)}")
        else:
            self.console_print("    - Output: Configured output directory")
        
        self.console_print(self.HORIZONTAL_LINE)

        # === Selected Files ===
        self.console_print(f"\n  {self.BOLD}Selected Files (first 5 shown):{self.RESET}")
        for i, item in enumerate(files[:5], 1):
            self.console_print(f"    {self.DIM}{i}. {item.name}{self.RESET}")

        if len(files) > 5:
            self.console_print(f"    {self.DIM}... and {len(files) - 5} more{self.RESET}")

        # Pre-check existing output files (CM-10)
        if existing_output_count is not None and existing_output_count > 0:
            self.console_print("")
            self.console_print(
                f"    {self.WARNING}Warning: {existing_output_count}/{len(files)} output file(s) already exist "
                f"and will be overwritten.{self.RESET}"
            )

        self.console_print("")  # Empty line
        return self.confirm("Proceed with processing?", default=True)
    
    def ask_context_selection(self, allow_back: bool = False) -> Optional[Dict[str, Any]]:
        """
        Ask user to select a context mode for extraction.

        :param allow_back: Whether to allow going back to the previous step
        :return: Dict with 'mode' ('auto', 'none', or 'manual') and optional 'path', or None on back
        """
        self.print_section_header("Context Selection")

        context_options = [
            ("auto", "Automatic - Use hierarchical context resolution (file/folder/global fallback)"),
            ("manual", "Manual - Enter a specific context file path"),
            ("none", "No context - Disable context for this run"),
        ]

        mode = self.select_option(
            "Select context mode:",
            context_options,
            allow_back=allow_back,
        )

        if mode is None:
            return None

        if mode == "auto":
            return {"mode": "auto", "path": None}

        if mode == "none":
            return {"mode": "none", "path": None}

        # Manual path entry
        while True:
            path_input = self.get_input(
                "Enter path to context file:",
                allow_back=True,
            )
            if path_input is None:
                return self.ask_context_selection(allow_back=allow_back)

            from pathlib import Path as _Path
            context_path = _Path(path_input)
            if context_path.exists():
                return {"mode": "manual", "path": context_path}
            else:
                self.print_error(f"File not found: {path_input}")
                self.print_info("Please enter a valid file path or press 'b' to go back.")

    def display_completion_summary(
        self,
        processed_count: int,
        failed_count: int,
        use_batch: bool,
        duration_seconds: float = 0.0,
        paths_config: Optional[Dict[str, Any]] = None,
        selected_schema_name: Optional[str] = None,
    ) -> None:
        """
        Display a detailed completion summary after processing.

        :param processed_count: Number of successfully processed files
        :param failed_count: Number of failed files
        :param use_batch: Whether batch processing was used
        :param duration_seconds: Total processing duration in seconds
        :param paths_config: Paths configuration dictionary
        :param selected_schema_name: Name of the schema used
        """
        self.print_section_header("Processing Complete")
        
        total_count = processed_count + failed_count
        
        # === Results Section ===
        self.console_print(f"  {self.BOLD}Results:{self.RESET}")
        self.console_print(self.HORIZONTAL_LINE)
        
        if use_batch:
            self.print_success("Batch processing jobs have been submitted!")
            self.console_print(f"    - Jobs submitted: {total_count}")
        else:
            if failed_count == 0 and processed_count > 0:
                self.print_success(f"All {processed_count} file(s) processed successfully!")
            elif processed_count > 0:
                self.console_print(f"    - Processed: {processed_count}/{total_count} file(s)")
                if failed_count > 0:
                    self.print_warning(f"    - Failed: {failed_count} file(s)")
            else:
                self.print_warning("    - No files were processed.")
        
        # Duration
        if duration_seconds > 0:
            if duration_seconds >= 3600:
                hours = duration_seconds / 3600
                self.console_print(f"    - Duration: {hours:.1f} hours")
            elif duration_seconds >= 60:
                minutes = duration_seconds / 60
                self.console_print(f"    - Duration: {minutes:.1f} minutes")
            else:
                self.console_print(f"    - Duration: {duration_seconds:.1f} seconds")
        
        self.console_print(self.HORIZONTAL_LINE)
        
        # === Output Location ===
        self.console_print(f"\n  {self.BOLD}Output:{self.RESET}")
        self.console_print(self.HORIZONTAL_LINE)
        
        if paths_config:
            use_input_as_output = paths_config.get('general', {}).get('input_paths_is_output_path', False)
            if use_input_as_output:
                self.console_print("    - Location: Same directory as input files")
            elif selected_schema_name:
                schemas_paths = paths_config.get('schemas_paths', {})
                schema_config = schemas_paths.get(selected_schema_name, {})
                output_dir = schema_config.get('output', 'configured output directory')
                self.console_print(f"    - Location: {output_dir}")
                
                # Show output formats
                output_formats = []
                if schema_config.get('csv_output', False):
                    output_formats.append('CSV')
                if schema_config.get('docx_output', False):
                    output_formats.append('DOCX')
                if schema_config.get('txt_output', False):
                    output_formats.append('TXT')
                if output_formats:
                    self.console_print(f"    - Formats: {', '.join(output_formats)}")
        
        self.console_print(self.HORIZONTAL_LINE)
        
        # === Next Steps (for batch mode) ===
        if use_batch:
            self.console_print(f"\n  {self.BOLD}Next steps:{self.RESET}")
            self.console_print(self.HORIZONTAL_LINE)
            self.console_print(f"    {self.DIM}- Check batch status: python main/check_batches.py{self.RESET}")
            self.console_print(f"    {self.DIM}- Cancel pending batches: python main/cancel_batches.py{self.RESET}")
            self.console_print(self.HORIZONTAL_LINE)
        
        self.console_print(f"\n  {self.BOLD}Thank you for using ChronoMiner!{self.RESET}\n")

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
        status_counts: dict[str, int] = {}
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
        # ASCII-safe status icons for Windows compatibility
        status_icons = {
            "completed": "[OK]",
            "failed": "[X]",
            "cancelled": "[X]",
            "expired": "[T]",
            "validating": "[.]",
            "in_progress": "[>]",
            "finalizing": "[O]"
        }
        
        for status, count in sorted(status_counts.items()):
            icon = status_icons.get(status, "-")
            self.console_print(f"  {icon} {status.capitalize()}: {count}")

        # Display in-progress batches if any
        if in_progress_batches:
            self.print_subsection_header("Batches In Progress")
            for batch_id, status, created_time in in_progress_batches:
                self.console_print(f"  - {batch_id} | {status} | Created: {created_time}")

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
                self.console_print(f"    - {bid} | Status: {status}")

    def display_batch_operation_result(
        self,
        batch_id: str,
        operation: str,
        success: bool,
        message: str | None = None
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
