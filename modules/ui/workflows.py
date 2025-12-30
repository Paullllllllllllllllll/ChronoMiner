"""Enhanced workflow UI components using the new prompt system.

This module provides workflow-specific UI components that use the improved
prompting utilities for a consistent and navigable user experience.

Synchronized with ChronoTranscriber's UI system for consistent UX across projects.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from modules.ui.prompts import (
    NavigationAction,
    PromptResult,
    prompt_select,
    prompt_yes_no,
    prompt_multiselect,
    prompt_text,
    print_header,
    print_separator,
    print_info,
    print_success,
    print_warning,
    print_error,
    ui_print,
    PromptStyle,
)
from modules.core.logger import setup_logger

logger = setup_logger(__name__)


class WorkflowUI:
    """Enhanced workflow UI with navigation support for ChronoMiner."""
    
    @staticmethod
    def display_welcome() -> None:
        """Display welcome banner."""
        print_header(
            "CHRONO MINER",
            "Structured Data Extraction Tool"
        )
        ui_print("  Extract structured data from historical documents using", PromptStyle.INFO)
        ui_print("  advanced AI models and customizable schemas.\n", PromptStyle.INFO)
    
    @staticmethod
    def display_step(step_number: int, total_steps: int, title: str) -> None:
        """Display a step header in a multi-step workflow.
        
        Args:
            step_number: Current step number
            total_steps: Total number of steps
            title: Step title
        """
        ui_print(f"\n  Step {step_number}/{total_steps}: {title}", PromptStyle.HEADER)
        print_separator(PromptStyle.SINGLE_LINE, 60)
    
    @staticmethod
    def display_summary(title: str, items: Dict[str, Any]) -> None:
        """Display a configuration summary.
        
        Args:
            title: Summary title
            items: Dictionary of configuration items to display
        """
        print_header(title, "")
        for key, value in items.items():
            if value is not None:
                ui_print(f"  {key}: {value}", PromptStyle.INFO)
    
    @staticmethod
    def confirm_and_proceed(message: str = "Proceed with these settings?") -> bool:
        """Ask for confirmation to proceed.
        
        Args:
            message: Confirmation message
            
        Returns:
            True if user confirms, False otherwise
        """
        result = prompt_yes_no(message, default=True, allow_back=True)
        if result.action == NavigationAction.CONTINUE:
            return result.value
        return False
    
    @staticmethod
    def select_schema(
        schema_options: List[Tuple[str, str]],
        allow_back: bool = True
    ) -> Optional[str]:
        """Prompt user to select a schema.
        
        Args:
            schema_options: List of (schema_name, display_name) tuples
            allow_back: Whether to allow back navigation
            
        Returns:
            Selected schema name or None if cancelled
        """
        if not schema_options:
            print_error("No schemas available.")
            return None
        
        result = prompt_select(
            "Select a schema for extraction:",
            schema_options,
            allow_back=allow_back
        )
        
        if result.action == NavigationAction.CONTINUE:
            logger.info(f"User selected schema: {result.value}")
            return result.value
        
        return None
    
    @staticmethod
    def select_files(
        available_files: List[Path],
        allow_back: bool = True
    ) -> Optional[List[Path]]:
        """Prompt user to select files for processing.
        
        Args:
            available_files: List of available file paths
            allow_back: Whether to allow back navigation
            
        Returns:
            List of selected file paths or None if cancelled
        """
        if not available_files:
            print_warning("No files available for selection.")
            return None
        
        # Build options list
        options = [(str(f), f.name) for f in available_files]
        
        result = prompt_multiselect(
            "Select files to process:",
            options,
            allow_all=True,
            allow_back=allow_back
        )
        
        if result.action == NavigationAction.CONTINUE:
            selected_paths = [Path(p) for p in result.value]
            logger.info(f"User selected {len(selected_paths)} file(s)")
            return selected_paths
        
        return None
    
    @staticmethod
    def select_processing_mode(allow_back: bool = True) -> Optional[str]:
        """Prompt user to select processing mode.
        
        Args:
            allow_back: Whether to allow back navigation
            
        Returns:
            Selected mode ('sync' or 'batch') or None if cancelled
        """
        options = [
            ("sync", "Synchronous — Process files immediately, get results right away"),
            ("batch", "Batch — Queue files for async processing (lower cost, longer wait)"),
        ]
        
        result = prompt_select(
            "Select processing mode:",
            options,
            allow_back=allow_back
        )
        
        if result.action == NavigationAction.CONTINUE:
            logger.info(f"User selected processing mode: {result.value}")
            return result.value
        
        return None
    
    @staticmethod
    def get_input_path(
        prompt_message: str = "Enter the path to the input file or directory:",
        allow_back: bool = True
    ) -> Optional[Path]:
        """Prompt user for an input path.
        
        Args:
            prompt_message: The prompt message to display
            allow_back: Whether to allow back navigation
            
        Returns:
            Valid input path or None if cancelled
        """
        while True:
            result = prompt_text(
                prompt_message,
                allow_back=allow_back
            )
            
            if result.action != NavigationAction.CONTINUE:
                return None
            
            path = Path(result.value).resolve()
            if path.exists():
                return path
            
            print_error(f"Path does not exist: {path}")
    
    @staticmethod
    def get_output_path(
        prompt_message: str = "Enter the output directory path:",
        allow_back: bool = True,
        create_if_missing: bool = True
    ) -> Optional[Path]:
        """Prompt user for an output path.
        
        Args:
            prompt_message: The prompt message to display
            allow_back: Whether to allow back navigation
            create_if_missing: Whether to create the directory if it doesn't exist
            
        Returns:
            Output path or None if cancelled
        """
        result = prompt_text(
            prompt_message,
            allow_back=allow_back
        )
        
        if result.action != NavigationAction.CONTINUE:
            return None
        
        path = Path(result.value).resolve()
        
        if create_if_missing and not path.exists():
            try:
                path.mkdir(parents=True, exist_ok=True)
                print_success(f"Created output directory: {path}")
            except Exception as e:
                print_error(f"Failed to create directory: {e}")
                return None
        
        return path
    
    @staticmethod
    def display_progress(current: int, total: int, item_name: str = "item") -> None:
        """Display a progress indicator.
        
        Args:
            current: Current item number
            total: Total number of items
            item_name: Name of the item being processed
        """
        percentage = (current / total) * 100 if total > 0 else 0
        ui_print(
            f"  Processing {item_name} {current}/{total} ({percentage:.1f}%)",
            PromptStyle.INFO
        )
    
    @staticmethod
    def display_completion(
        success_count: int,
        total_count: int,
        error_count: int = 0
    ) -> None:
        """Display completion summary.
        
        Args:
            success_count: Number of successful items
            total_count: Total number of items
            error_count: Number of failed items
        """
        print_separator()
        if error_count == 0:
            print_success(f"Processing complete! {success_count}/{total_count} items processed successfully.")
        else:
            print_warning(
                f"Processing complete with errors: {success_count} succeeded, {error_count} failed "
                f"(out of {total_count} total)"
            )
