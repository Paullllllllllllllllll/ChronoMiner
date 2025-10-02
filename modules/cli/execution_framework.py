"""
Unified execution framework for dual-mode (Interactive/CLI) scripts.

This module provides a base class and utilities to eliminate code duplication
across all main entry point scripts that support both interactive UI mode
and command-line argument mode.
"""

import sys
from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from modules.cli.mode_detector import should_use_interactive_mode
from modules.config.loader import ConfigLoader
from modules.core.logger import setup_logger
from modules.core.workflow_utils import load_core_resources
from modules.ui.core import UserInterface


class DualModeScript(ABC):
    """
    Base class for scripts that support both interactive and CLI modes.
    
    This class handles:
    - Mode detection (interactive vs CLI)
    - Configuration loading
    - UI initialization
    - Logger setup
    - Common error handling
    
    Subclasses must implement:
    - create_argument_parser(): Return configured ArgumentParser
    - run_interactive(): Execute interactive workflow
    - run_cli(): Execute CLI workflow
    """
    
    def __init__(self, script_name: str):
        """
        Initialize the dual-mode script.
        
        Args:
            script_name: Name of the script for logging purposes
        """
        self.script_name = script_name
        self.logger = setup_logger(script_name)
        self.config_loader: Optional[ConfigLoader] = None
        self.ui: Optional[UserInterface] = None
        self.is_interactive: bool = False
        
        # Configuration dictionaries (loaded on demand)
        self.paths_config: Dict[str, Any] = {}
        self.model_config: Dict[str, Any] = {}
        self.chunking_and_context_config: Dict[str, Any] = {}
        self.schemas_paths: Dict[str, Any] = {}
    
    def initialize_config(self) -> None:
        """Load all configuration resources."""
        (
            self.config_loader,
            self.paths_config,
            self.model_config,
            self.chunking_and_context_config,
            self.schemas_paths,
        ) = load_core_resources()
    
    def initialize_ui(self) -> None:
        """Initialize the user interface for interactive mode."""
        if not self.ui:
            self.ui = UserInterface(self.logger)
            self.ui.display_banner()
    
    @abstractmethod
    def create_argument_parser(self) -> ArgumentParser:
        """
        Create and configure the argument parser for CLI mode.
        
        Returns:
            Configured ArgumentParser instance
        """
        pass
    
    @abstractmethod
    def run_interactive(self) -> None:
        """
        Execute the interactive workflow with UI prompts.
        
        This method is called when the script runs in interactive mode.
        The UI instance is available as self.ui.
        """
        pass
    
    @abstractmethod
    def run_cli(self, args: Namespace) -> None:
        """
        Execute the CLI workflow with parsed arguments.
        
        This method is called when the script runs in CLI mode.
        
        Args:
            args: Parsed command-line arguments
        """
        pass
    
    def execute(self) -> None:
        """
        Main entry point that orchestrates mode detection and execution.
        
        This method:
        1. Loads configuration
        2. Detects execution mode (interactive vs CLI)
        3. Initializes appropriate interfaces
        4. Calls the appropriate run method
        5. Handles common error scenarios
        """
        try:
            # Load configuration
            self.initialize_config()
            
            # Determine execution mode
            self.is_interactive = should_use_interactive_mode(self.config_loader)
            
            if self.is_interactive:
                # Interactive mode
                self.initialize_ui()
                self.logger.info(f"Starting {self.script_name} (Interactive Mode)")
                self.run_interactive()
            else:
                # CLI mode
                self.logger.info(f"Starting {self.script_name} (CLI Mode)")
                parser = self.create_argument_parser()
                args = parser.parse_args()
                self.run_cli(args)
                
        except KeyboardInterrupt:
            self._handle_interrupt()
        except Exception as e:
            self._handle_error(e)
    
    def _handle_interrupt(self) -> None:
        """Handle keyboard interrupt gracefully."""
        if self.ui:
            self.ui.print_info("\nOperation cancelled by user.")
        else:
            print("\n[INFO] Operation cancelled by user.")
        self.logger.info(f"{self.script_name} cancelled by user")
        sys.exit(0)
    
    def _handle_error(self, error: Exception) -> None:
        """
        Handle unexpected errors gracefully.
        
        Args:
            error: The exception that was raised
        """
        error_msg = f"Unexpected error: {error}"
        if self.ui:
            self.ui.print_error(error_msg)
        else:
            print(f"[ERROR] {error_msg}")
        self.logger.error(f"{self.script_name} failed", exc_info=error)
        sys.exit(1)
    
    def print_or_log(self, message: str, level: str = "info") -> None:
        """
        Print message to UI if in interactive mode, otherwise log.
        
        Args:
            message: Message to display/log
            level: Log level (info, warning, error, success)
        """
        if self.ui:
            if level == "error":
                self.ui.print_error(message)
            elif level == "warning":
                self.ui.print_warning(message)
            elif level == "success":
                self.ui.print_success(message)
            else:
                self.ui.print_info(message)
        else:
            prefix = f"[{level.upper()}]"
            print(f"{prefix} {message}")
        
        # Always log
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(message)


def create_simple_dual_mode_executor(
    script_name: str,
    parser_factory: Callable[[], ArgumentParser],
    interactive_runner: Callable[[UserInterface, Dict[str, Any]], None],
    cli_runner: Callable[[Namespace, Dict[str, Any]], None]
) -> Callable[[], None]:
    """
    Factory function for creating simple dual-mode executors without subclassing.
    
    This is useful for simpler scripts that don't need the full DualModeScript class.
    
    Args:
        script_name: Name of the script
        parser_factory: Function that creates and returns ArgumentParser
        interactive_runner: Function to run in interactive mode
        cli_runner: Function to run in CLI mode
    
    Returns:
        A callable main function that can be executed
    
    Example:
        def create_parser():
            parser = ArgumentParser()
            parser.add_argument('--input', required=True)
            return parser
        
        def run_interactive(ui, config):
            ui.print_info("Running interactively...")
        
        def run_cli(args, config):
            print(f"Processing {args.input}")
        
        main = create_simple_dual_mode_executor(
            'my_script',
            create_parser,
            run_interactive,
            run_cli
        )
        
        if __name__ == '__main__':
            main()
    """
    class SimpleDualModeScript(DualModeScript):
        def create_argument_parser(self) -> ArgumentParser:
            return parser_factory()
        
        def run_interactive(self) -> None:
            config = {
                'paths': self.paths_config,
                'model': self.model_config,
                'chunking': self.chunking_and_context_config,
                'schemas': self.schemas_paths,
            }
            interactive_runner(self.ui, config)
        
        def run_cli(self, args: Namespace) -> None:
            config = {
                'paths': self.paths_config,
                'model': self.model_config,
                'chunking': self.chunking_and_context_config,
                'schemas': self.schemas_paths,
            }
            cli_runner(args, config)
    
    def main():
        script = SimpleDualModeScript(script_name)
        script.execute()
    
    return main
