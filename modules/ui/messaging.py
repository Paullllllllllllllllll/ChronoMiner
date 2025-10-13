# modules/ui/messaging.py

"""
Unified messaging interface for console output and logging.
Provides consistent formatting across interactive and CLI modes.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class MessageLevel:
    """Message severity levels."""
    INFO = "INFO"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ERROR = "ERROR"
    DEBUG = "DEBUG"


class MessagingAdapter:
    """
    Unified messaging adapter for console and log output.
    Provides consistent formatting and routing.
    """

    def __init__(self, ui_instance=None, use_colors: bool = True):
        """
        Initialize messaging adapter.

        :param ui_instance: Optional UserInterface instance
        :param use_colors: Whether to use colored output
        """
        self.ui = ui_instance
        self.use_colors = use_colors

    def info(self, message: str, log: bool = True) -> None:
        """
        Print info message.

        :param message: Message to print
        :param log: Whether to also log the message
        """
        if self.ui:
            self.ui.print_info(message)
        else:
            print(f"[INFO] {message}")
        
        if log:
            logger.info(message)

    def success(self, message: str, log: bool = True) -> None:
        """
        Print success message.

        :param message: Message to print
        :param log: Whether to also log the message
        """
        if self.ui:
            self.ui.print_success(message)
        else:
            print(f"[SUCCESS] {message}")
        
        if log:
            logger.info(f"SUCCESS: {message}")

    def warning(self, message: str, log: bool = True) -> None:
        """
        Print warning message.

        :param message: Message to print
        :param log: Whether to also log the message
        """
        if self.ui:
            self.ui.print_warning(message)
        else:
            print(f"[WARNING] {message}")
        
        if log:
            logger.warning(message)

    def error(self, message: str, log: bool = True, exc_info: Optional[Exception] = None) -> None:
        """
        Print error message.

        :param message: Message to print
        :param log: Whether to also log the message
        :param exc_info: Optional exception for detailed logging
        """
        if self.ui:
            self.ui.print_error(message)
        else:
            print(f"[ERROR] {message}")
        
        if log:
            if exc_info:
                logger.error(message, exc_info=exc_info)
            else:
                logger.error(message)

    def debug(self, message: str, log: bool = True) -> None:
        """
        Print debug message.

        :param message: Message to print
        :param log: Whether to also log the message
        """
        if log:
            logger.debug(message)

    def console_print(self, message: str) -> None:
        """
        Print message directly to console (for compatibility).

        :param message: Message to print
        """
        if self.ui and hasattr(self.ui, 'console_print'):
            self.ui.console_print(message)
        else:
            print(message)

    @classmethod
    def create_for_cli(cls) -> "MessagingAdapter":
        """Create messaging adapter for CLI mode."""
        return cls(ui_instance=None, use_colors=False)

    @classmethod
    def create_for_interactive(cls, ui_instance) -> "MessagingAdapter":
        """Create messaging adapter for interactive mode."""
        return cls(ui_instance=ui_instance, use_colors=True)


def create_messaging_adapter(ui_instance=None) -> MessagingAdapter:
    """
    Factory function to create appropriate messaging adapter.

    :param ui_instance: Optional UserInterface instance
    :return: MessagingAdapter instance
    """
    if ui_instance:
        return MessagingAdapter.create_for_interactive(ui_instance)
    else:
        return MessagingAdapter.create_for_cli()
