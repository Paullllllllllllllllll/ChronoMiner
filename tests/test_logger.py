"""Tests for the centralized logger configuration."""

from pathlib import Path
import logging

import pytest

from modules.core.logger import setup_logger, _resolve_logs_dir


class TestSetupLogger:
    """Tests for setup_logger function."""
    
    @pytest.mark.unit
    def test_returns_logger_instance(self):
        """Test that setup_logger returns a Logger instance."""
        logger = setup_logger("test_logger")
        
        assert isinstance(logger, logging.Logger)
    
    @pytest.mark.unit
    def test_logger_has_handlers(self):
        """Test that logger has handlers attached."""
        logger = setup_logger("test_with_handlers")
        
        # Should have at least one handler
        assert len(logger.handlers) > 0
    
    @pytest.mark.unit
    def test_same_name_returns_same_logger(self):
        """Test that same name returns the same logger instance."""
        logger1 = setup_logger("identical_name")
        logger2 = setup_logger("identical_name")
        
        assert logger1 is logger2
    
    @pytest.mark.unit
    def test_different_names_return_different_loggers(self):
        """Test that different names return different loggers."""
        logger1 = setup_logger("name_one")
        logger2 = setup_logger("name_two")
        
        assert logger1 is not logger2
    
    @pytest.mark.unit
    def test_logger_level_is_info(self):
        """Test that logger level is set to INFO."""
        logger = setup_logger("test_level")
        
        assert logger.level == logging.INFO


class TestResolveLogsDir:
    """Tests for _resolve_logs_dir function."""
    
    @pytest.mark.unit
    def test_returns_path_object(self):
        """Test that _resolve_logs_dir returns a Path object."""
        result = _resolve_logs_dir()
        
        assert isinstance(result, Path)
    
    @pytest.mark.unit
    def test_returns_logs_subdirectory(self):
        """Test that result contains 'logs' in path."""
        result = _resolve_logs_dir()
        
        # Should be a logs directory somewhere
        assert "logs" in str(result).lower() or result.name == "logs"
