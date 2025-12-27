import sys
import pytest
from unittest.mock import Mock, patch
from modules.cli.mode_detector import (
    detect_execution_mode,
    should_use_interactive_mode,
    get_mode_description
)


@pytest.mark.unit
def test_detect_execution_mode_no_args_interactive_config():
    mock_loader = Mock()
    mock_loader.get_paths_config = Mock(return_value={
        "general": {"interactive_mode": True}
    })
    
    with patch.object(sys, "argv", ["script.py"]):
        is_interactive, has_cli_args = detect_execution_mode(mock_loader)
    
    assert is_interactive is True
    assert has_cli_args is False


@pytest.mark.unit
def test_detect_execution_mode_with_args_overrides_config():
    mock_loader = Mock()
    mock_loader.get_paths_config = Mock(return_value={
        "general": {"interactive_mode": True}
    })
    
    with patch.object(sys, "argv", ["script.py", "--input", "file.txt"]):
        is_interactive, has_cli_args = detect_execution_mode(mock_loader)
    
    assert is_interactive is False
    assert has_cli_args is True


@pytest.mark.unit
def test_detect_execution_mode_cli_config_no_args():
    mock_loader = Mock()
    mock_loader.get_paths_config = Mock(return_value={
        "general": {"interactive_mode": False}
    })
    
    with patch.object(sys, "argv", ["script.py"]):
        is_interactive, has_cli_args = detect_execution_mode(mock_loader)
    
    assert is_interactive is False
    assert has_cli_args is False


@pytest.mark.unit
def test_detect_execution_mode_help_flag():
    mock_loader = Mock()
    mock_loader.get_paths_config = Mock(return_value={
        "general": {"interactive_mode": True}
    })
    
    with patch.object(sys, "argv", ["script.py", "-h"]):
        is_interactive, has_cli_args = detect_execution_mode(mock_loader)
    
    assert is_interactive is False
    assert has_cli_args is True


@pytest.mark.unit
def test_detect_execution_mode_long_help_flag():
    mock_loader = Mock()
    mock_loader.get_paths_config = Mock(return_value={
        "general": {"interactive_mode": True}
    })
    
    with patch.object(sys, "argv", ["script.py", "--help"]):
        is_interactive, has_cli_args = detect_execution_mode(mock_loader)
    
    assert is_interactive is False
    assert has_cli_args is True


@pytest.mark.unit
def test_detect_execution_mode_default_interactive_when_not_specified():
    mock_loader = Mock()
    mock_loader.get_paths_config = Mock(return_value={
        "general": {}
    })
    
    with patch.object(sys, "argv", ["script.py"]):
        is_interactive, has_cli_args = detect_execution_mode(mock_loader)
    
    assert is_interactive is True
    assert has_cli_args is False


@pytest.mark.unit
def test_should_use_interactive_mode_returns_bool():
    mock_loader = Mock()
    mock_loader.get_paths_config = Mock(return_value={
        "general": {"interactive_mode": True}
    })
    
    with patch.object(sys, "argv", ["script.py"]):
        result = should_use_interactive_mode(mock_loader)
    
    assert isinstance(result, bool)
    assert result is True


@pytest.mark.unit
def test_should_use_interactive_mode_with_cli_args():
    mock_loader = Mock()
    mock_loader.get_paths_config = Mock(return_value={
        "general": {"interactive_mode": True}
    })
    
    with patch.object(sys, "argv", ["script.py", "--schema", "TestSchema"]):
        result = should_use_interactive_mode(mock_loader)
    
    assert result is False


@pytest.mark.unit
def test_get_mode_description_interactive():
    description = get_mode_description(True)
    assert "interactive" in description.lower()
    assert isinstance(description, str)


@pytest.mark.unit
def test_get_mode_description_cli():
    description = get_mode_description(False)
    assert "cli" in description.lower()
    assert isinstance(description, str)


@pytest.mark.unit
def test_detect_execution_mode_missing_general_section():
    mock_loader = Mock()
    mock_loader.get_paths_config = Mock(return_value={})
    
    with patch.object(sys, "argv", ["script.py"]):
        is_interactive, has_cli_args = detect_execution_mode(mock_loader)
    
    assert is_interactive is True
    assert has_cli_args is False


@pytest.mark.unit
def test_detect_execution_mode_multiple_cli_args():
    mock_loader = Mock()
    mock_loader.get_paths_config = Mock(return_value={
        "general": {"interactive_mode": True}
    })
    
    with patch.object(sys, "argv", ["script.py", "--input", "file.txt", "--output", "out.json"]):
        is_interactive, has_cli_args = detect_execution_mode(mock_loader)
    
    assert is_interactive is False
    assert has_cli_args is True
