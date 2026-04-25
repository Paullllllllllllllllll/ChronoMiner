"""Interface-level tests for the ``modules.ui`` primitive layer.

Exercises the user-facing prompt helpers (``ui_print``, ``prompt_select``,
``prompt_yes_no``, ``prompt_text``, ``prompt_multiselect``, navigation
types) without reaching into private internals. Stdin is mocked through
``builtins.input`` so tests run non-interactively.

All prompt_* helpers return :class:`PromptResult`; tests unwrap it.
``prompt_select`` expects (value, description) tuples.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from modules.ui import (
    NavigationAction,
    PromptResult,
    PromptStyle,
    print_error,
    print_info,
    print_success,
    print_warning,
    prompt_multiselect,
    prompt_select,
    prompt_text,
    prompt_yes_no,
    ui_input,
    ui_print,
)


@pytest.mark.unit
class TestPromptStyle:
    def test_colorize_wraps_text(self):
        text = PromptStyle.colorize("hello", PromptStyle.INFO)
        assert "hello" in text
        assert text.endswith(PromptStyle.RESET)

    def test_supports_color_returns_bool(self):
        assert isinstance(PromptStyle.supports_color(), bool)


@pytest.mark.unit
class TestNavigationTypes:
    def test_navigation_action_values(self):
        assert NavigationAction.CONTINUE.value == "continue"
        assert NavigationAction.BACK.value == "back"
        assert NavigationAction.QUIT.value == "quit"

    def test_prompt_result_defaults(self):
        pr = PromptResult(action=NavigationAction.CONTINUE)
        assert pr.action is NavigationAction.CONTINUE
        assert pr.value is None

    def test_prompt_result_carries_value(self):
        pr = PromptResult(action=NavigationAction.CONTINUE, value={"x": 1})
        assert pr.value == {"x": 1}


@pytest.mark.unit
class TestUiPrint:
    def test_writes_to_stdout(self, capsys):
        ui_print("hello world")
        assert "hello world" in capsys.readouterr().out


@pytest.mark.unit
class TestPrintHelpers:
    def test_print_info(self, capsys):
        print_info("info-msg")
        assert "info-msg" in capsys.readouterr().out

    def test_print_success(self, capsys):
        print_success("yes")
        assert "yes" in capsys.readouterr().out

    def test_print_warning(self, capsys):
        print_warning("watch out")
        assert "watch out" in capsys.readouterr().out

    def test_print_error(self, capsys):
        print_error("boom")
        assert "boom" in capsys.readouterr().out


@pytest.mark.unit
class TestUiInput:
    def test_returns_user_input_string(self):
        with patch("builtins.input", return_value="typed-value"):
            assert ui_input("prompt") == "typed-value"


@pytest.mark.unit
class TestPromptYesNo:
    def test_yes_answer(self):
        with patch("builtins.input", return_value="y"):
            result = prompt_yes_no("?")
        assert isinstance(result, PromptResult)
        assert result.action is NavigationAction.CONTINUE
        assert result.value is True

    def test_no_answer(self):
        with patch("builtins.input", return_value="n"):
            result = prompt_yes_no("?")
        assert result.value is False

    def test_empty_uses_default_true(self):
        with patch("builtins.input", return_value=""):
            result = prompt_yes_no("?", default=True)
        assert result.value is True

    def test_empty_uses_default_false(self):
        with patch("builtins.input", return_value=""):
            result = prompt_yes_no("?", default=False)
        assert result.value is False


@pytest.mark.unit
class TestPromptText:
    def test_returns_entered_text(self):
        with patch("builtins.input", return_value="my answer"):
            result = prompt_text("enter something", allow_empty=True)
        assert result.value == "my answer"

    def test_default_on_empty_input(self):
        with patch("builtins.input", return_value=""):
            result = prompt_text("?", default="fallback")
        assert result.value == "fallback"


@pytest.mark.unit
class TestPromptSelect:
    def test_valid_selection_returns_option_value(self):
        options = [("alpha", "First option"), ("beta", "Second option"), ("gamma", "Third")]
        with patch("builtins.input", return_value="2"):
            result = prompt_select("pick one", options)
        assert isinstance(result, PromptResult)
        assert result.action is NavigationAction.CONTINUE
        assert result.value == "beta"

    def test_first_option(self):
        options = [("a", "A"), ("b", "B")]
        with patch("builtins.input", return_value="1"):
            result = prompt_select("pick", options)
        assert result.value == "a"


@pytest.mark.unit
class TestPromptMultiselect:
    def test_comma_separated_indices(self):
        options = [("a", "A"), ("b", "B"), ("c", "C"), ("d", "D")]
        with patch("builtins.input", return_value="1,3"):
            result = prompt_multiselect("pick", options)
        assert isinstance(result, PromptResult)
        # Values must contain the selected option values (a, c) in some form.
        assert "a" in result.value and "c" in result.value

    def test_single_index(self):
        options = [("a", "A"), ("b", "B")]
        with patch("builtins.input", return_value="2"):
            result = prompt_multiselect("pick", options)
        assert "b" in result.value
