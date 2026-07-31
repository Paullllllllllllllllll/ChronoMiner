"""Interface-level tests for the ``modules.ui`` primitive layer.

Exercises the user-facing prompt helpers (``ui_print``, ``prompt_select``,
``prompt_yes_no``, ``prompt_text``, ``prompt_multiselect``, navigation
types) without reaching into private internals. Stdin is mocked through
``builtins.input`` so tests run non-interactively.

All prompt_* helpers return :class:`PromptResult`; tests unwrap it.
``prompt_select`` expects (value, description) tuples.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

import modules.ui.prompts as prompts_mod
from modules.ui import (
    IndexSelectionError,
    NavigationAction,
    PromptResult,
    PromptStyle,
    parse_index_selection,
    print_error,
    print_info,
    print_navigation_help,
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
    def test_colorize_wraps_text(self, monkeypatch):
        # Force color support on so this test exercises colorize() itself,
        # independent of the tty/colorama/NO_COLOR checks covered below.
        monkeypatch.setattr(PromptStyle, "supports_color", staticmethod(lambda: True))
        text = PromptStyle.colorize("hello", PromptStyle.INFO)
        assert "hello" in text
        assert text.endswith(PromptStyle.RESET)

    def test_supports_color_returns_bool(self):
        assert isinstance(PromptStyle.supports_color(), bool)


@pytest.mark.unit
class TestSupportsColorHonest:
    """supports_color() must reflect real terminal capability, not just
    assume it because colorama is imported (regression)."""

    def test_false_when_colorama_unavailable(self, monkeypatch):
        monkeypatch.setattr(prompts_mod, "_COLORAMA_AVAILABLE", False)
        assert PromptStyle.supports_color() is False

    def test_false_when_not_a_tty(self, monkeypatch):
        monkeypatch.setattr(prompts_mod, "_COLORAMA_AVAILABLE", True)
        monkeypatch.setattr(
            prompts_mod.sys, "stdout", SimpleNamespace(isatty=lambda: False)
        )
        assert PromptStyle.supports_color() is False

    def test_false_when_no_color_env_set(self, monkeypatch):
        monkeypatch.setattr(prompts_mod, "_COLORAMA_AVAILABLE", True)
        monkeypatch.setattr(
            prompts_mod.sys, "stdout", SimpleNamespace(isatty=lambda: True)
        )
        monkeypatch.setenv("NO_COLOR", "1")
        assert PromptStyle.supports_color() is False

    def test_true_when_tty_and_colorama_and_no_no_color(self, monkeypatch):
        monkeypatch.setattr(prompts_mod, "_COLORAMA_AVAILABLE", True)
        monkeypatch.setattr(
            prompts_mod.sys, "stdout", SimpleNamespace(isatty=lambda: True)
        )
        monkeypatch.delenv("NO_COLOR", raising=False)
        assert PromptStyle.supports_color() is True


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
        options = [
            ("alpha", "First option"),
            ("beta", "Second option"),
            ("gamma", "Third"),
        ]
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


@pytest.mark.unit
class TestAllowQuit:
    """'q' quits by default but is inert when allow_quit=False (mid-run prompts)."""

    def test_prompt_select_default_quits(self):
        options = [("a", "A"), ("b", "B")]
        with (
            patch("builtins.input", return_value="q"),
            pytest.raises(SystemExit),
        ):
            prompt_select("pick", options)

    def test_prompt_select_allow_quit_false_treats_q_as_invalid(self):
        options = [("a", "A"), ("b", "B")]
        # 'q' must not exit; it is invalid input, so the prompt re-asks and
        # the following '1' resolves the selection.
        with patch("builtins.input", side_effect=["q", "1"]):
            result = prompt_select("pick", options, allow_quit=False)
        assert result.action is NavigationAction.CONTINUE
        assert result.value == "a"

    def test_prompt_text_default_quits(self):
        with (
            patch("builtins.input", return_value="q"),
            pytest.raises(SystemExit),
        ):
            prompt_text("enter", allow_empty=True)

    def test_prompt_text_allow_quit_false_returns_q_as_text(self):
        with patch("builtins.input", return_value="q"):
            result = prompt_text("enter", allow_empty=True, allow_quit=False)
        assert result.action is NavigationAction.CONTINUE
        assert result.value == "q"


@pytest.mark.unit
class TestNonAsciiDigitSelection:
    """int() rejects digits that str.isdigit() accepts (superscripts, etc.);
    prompt_select must treat them as invalid input, not crash (regression)."""

    def test_superscript_digit_is_invalid_not_a_crash(self):
        options = [("a", "A"), ("b", "B")]
        # U+00B2 (superscript two) passes isdigit() but int() raises.
        with patch("builtins.input", side_effect=["²", "2"]):
            result = prompt_select("pick", options)
        assert result.action is NavigationAction.CONTINUE
        assert result.value == "b"

    def test_devanagari_digit_is_invalid_not_a_crash(self):
        options = [("a", "A")]
        with patch("builtins.input", side_effect=["१", "1"]):
            result = prompt_select("pick", options)
        assert result.value == "a"


@pytest.mark.unit
class TestNavigationHelp:
    """The help line must advertise only the navigation actually available."""

    def test_help_omits_quit_when_disallowed(self, capsys):
        print_navigation_help(allow_back=True, allow_quit=False)
        out = capsys.readouterr().out
        assert "go back" in out
        assert "quit" not in out

    def test_help_prints_nothing_when_both_disallowed(self, capsys):
        print_navigation_help(allow_back=False, allow_quit=False)
        assert capsys.readouterr().out.strip() == ""

    def test_prompt_select_help_omits_quit_when_disallowed(self, capsys):
        options = [("a", "A"), ("b", "B")]
        with patch("builtins.input", return_value="1"):
            prompt_select("pick", options, allow_back=False, allow_quit=False)
        assert "'q' to quit" not in capsys.readouterr().out

    def test_prompt_text_shows_quit_help_without_back(self, capsys):
        """allow_back=False, allow_quit=True still tells the user 'q' exits."""
        with patch("builtins.input", return_value="x"):
            prompt_text("enter", allow_back=False, allow_quit=True)
        assert "'q' to quit" in capsys.readouterr().out

    def test_prompt_yes_no_shows_quit_help_without_back(self, capsys):
        with patch("builtins.input", return_value="y"):
            prompt_yes_no("ok?", allow_back=False, allow_quit=True)
        assert "'q' to quit" in capsys.readouterr().out


@pytest.mark.unit
class TestEmptyChoiceGuards:
    """An empty option list must not strand the user in an unanswerable loop."""

    def test_prompt_select_empty_options_returns_back(self, capsys):
        with patch("builtins.input", side_effect=AssertionError("must not prompt")):
            result = prompt_select("pick", [])
        assert result.action is NavigationAction.BACK
        assert "No options available" in capsys.readouterr().out

    def test_prompt_multiselect_empty_items_returns_back(self, capsys):
        with patch("builtins.input", side_effect=AssertionError("must not prompt")):
            result = prompt_multiselect("pick", [])
        assert result.action is NavigationAction.BACK
        assert "No items available" in capsys.readouterr().out


@pytest.mark.unit
class TestPromptTextDefaultVersusNavigation:
    """The default must be substituted only after the navigation check, so a
    default that happens to read like a command is returned as a value."""

    def test_default_that_looks_like_quit_is_returned(self):
        with patch("builtins.input", return_value=""):
            result = prompt_text("enter", default="q")
        assert result.action is NavigationAction.CONTINUE
        assert result.value == "q"

    def test_default_that_looks_like_back_is_returned(self):
        with patch("builtins.input", return_value=""):
            result = prompt_text("enter", default="b", allow_back=True)
        assert result.action is NavigationAction.CONTINUE
        assert result.value == "b"

    def test_explicitly_typed_quit_still_exits(self):
        with (
            patch("builtins.input", return_value="q"),
            pytest.raises(SystemExit),
        ):
            prompt_text("enter", default="fallback")


@pytest.mark.unit
class TestParseIndexSelection:
    """The shared index/range parser backing every numbered-list prompt."""

    def test_comma_separated(self):
        assert parse_index_selection("1,3,5", 5) == {0, 2, 4}

    def test_range_and_single(self):
        assert parse_index_selection("1-3,5", 5) == {0, 1, 2, 4}

    def test_whitespace_tolerated(self):
        assert parse_index_selection(" 2 , 4 ", 4) == {1, 3}

    def test_empty_tokens_ignored(self):
        assert parse_index_selection("1,,3", 3) == {0, 2}

    def test_empty_string_selects_nothing(self):
        assert parse_index_selection("", 3) == set()

    def test_single_point_range(self):
        assert parse_index_selection("2-2", 3) == {1}

    def test_out_of_range_single_raises_with_token(self):
        with pytest.raises(IndexSelectionError) as exc:
            parse_index_selection("9", 3)
        assert exc.value.part == "9"
        assert exc.value.is_range is False
        assert str(exc.value) == "Index 9 out of range"

    def test_out_of_range_high_bound_raises(self):
        with pytest.raises(IndexSelectionError) as exc:
            parse_index_selection("1-9", 3)
        assert exc.value.part == "1-9"
        assert exc.value.is_range is True
        assert str(exc.value) == "Invalid range: 1-9"

    def test_reversed_range_raises(self):
        with pytest.raises(IndexSelectionError):
            parse_index_selection("3-1", 5)

    def test_zero_is_out_of_range(self):
        with pytest.raises(IndexSelectionError):
            parse_index_selection("0", 5)

    def test_non_numeric_raises_plain_value_error(self):
        with pytest.raises(ValueError) as exc:
            parse_index_selection("abc", 5)
        assert not isinstance(exc.value, IndexSelectionError)

    def test_negative_token_raises_plain_value_error(self):
        with pytest.raises(ValueError) as exc:
            parse_index_selection("-3", 5)
        assert not isinstance(exc.value, IndexSelectionError)


@pytest.mark.unit
class TestPromptMultiselectMessagesUnchanged:
    """The consolidated parser must keep the multiselect wording intact."""

    def test_out_of_range_single_message(self, capsys):
        items = [("a", "A"), ("b", "B")]
        with patch("builtins.input", side_effect=["9", "1"]):
            result = prompt_multiselect("pick", items)
        out = capsys.readouterr().out
        assert "Selection 9 is out of range. Must be between 1 and 2." in out
        assert result.value == ["a"]

    def test_out_of_range_range_message(self, capsys):
        items = [("a", "A"), ("b", "B")]
        with patch("builtins.input", side_effect=["1-9", "2"]):
            result = prompt_multiselect("pick", items)
        out = capsys.readouterr().out
        assert "Range 1-9 is invalid. Must be between 1 and 2." in out
        assert result.value == ["b"]

    def test_malformed_numeric_message(self, capsys):
        items = [("a", "A"), ("b", "B")]
        with patch("builtins.input", side_effect=["1-2-3", "1"]):
            result = prompt_multiselect("pick", items)
        assert "Invalid input: '1-2-3'" in capsys.readouterr().out
        assert result.value == ["a"]

    def test_range_selection_still_works(self):
        items = [("a", "A"), ("b", "B"), ("c", "C")]
        with patch("builtins.input", return_value="1-2"):
            result = prompt_multiselect("pick", items)
        assert result.value == ["a", "b"]
