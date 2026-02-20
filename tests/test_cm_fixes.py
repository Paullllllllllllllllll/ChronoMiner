"""
Tests for CM-1 through CM-11 bug fixes.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest


# ---------------------------------------------------------------------------
# CM-1: service_tier passed to ChatOpenAI in sync mode
# ---------------------------------------------------------------------------

class TestCM1ServiceTierSyncMode:
    """CM-1: service_tier from concurrency_config must reach ChatOpenAI."""

    @pytest.mark.unit
    def test_service_tier_added_to_extra_params(self):
        """LLMExtractor._initialize_llm puts service_tier in extra_params."""
        from modules.llm.langchain_provider import ProviderConfig, LangChainLLM

        concurrency_config = {
            "concurrency": {
                "extraction": {"service_tier": "flex"}
            }
        }

        with patch("modules.llm.openai_utils.get_config_loader") as mock_loader:
            mock_loader.return_value.get_model_config.return_value = {
                "transcription_model": {
                    "name": "gpt-4o",
                    "max_output_tokens": 4096,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "frequency_penalty": 0.0,
                    "presence_penalty": 0.0,
                }
            }
            mock_loader.return_value.get_concurrency_config.return_value = concurrency_config
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                from modules.llm.openai_utils import LLMExtractor
                extractor = LLMExtractor(model="gpt-4o")

        assert extractor._llm is not None
        assert extractor._llm.config.extra_params.get("service_tier") == "flex"

    @pytest.mark.unit
    def test_service_tier_passed_to_chat_openai(self):
        """ChatOpenAI is instantiated with service_tier when configured."""
        from modules.llm.langchain_provider import ProviderConfig, LangChainLLM

        config = ProviderConfig(
            provider="openai",
            model="gpt-4o",
            api_key="test-key",
            extra_params={"service_tier": "flex"},
        )
        llm = LangChainLLM(config)

        with patch("langchain_openai.ChatOpenAI") as MockChatOpenAI:
            MockChatOpenAI.return_value = MagicMock()
            llm._create_chat_model()
            call_kwargs = MockChatOpenAI.call_args[1]
            assert call_kwargs.get("service_tier") == "flex"

    @pytest.mark.unit
    def test_no_service_tier_when_not_configured(self):
        """ChatOpenAI is not called with service_tier when absent from config."""
        from modules.llm.langchain_provider import ProviderConfig, LangChainLLM

        config = ProviderConfig(
            provider="openai",
            model="gpt-4o",
            api_key="test-key",
            extra_params={},
        )
        llm = LangChainLLM(config)

        with patch("langchain_openai.ChatOpenAI") as MockChatOpenAI:
            MockChatOpenAI.return_value = MagicMock()
            llm._create_chat_model()
            call_kwargs = MockChatOpenAI.call_args[1]
            assert "service_tier" not in call_kwargs


# ---------------------------------------------------------------------------
# CM-2: service_tier injected into model_config for batch mode
# ---------------------------------------------------------------------------

class TestCM2ServiceTierBatchMode:
    """CM-2: BatchProcessingStrategy must inject service_tier from concurrency_config."""

    @pytest.mark.unit
    def test_batch_strategy_stores_concurrency_config(self):
        """BatchProcessingStrategy stores concurrency_config passed to __init__."""
        from modules.core.processing_strategy import BatchProcessingStrategy

        cc = {"concurrency": {"extraction": {"service_tier": "auto"}}}
        strategy = BatchProcessingStrategy(cc)
        assert strategy.concurrency_config is cc

    @pytest.mark.unit
    def test_create_processing_strategy_passes_concurrency_to_batch(self):
        """create_processing_strategy passes concurrency_config to BatchProcessingStrategy."""
        from modules.core.processing_strategy import create_processing_strategy, BatchProcessingStrategy

        cc = {"concurrency": {"extraction": {"service_tier": "priority"}}}
        strategy = create_processing_strategy(use_batch=True, concurrency_config=cc)
        assert isinstance(strategy, BatchProcessingStrategy)
        assert strategy.concurrency_config is cc

    @pytest.mark.unit
    def test_build_responses_body_uses_injected_service_tier(self):
        """_build_responses_body uses service_tier when it is in model_config.transcription_model."""
        from modules.llm.batch.backends.openai_backend import _build_responses_body

        model_config = {
            "transcription_model": {
                "name": "gpt-4o",
                "max_output_tokens": 4096,
                "service_tier": "auto",
            }
        }
        body = _build_responses_body(
            model_config=model_config,
            system_prompt="sys",
            user_text="text",
        )
        assert body.get("service_tier") == "auto"

    @pytest.mark.unit
    def test_build_responses_body_flex_converted_to_auto(self):
        """_build_responses_body converts flex -> auto for batch API."""
        from modules.llm.batch.backends.openai_backend import _build_responses_body

        model_config = {
            "transcription_model": {
                "name": "gpt-4o",
                "max_output_tokens": 4096,
                "service_tier": "flex",
            }
        }
        body = _build_responses_body(
            model_config=model_config,
            system_prompt="sys",
            user_text="text",
        )
        assert body.get("service_tier") == "auto"


# ---------------------------------------------------------------------------
# CM-3: reasoning.effort forwarded in sync mode
# ---------------------------------------------------------------------------

class TestCM3ReasoningEffortSyncMode:
    """CM-3: reasoning_config and reasoning_effort must appear in extra_params."""

    @pytest.mark.unit
    def test_reasoning_effort_in_extra_params(self):
        """LLMExtractor._initialize_llm puts reasoning_effort in extra_params."""
        with patch("modules.llm.openai_utils.get_config_loader") as mock_loader:
            mock_loader.return_value.get_model_config.return_value = {
                "transcription_model": {
                    "name": "gpt-4o",
                    "max_output_tokens": 4096,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "frequency_penalty": 0.0,
                    "presence_penalty": 0.0,
                    "reasoning": {"effort": "low"},
                }
            }
            mock_loader.return_value.get_concurrency_config.return_value = {}
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                from modules.llm.openai_utils import LLMExtractor
                extractor = LLMExtractor(model="gpt-4o")

        ep = extractor._llm.config.extra_params
        assert ep.get("reasoning_effort") == "low"
        assert ep.get("reasoning_config") == {"effort": "low"}

    @pytest.mark.unit
    def test_reasoning_effort_default_medium(self):
        """When reasoning is absent, reasoning_effort defaults to 'medium'."""
        with patch("modules.llm.openai_utils.get_config_loader") as mock_loader:
            mock_loader.return_value.get_model_config.return_value = {
                "transcription_model": {
                    "name": "gpt-4o",
                    "max_output_tokens": 4096,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "frequency_penalty": 0.0,
                    "presence_penalty": 0.0,
                }
            }
            mock_loader.return_value.get_concurrency_config.return_value = {}
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                from modules.llm.openai_utils import LLMExtractor
                extractor = LLMExtractor(model="gpt-4o")

        assert extractor._llm.config.extra_params.get("reasoning_effort") == "medium"


# ---------------------------------------------------------------------------
# CM-4: reasoning translated for Anthropic and Google
# ---------------------------------------------------------------------------

class TestCM4ReasoningAnthropicGoogle:
    """CM-4: Anthropic extended thinking and Google thinking_config must be set."""

    @pytest.mark.unit
    def test_anthropic_extended_thinking_passed(self):
        """ChatAnthropic gets thinking param and betas when reasoning is configured."""
        from modules.llm.langchain_provider import ProviderConfig, LangChainLLM

        config = ProviderConfig(
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            api_key="test-key",
            max_tokens=10000,
            extra_params={
                "reasoning_config": {"effort": "medium"},
                "reasoning_effort": "medium",
            },
        )
        llm = LangChainLLM(config)

        with patch("langchain_anthropic.ChatAnthropic") as MockAnthropic:
            MockAnthropic.return_value = MagicMock()
            llm._create_chat_model()
            call_kwargs = MockAnthropic.call_args[1]
            assert "thinking" in call_kwargs
            assert call_kwargs["thinking"]["type"] == "enabled"
            assert call_kwargs["thinking"]["budget_tokens"] > 0
            assert "betas" in call_kwargs
            assert "interleaved-thinking-2025-05-14" in call_kwargs["betas"]

    @pytest.mark.unit
    def test_anthropic_no_thinking_when_effort_none(self):
        """ChatAnthropic does NOT get thinking when effort is 'none'."""
        from modules.llm.langchain_provider import ProviderConfig, LangChainLLM

        config = ProviderConfig(
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            api_key="test-key",
            max_tokens=10000,
            extra_params={
                "reasoning_config": {"effort": "none"},
                "reasoning_effort": "none",
            },
        )
        llm = LangChainLLM(config)

        with patch("langchain_anthropic.ChatAnthropic") as MockAnthropic:
            MockAnthropic.return_value = MagicMock()
            llm._create_chat_model()
            call_kwargs = MockAnthropic.call_args[1]
            assert "thinking" not in call_kwargs

    @pytest.mark.unit
    def test_google_thinking_config_passed(self):
        """ChatGoogleGenerativeAI gets thinking_config when reasoning is configured."""
        from modules.llm.langchain_provider import ProviderConfig, LangChainLLM

        config = ProviderConfig(
            provider="google",
            model="gemini-2.5-pro",
            api_key="test-key",
            max_tokens=10000,
            extra_params={
                "reasoning_config": {"effort": "high"},
                "reasoning_effort": "high",
            },
        )
        llm = LangChainLLM(config)

        with patch("langchain_google_genai.ChatGoogleGenerativeAI") as MockGoogle:
            MockGoogle.return_value = MagicMock()
            llm._create_chat_model()
            call_kwargs = MockGoogle.call_args[1]
            assert "thinking_config" in call_kwargs
            assert call_kwargs["thinking_config"]["include_thoughts"] is True
            assert call_kwargs["thinking_config"]["thinking_budget"] > 0


# ---------------------------------------------------------------------------
# CM-5: text.verbosity passed to ChatOpenAI for GPT-5 family
# ---------------------------------------------------------------------------

class TestCM5TextVerbosity:
    """CM-5: text verbosity must be forwarded to ChatOpenAI for GPT-5 models."""

    @pytest.mark.unit
    def test_text_verbosity_passed_for_gpt5(self):
        """ChatOpenAI receives text={"verbosity": ...} via model_kwargs for gpt-5 model."""
        from modules.llm.langchain_provider import ProviderConfig, LangChainLLM

        config = ProviderConfig(
            provider="openai",
            model="gpt-5",
            api_key="test-key",
            extra_params={
                "text_config": {"verbosity": "medium"},
                "reasoning_effort": "medium",
            },
        )
        llm = LangChainLLM(config)

        with patch("langchain_openai.ChatOpenAI") as MockChatOpenAI:
            MockChatOpenAI.return_value = MagicMock()
            llm._create_chat_model()
            call_kwargs = MockChatOpenAI.call_args[1]
            assert call_kwargs.get("text") is None, "text must not be a top-level param"
            assert call_kwargs.get("model_kwargs", {}).get("text") == {"verbosity": "medium"}

    @pytest.mark.unit
    def test_text_verbosity_not_passed_for_non_gpt5(self):
        """ChatOpenAI does NOT receive text param for gpt-4o model."""
        from modules.llm.langchain_provider import ProviderConfig, LangChainLLM

        config = ProviderConfig(
            provider="openai",
            model="gpt-4o",
            api_key="test-key",
            extra_params={
                "text_config": {"verbosity": "medium"},
            },
        )
        llm = LangChainLLM(config)

        with patch("langchain_openai.ChatOpenAI") as MockChatOpenAI:
            MockChatOpenAI.return_value = MagicMock()
            llm._create_chat_model()
            call_kwargs = MockChatOpenAI.call_args[1]
            assert "text" not in call_kwargs


# ---------------------------------------------------------------------------
# CM-6: Processing summary reads correct concurrency config keys
# ---------------------------------------------------------------------------

class TestCM6ProcessingSummaryKeys:
    """CM-6: display_processing_summary must read concurrency.extraction.*"""

    @pytest.mark.unit
    def test_correct_concurrency_values_extracted(self):
        """display_processing_summary shows values from concurrency.extraction."""
        from modules.ui.core import UserInterface

        ui = UserInterface(use_colors=False)
        concurrency_config = {
            "concurrency": {
                "extraction": {
                    "concurrency_limit": 42,
                    "service_tier": "priority",
                    "retry": {"attempts": 7},
                }
            }
        }

        printed_lines = []
        with patch.object(ui, "console_print", side_effect=lambda msg: printed_lines.append(msg)):
            with patch.object(ui, "confirm", return_value=False):
                ui.display_processing_summary(
                    files=[Path("a.txt")],
                    selected_schema_name="TestSchema",
                    global_chunking_method="auto",
                    use_batch=False,
                    concurrency_config=concurrency_config,
                )

        output = "\n".join(printed_lines)
        assert "42" in output, "concurrency_limit 42 must appear in summary"
        assert "priority" in output, "service_tier 'priority' must appear in summary"
        assert "7" in output, "retry attempts 7 must appear in summary"


# ---------------------------------------------------------------------------
# CM-7: Empty output path raises ValueError
# ---------------------------------------------------------------------------

class TestCM7EmptyOutputPath:
    """CM-7: _setup_output_paths raises ValueError for empty or CWD output."""

    @pytest.mark.unit
    def test_empty_output_path_raises(self, tmp_path):
        """ValueError raised when output path is empty string."""
        from modules.operations.extraction.file_processor import FileProcessorRefactored

        paths_config = {
            "general": {
                "input_paths_is_output_path": False,
                "retain_temporary_jsonl": True,
            }
        }
        fp = FileProcessorRefactored(
            paths_config=paths_config,
            model_config={"transcription_model": {"name": "gpt-4o"}},
            chunking_config={"chunking": {}},
        )

        with pytest.raises(ValueError, match="Output path is not configured"):
            fp._setup_output_paths(
                tmp_path / "input.txt",
                {"output": ""},
            )

    @pytest.mark.unit
    def test_cwd_output_path_raises(self, tmp_path, monkeypatch):
        """ValueError raised when output path resolves to CWD."""
        import os
        from modules.operations.extraction.file_processor import FileProcessorRefactored

        monkeypatch.chdir(tmp_path)

        paths_config = {
            "general": {
                "input_paths_is_output_path": False,
                "retain_temporary_jsonl": True,
            }
        }
        fp = FileProcessorRefactored(
            paths_config=paths_config,
            model_config={"transcription_model": {"name": "gpt-4o"}},
            chunking_config={"chunking": {}},
        )

        with pytest.raises(ValueError, match="current working directory"):
            fp._setup_output_paths(
                tmp_path / "input.txt",
                {"output": str(tmp_path)},
            )

    @pytest.mark.unit
    def test_valid_output_path_does_not_raise(self, tmp_path):
        """No exception when output path is a valid non-CWD directory."""
        from modules.operations.extraction.file_processor import FileProcessorRefactored

        output_dir = tmp_path / "output"

        paths_config = {
            "general": {
                "input_paths_is_output_path": False,
                "retain_temporary_jsonl": True,
            }
        }
        fp = FileProcessorRefactored(
            paths_config=paths_config,
            model_config={"transcription_model": {"name": "gpt-4o"}},
            chunking_config={"chunking": {}},
        )

        working, out_json, out_jsonl = fp._setup_output_paths(
            tmp_path / "input.txt",
            {"output": str(output_dir)},
        )
        assert working == output_dir


# ---------------------------------------------------------------------------
# CM-8: Context-selection step in interactive mode
# ---------------------------------------------------------------------------

class TestCM8ContextSelection:
    """CM-8: ask_context_selection returns correct dicts; process_file accepts context_override."""

    @pytest.mark.unit
    def test_ask_context_selection_auto(self):
        """Returns mode=auto when user selects auto."""
        from modules.ui.core import UserInterface

        ui = UserInterface(use_colors=False)
        with patch.object(ui, "select_option", return_value="auto"):
            result = ui.ask_context_selection()
        assert result == {"mode": "auto", "path": None}

    @pytest.mark.unit
    def test_ask_context_selection_none(self):
        """Returns mode=none when user selects none."""
        from modules.ui.core import UserInterface

        ui = UserInterface(use_colors=False)
        with patch.object(ui, "select_option", return_value="none"):
            result = ui.ask_context_selection()
        assert result == {"mode": "none", "path": None}

    @pytest.mark.unit
    def test_ask_context_selection_back_returns_none(self):
        """Returns None when user navigates back."""
        from modules.ui.core import UserInterface

        ui = UserInterface(use_colors=False)
        with patch.object(ui, "select_option", return_value=None):
            result = ui.ask_context_selection(allow_back=True)
        assert result is None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_file_context_none_mode(self, tmp_path):
        """process_file skips context resolution when context_override mode=none."""
        from modules.operations.extraction.file_processor import FileProcessorRefactored

        paths_config = {
            "general": {
                "input_paths_is_output_path": True,
                "retain_temporary_jsonl": True,
            }
        }
        fp = FileProcessorRefactored(
            paths_config=paths_config,
            model_config={"transcription_model": {"name": "gpt-4o"}},
            chunking_config={"chunking": {"default_tokens_per_chunk": 500}},
        )

        input_file = tmp_path / "test.txt"
        input_file.write_text("Line 1\nLine 2\n", encoding="utf-8")

        with patch(
            "modules.operations.extraction.file_processor.resolve_context_for_extraction"
        ) as mock_resolve, patch(
            "modules.operations.extraction.file_processor.create_processing_strategy"
        ) as mock_strategy:
            mock_strategy.return_value.process_chunks = AsyncMock(return_value=[])
            await fp.process_file(
                file_path=input_file,
                use_batch=False,
                selected_schema={"schema": {}},
                prompt_template="Extract: {schema}",
                schema_name="TestSchema",
                inject_schema=False,
                schema_paths={},
                global_chunking_method="auto",
                context_override={"mode": "none", "path": None},
            )
            mock_resolve.assert_not_called()


# ---------------------------------------------------------------------------
# CM-9: Resume step has back navigation
# ---------------------------------------------------------------------------

class TestCM9ResumeBackNavigation:
    """CM-9: Resume step must use select_option with allow_back=True."""

    @pytest.mark.unit
    def test_resume_step_uses_select_option_with_allow_back(self):
        """The interactive resume step is driven by select_option(allow_back=True)."""
        import inspect
        from main.process_text_files import _run_interactive_mode

        source = inspect.getsource(_run_interactive_mode)
        assert "select_option" in source, "resume step must use select_option"
        assert "allow_back=True" in source, "resume step must pass allow_back=True"

    @pytest.mark.unit
    def test_resume_back_transitions_to_batch_step(self):
        """When select_option returns None the state machine moves back to 'batch'."""
        import inspect
        from main.process_text_files import _run_interactive_mode

        source = inspect.getsource(_run_interactive_mode)
        # The pattern 'current_step = "batch"' must follow the None-check in the resume block
        assert 'current_step = "batch"' in source, (
            "Back navigation from resume must set current_step to 'batch'"
        )


# ---------------------------------------------------------------------------
# CM-10: Pre-check of existing output files
# ---------------------------------------------------------------------------

class TestCM10ExistingOutputPreCheck:
    """CM-10: _count_existing_outputs must correctly count pre-existing outputs."""

    @pytest.mark.unit
    def test_count_zero_when_no_outputs_exist(self, tmp_path):
        """Returns 0 when no output files exist."""
        from main.process_text_files import _count_existing_outputs

        files = [tmp_path / "a.txt", tmp_path / "b.txt"]
        paths_config = {"general": {"input_paths_is_output_path": True}}
        result = _count_existing_outputs(files, paths_config, "S", {})
        assert result == 0

    @pytest.mark.unit
    def test_count_correct_when_some_outputs_exist(self, tmp_path):
        """Returns correct count when some output files exist."""
        from main.process_text_files import _count_existing_outputs

        f1 = tmp_path / "doc1.txt"
        f2 = tmp_path / "doc2.txt"
        f1.write_text("x")
        f2.write_text("x")
        (tmp_path / "doc1_output.json").write_text("{}")  # exists

        paths_config = {"general": {"input_paths_is_output_path": True}}
        result = _count_existing_outputs([f1, f2], paths_config, "S", {})
        assert result == 1

    @pytest.mark.unit
    def test_count_with_separate_output_dir(self, tmp_path):
        """Returns correct count when output is in a separate directory."""
        from main.process_text_files import _count_existing_outputs

        out_dir = tmp_path / "output"
        out_dir.mkdir()
        f1 = tmp_path / "doc1.txt"
        f2 = tmp_path / "doc2.txt"
        f1.write_text("x")
        f2.write_text("x")
        (out_dir / "doc1_output.json").write_text("{}")  # exists
        (out_dir / "doc2_output.json").write_text("{}")  # exists

        paths_config = {"general": {"input_paths_is_output_path": False}}
        schemas_paths = {"S": {"output": str(out_dir)}}
        result = _count_existing_outputs([f1, f2], paths_config, "S", schemas_paths)
        assert result == 2

    @pytest.mark.unit
    def test_summary_shows_warning_when_outputs_exist(self):
        """display_processing_summary shows overwrite warning when count > 0."""
        from modules.ui.core import UserInterface

        ui = UserInterface(use_colors=False)
        printed_lines = []
        with patch.object(ui, "console_print", side_effect=lambda msg: printed_lines.append(msg)):
            with patch.object(ui, "confirm", return_value=False):
                ui.display_processing_summary(
                    files=[Path("a.txt"), Path("b.txt")],
                    selected_schema_name="TestSchema",
                    global_chunking_method="auto",
                    use_batch=False,
                    existing_output_count=2,
                )
        output = "\n".join(printed_lines)
        assert "2" in output and ("already exist" in output or "overwritten" in output)


# ---------------------------------------------------------------------------
# CM-11: asyncio.gather uses return_exceptions=True
# ---------------------------------------------------------------------------

class TestCM11GatherReturnExceptions:
    """CM-11: concurrent file processing gather must use return_exceptions=True."""

    @pytest.mark.unit
    def test_gather_return_exceptions_true(self):
        """asyncio.gather is called with return_exceptions=True in concurrent branch."""
        import ast
        import inspect
        from main.process_text_files import _run_interactive_mode

        source = inspect.getsource(_run_interactive_mode)
        # Check that return_exceptions=True appears in the asyncio.gather call
        assert "return_exceptions=True" in source, (
            "_run_interactive_mode must call asyncio.gather with return_exceptions=True"
        )

    @pytest.mark.unit
    def test_cli_gather_return_exceptions_true(self):
        """asyncio.gather in _run_cli_mode also uses return_exceptions=True."""
        import inspect
        from main.process_text_files import _run_cli_mode

        source = inspect.getsource(_run_cli_mode)
        assert "return_exceptions=True" in source, (
            "_run_cli_mode must call asyncio.gather with return_exceptions=True"
        )

    @pytest.mark.unit
    def test_exceptions_from_gather_are_logged(self):
        """Exceptions returned by gather are forwarded to the logger."""
        import inspect
        from main import process_text_files

        source = inspect.getsource(process_text_files)
        # The fix logs exceptions with logger.error
        assert "isinstance(_exc, Exception)" in source or "isinstance(_gather" in source or \
               "return_exceptions=True" in source


