"""Regression tests for live-testing bugs CM-1 through CM-5 (July 2026).

CM-4: check_batches.py must not emit a false all-clear when a configured
      output directory cannot be resolved.
CM-2: max_output_tokens must be clamped to the target model's real cap.
CM-1: Gemini 3.x reasoning control must use a supported constructor
      parameter (thinking_level / thinking_budget, not thinking_config).
CM-3: Anthropic structured output must degrade gracefully when the schema
      exceeds Anthropic's union-type parameter limit.
CM-5: check_batches / cancel_batches / repair_extractions must accept the
      shared --interactive/--non-interactive mode-override flags.
CM-6: batch finalization (check_batches / repair_extractions) must write the
      final output next to the submission (parent of temp_jsonl/), not into
      the schema's configured default output directory.
"""

from __future__ import annotations

import json
from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# CM-4: check_batches false all-clear on directory-resolution error
# ---------------------------------------------------------------------------


class TestCM4CheckBatchesScanErrors:
    """A swallowed directory-resolution error must surface as a scan error."""

    @staticmethod
    def _write_batch_temp_file(directory):
        temp_file = directory / "doc_temp.jsonl"
        record = {"batch_tracking": {"batch_id": "batch-123", "provider": "openai"}}
        temp_file.write_text(
            json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return temp_file

    @pytest.mark.unit
    def test_output_dir_resolution_error_bumps_errors(self, tmp_path):
        """process_all_batches records an 'errors' count instead of silently
        treating an unresolvable output directory as 'no batches'."""
        from main.check_batches import process_all_batches

        self._write_batch_temp_file(tmp_path)
        agg: dict[str, int] = {
            "finalized": 0,
            "pending": 0,
            "failed": 0,
            "errors": 0,
        }

        with patch(
            "main.check_batches._resolve_group_output_dir",
            side_effect=ValueError(
                "Output directory not specified in schema configuration"
            ),
        ):
            process_all_batches(
                root_folder=tmp_path,
                processing_settings={},
                schema_name="TestSchema",
                schema_config={},
                ui=None,
                agg=agg,
            )

        assert agg["errors"] == 1
        assert agg["finalized"] == 0

    @pytest.mark.unit
    def test_no_error_bump_when_no_temp_files(self, tmp_path):
        """An empty directory is a genuine all-clear, not a scan error."""
        from main.check_batches import process_all_batches

        agg: dict[str, int] = {
            "finalized": 0,
            "pending": 0,
            "failed": 0,
            "errors": 0,
        }
        process_all_batches(
            root_folder=tmp_path,
            processing_settings={},
            schema_name="TestSchema",
            schema_config={},
            ui=None,
            agg=agg,
        )
        assert agg["errors"] == 0

    @pytest.mark.unit
    def test_cli_exits_nonzero_and_reports_errors_in_json(self, tmp_path, capsys):
        """run_cli exits non-zero and the --json summary carries the error
        count when any configured directory cannot be scanned."""
        from main.check_batches import CheckBatchesScript

        script = CheckBatchesScript()
        repo_dir = tmp_path
        self._write_batch_temp_file(repo_dir)

        args = Namespace(
            schema=None,
            input=None,
            verbose=False,
            json_summary=True,
            interactive=False,
            non_interactive=True,
        )

        with (
            patch.object(
                CheckBatchesScript,
                "_load_batch_config",
                lambda self_: setattr(
                    self_, "repo_info_list", [("TestSchema", repo_dir, {})]
                ),
            ),
            patch(
                "main.check_batches._resolve_group_output_dir",
                side_effect=ValueError(
                    "Output directory not specified in schema configuration"
                ),
            ),
            pytest.raises(SystemExit) as excinfo,
        ):
            script.run_cli(args)

        assert excinfo.value.code == 1
        summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert summary["errors"] == 1
        assert summary["pending"] == 0

    @pytest.mark.unit
    def test_cli_exits_zero_when_nothing_found(self, tmp_path, capsys):
        """A genuinely empty scan still exits 0 (no false alarm either)."""
        from main.check_batches import CheckBatchesScript

        script = CheckBatchesScript()
        args = Namespace(
            schema=None,
            input=None,
            verbose=False,
            json_summary=True,
            interactive=False,
            non_interactive=True,
        )

        with patch.object(
            CheckBatchesScript,
            "_load_batch_config",
            lambda self_: setattr(
                self_, "repo_info_list", [("TestSchema", tmp_path, {})]
            ),
        ):
            script.run_cli(args)  # must NOT raise SystemExit

        summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert summary == {"finalized": 0, "pending": 0, "failed": 0, "errors": 0}


# ---------------------------------------------------------------------------
# CM-2: max_output_tokens clamped to the model's real output cap
# ---------------------------------------------------------------------------


class TestCM2MaxOutputTokensClamp:
    """Requested max_tokens above the registry cap must be clamped."""

    @pytest.mark.unit
    def test_registry_knows_haiku_4_5_cap(self):
        from modules.config.capabilities import detect_capabilities

        caps = detect_capabilities("claude-haiku-4-5-20251001")
        assert caps.max_output_tokens == 64000

    @pytest.mark.unit
    def test_anthropic_max_tokens_clamped(self):
        """128,000 requested for claude-haiku-4-5 is clamped to 64,000."""
        from modules.llm.langchain_provider import LangChainLLM, ProviderConfig

        config = ProviderConfig(
            provider="anthropic",
            model="claude-haiku-4-5-20251001",
            api_key="test-key",
            max_tokens=128000,
        )
        llm = LangChainLLM(config)

        with patch("langchain_anthropic.ChatAnthropic") as MockAnthropic:
            MockAnthropic.return_value = MagicMock()
            llm._create_chat_model()
            call_kwargs = MockAnthropic.call_args[1]
            assert call_kwargs["max_tokens"] == 64000

    @pytest.mark.unit
    def test_request_below_cap_is_untouched(self):
        from modules.llm.langchain_provider import LangChainLLM, ProviderConfig

        config = ProviderConfig(
            provider="anthropic",
            model="claude-haiku-4-5-20251001",
            api_key="test-key",
            max_tokens=8000,
        )
        llm = LangChainLLM(config)

        with patch("langchain_anthropic.ChatAnthropic") as MockAnthropic:
            MockAnthropic.return_value = MagicMock()
            llm._create_chat_model()
            assert MockAnthropic.call_args[1]["max_tokens"] == 8000

    @pytest.mark.unit
    def test_unknown_cap_means_no_clamping(self):
        """Models without a registered cap (None) are never clamped."""
        from modules.llm.langchain_provider import LangChainLLM, ProviderConfig

        config = ProviderConfig(
            provider="anthropic",
            model="claude-experimental-9",  # generic 'claude' family, cap None
            api_key="test-key",
            max_tokens=128000,
        )
        llm = LangChainLLM(config)

        with patch("langchain_anthropic.ChatAnthropic") as MockAnthropic:
            MockAnthropic.return_value = MagicMock()
            llm._create_chat_model()
            assert MockAnthropic.call_args[1]["max_tokens"] == 128000

    @pytest.mark.unit
    def test_thinking_budget_computed_from_clamped_value(self):
        """The Anthropic thinking budget derives from the clamped budget so
        budget_tokens can never exceed the model's real output ceiling."""
        from modules.llm.langchain_provider import (
            LangChainLLM,
            ProviderConfig,
            _compute_reasoning_budget,
        )

        config = ProviderConfig(
            provider="anthropic",
            model="claude-haiku-4-5-20251001",
            api_key="test-key",
            max_tokens=128000,
            extra_params={
                "reasoning_config": {"effort": "high"},
                "reasoning_effort": "high",
            },
        )
        llm = LangChainLLM(config)

        with patch("langchain_anthropic.ChatAnthropic") as MockAnthropic:
            MockAnthropic.return_value = MagicMock()
            llm._create_chat_model()
            call_kwargs = MockAnthropic.call_args[1]
            expected = _compute_reasoning_budget(max_tokens=64000, effort="high")
            assert call_kwargs["thinking"]["budget_tokens"] == expected


# ---------------------------------------------------------------------------
# CM-1: Gemini 3.x reasoning control uses a supported parameter
# ---------------------------------------------------------------------------


class TestCM1GeminiThinkingParameter:
    """Reasoning effort must map to thinking_level / thinking_budget."""

    @pytest.mark.unit
    def test_installed_langchain_supports_chosen_params(self):
        """Guard against dependency drift: the constructor fields we emit
        must exist on the installed ChatGoogleGenerativeAI."""
        from langchain_google_genai import ChatGoogleGenerativeAI

        fields = ChatGoogleGenerativeAI.model_fields
        assert "thinking_level" in fields
        assert "thinking_budget" in fields
        assert "thinking_config" not in fields

    @pytest.mark.unit
    def test_gemini_3x_uses_thinking_level(self):
        """gemini-3.1 with --reasoning-effort low gets thinking_level='low'
        and never the unsupported thinking_config dict."""
        from modules.llm.langchain_provider import LangChainLLM, ProviderConfig

        config = ProviderConfig(
            provider="google",
            model="gemini-3.1-flash-lite",
            api_key="test-key",
            max_tokens=10000,
            extra_params={
                "reasoning_config": {"effort": "low"},
                "reasoning_effort": "low",
            },
        )
        llm = LangChainLLM(config)

        with patch("langchain_google_genai.ChatGoogleGenerativeAI") as MockGoogle:
            MockGoogle.return_value = MagicMock()
            llm._create_chat_model()
            call_kwargs = MockGoogle.call_args[1]
            assert "thinking_config" not in call_kwargs
            assert "thinking_budget" not in call_kwargs
            assert call_kwargs["thinking_level"] == "low"

    @pytest.mark.unit
    def test_gemini_3x_xhigh_maps_to_high(self):
        from modules.llm.langchain_provider import LangChainLLM, ProviderConfig

        config = ProviderConfig(
            provider="google",
            model="gemini-3-pro",
            api_key="test-key",
            max_tokens=10000,
            extra_params={
                "reasoning_config": {"effort": "xhigh"},
                "reasoning_effort": "xhigh",
            },
        )
        llm = LangChainLLM(config)

        with patch("langchain_google_genai.ChatGoogleGenerativeAI") as MockGoogle:
            MockGoogle.return_value = MagicMock()
            llm._create_chat_model()
            assert MockGoogle.call_args[1]["thinking_level"] == "high"

    @pytest.mark.unit
    def test_gemini_25_uses_thinking_budget(self):
        """Pre-3.x Gemini models keep the token-budget control."""
        from modules.llm.langchain_provider import LangChainLLM, ProviderConfig

        config = ProviderConfig(
            provider="google",
            model="gemini-2.5-flash",
            api_key="test-key",
            max_tokens=10000,
            extra_params={
                "reasoning_config": {"effort": "medium"},
                "reasoning_effort": "medium",
            },
        )
        llm = LangChainLLM(config)

        with patch("langchain_google_genai.ChatGoogleGenerativeAI") as MockGoogle:
            MockGoogle.return_value = MagicMock()
            llm._create_chat_model()
            call_kwargs = MockGoogle.call_args[1]
            assert "thinking_config" not in call_kwargs
            assert "thinking_level" not in call_kwargs
            assert call_kwargs["thinking_budget"] > 0
            assert call_kwargs["include_thoughts"] is True

    @pytest.mark.unit
    def test_gemini_no_thinking_params_when_effort_none(self):
        from modules.llm.langchain_provider import LangChainLLM, ProviderConfig

        config = ProviderConfig(
            provider="google",
            model="gemini-3.1-flash-lite",
            api_key="test-key",
            max_tokens=10000,
            extra_params={
                "reasoning_config": {"effort": "none"},
                "reasoning_effort": "none",
            },
        )
        llm = LangChainLLM(config)

        with patch("langchain_google_genai.ChatGoogleGenerativeAI") as MockGoogle:
            MockGoogle.return_value = MagicMock()
            llm._create_chat_model()
            call_kwargs = MockGoogle.call_args[1]
            assert "thinking_level" not in call_kwargs
            assert "thinking_budget" not in call_kwargs
            assert "thinking_config" not in call_kwargs


# ---------------------------------------------------------------------------
# CM-3: Anthropic structured output degrades gracefully on complex schemas
# ---------------------------------------------------------------------------


class _FakeAIMessage:
    """Minimal stand-in for a LangChain AIMessage (no MagicMock truthiness)."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.usage_metadata = None
        self.response_metadata: dict = {}


class TestCM3AnthropicUnionLimitFallback:
    """A 400 for too many union-typed parameters must trigger the
    JSON-instructed fallback instead of failing the chunk."""

    UNION_LIMIT_400 = (
        "Error code: 400 - {'type': 'error', 'error': {'type': "
        "'invalid_request_error', 'message': 'Schemas contains too many "
        "parameters with union types (23 parameters with unions, limit: 16 "
        "parameters with unions)'}}"
    )

    @staticmethod
    def _make_llm(structured_error: Exception, plain_content: str):
        from modules.llm.langchain_provider import LangChainLLM, ProviderConfig

        config = ProviderConfig(
            provider="anthropic",
            model="claude-haiku-4-5-20251001",
            api_key="test-key",
            max_tokens=8000,
        )
        llm = LangChainLLM(config)

        structured_model = MagicMock()

        async def _raise(_messages):
            raise structured_error

        structured_model.ainvoke = _raise

        chat_model = MagicMock(spec=["with_structured_output", "ainvoke"])
        chat_model.with_structured_output.return_value = structured_model

        async def _plain(_messages):
            return _FakeAIMessage(plain_content)

        chat_model.ainvoke = _plain

        llm._chat_model = chat_model
        llm._initialized = True
        return llm

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_union_limit_400_falls_back_to_plain_invocation(self):
        llm = self._make_llm(
            Exception(self.UNION_LIMIT_400), '{"entries": [{"title": "x"}]}'
        )

        result = await llm.ainvoke_with_structured_output(
            [{"role": "user", "content": "extract"}],
            json_schema={"name": "BibliographicEntries", "schema": {"type": "object"}},
        )

        assert result["response_data"].get("structured_output_fallback") is True
        assert result["output_text"] == '{"entries": [{"title": "x"}]}'

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_unrelated_400_still_raises(self):
        """Only schema-complexity 400s trigger the fallback; other errors
        must propagate."""
        llm = self._make_llm(Exception("Error code: 429 - rate limited"), "{}")

        with pytest.raises(Exception, match="429"):
            await llm.ainvoke_with_structured_output(
                [{"role": "user", "content": "extract"}],
                json_schema={"name": "S", "schema": {"type": "object"}},
            )


# ---------------------------------------------------------------------------
# CM-5: mode-override flags on batch-ops parsers
# ---------------------------------------------------------------------------


class TestCM5ModeOverrideFlags:
    """check_batches, cancel_batches, and repair_extractions must accept the
    shared --interactive/--non-interactive flags."""

    @staticmethod
    def _parsers():
        from main.cli_args import (
            create_cancel_batches_parser,
            create_check_batches_parser,
            create_repair_parser,
        )

        return [
            create_check_batches_parser(),
            create_cancel_batches_parser(),
            create_repair_parser(),
        ]

    @pytest.mark.unit
    def test_non_interactive_accepted(self):
        for parser in self._parsers():
            args = parser.parse_args(["--non-interactive"])
            assert args.non_interactive is True
            assert args.interactive is False

    @pytest.mark.unit
    def test_interactive_accepted(self):
        for parser in self._parsers():
            args = parser.parse_args(["--interactive"])
            assert args.interactive is True
            assert args.non_interactive is False

    @pytest.mark.unit
    def test_flags_are_mutually_exclusive(self):
        for parser in self._parsers():
            with pytest.raises(SystemExit):
                parser.parse_args(["--interactive", "--non-interactive"])


# ---------------------------------------------------------------------------
# CM-6: finalized batch output lands with the submission, not schema default
# ---------------------------------------------------------------------------


class TestCM6SubmissionLocalFinalization:
    """Finalized outputs belong with the batch submission directory."""

    @staticmethod
    def _write_submission(tmp_path, name="doc"):
        """Create the default batch submission layout:
        <submission>/temp_jsonl/<name>_temp.jsonl with request + tracking."""
        submission = tmp_path / "submission"
        temp_dir = submission / "temp_jsonl"
        temp_dir.mkdir(parents=True)
        temp_file = temp_dir / f"{name}_temp.jsonl"
        source_file = submission / f"{name}.txt"
        lines = [
            {
                "batch_request": {
                    "custom_id": "req-0",
                    "order_index": 0,
                    "metadata": {"file_path": str(source_file)},
                }
            },
            {"batch_tracking": {"batch_id": "batch-123", "provider": "openai"}},
        ]
        temp_file.write_text(
            "\n".join(json.dumps(line, ensure_ascii=False) for line in lines) + "\n",
            encoding="utf-8",
        )
        return submission, temp_file

    @pytest.mark.unit
    def test_derive_submission_output_dir_layouts(self, tmp_path):
        """Derivation handles both temp_jsonl/ and flat submission layouts."""
        from modules.batch.ops import derive_submission_output_dir

        nested = tmp_path / "run" / "temp_jsonl" / "doc_temp.jsonl"
        flat = tmp_path / "run" / "doc_temp.jsonl"
        assert derive_submission_output_dir(nested) == tmp_path / "run"
        assert derive_submission_output_dir(flat) == tmp_path / "run"

    @pytest.mark.unit
    def test_check_batches_writes_output_next_to_submission(self, tmp_path):
        """A completed batch is finalized into the submission directory; the
        schema's default output directory is neither used nor created."""
        from main.check_batches import process_all_batches
        from modules.batch import BatchStatus, BatchStatusInfo

        submission, _temp_file = self._write_submission(tmp_path)
        schema_default = tmp_path / "schema_default"

        backend = MagicMock()
        backend.get_status.return_value = BatchStatusInfo(
            status=BatchStatus.COMPLETED, results_available=True
        )
        backend.cleanup.return_value = None

        agg: dict[str, int] = {
            "finalized": 0,
            "pending": 0,
            "failed": 0,
            "errors": 0,
        }

        with (
            patch("main.check_batches.get_batch_backend", return_value=backend),
            patch(
                "main.check_batches.retrieve_responses_from_batch",
                return_value=[{"custom_id": "req-0", "response": {"ok": True}}],
            ),
            patch(
                "main.check_batches.build_unified_batch_output",
                return_value={"records": [{"ok": True}]},
            ),
            patch("main.check_batches.get_schema_handler", return_value=MagicMock()),
        ):
            process_all_batches(
                root_folder=submission,
                processing_settings={"retain_temporary_jsonl": True},
                schema_name="BibliographicEntries",
                schema_config={"output": str(schema_default)},
                ui=None,
                agg=agg,
            )

        final_json = submission / "doc_output.json"
        assert final_json.exists(), "final output must land with the submission"
        assert not schema_default.exists(), (
            "the schema default output directory must not be used or created"
        )
        assert agg["finalized"] == 1

    @pytest.mark.unit
    def test_partial_finalization_keeps_remote_files(self, tmp_path):
        """A partial finalization (one batch completed, one missing) writes a
        partial output but must NOT delete remote result files, so
        repair_extractions can still retrieve the missing pieces."""
        from main.check_batches import process_all_batches
        from modules.batch import BatchHandle, BatchStatus, BatchStatusInfo

        submission = tmp_path / "submission"
        temp_dir = submission / "temp_jsonl"
        temp_dir.mkdir(parents=True)
        temp_file = temp_dir / "doc_temp.jsonl"
        source_file = submission / "doc.txt"
        lines = [
            {
                "batch_request": {
                    "custom_id": "doc-chunk-1",
                    "order_index": 1,
                    "metadata": {"file_path": str(source_file)},
                }
            },
            {"batch_tracking": {"batch_id": "batch-A", "provider": "openai"}},
            {"batch_tracking": {"batch_id": "batch-B", "provider": "openai"}},
        ]
        temp_file.write_text(
            "\n".join(json.dumps(line, ensure_ascii=False) for line in lines) + "\n",
            encoding="utf-8",
        )

        def _status(handle: BatchHandle) -> BatchStatusInfo:
            if handle.batch_id == "batch-A":
                return BatchStatusInfo(
                    status=BatchStatus.COMPLETED, results_available=True
                )
            raise RuntimeError("batch-B not found (expired/deleted)")

        backend = MagicMock()
        backend.get_status.side_effect = _status
        backend.cleanup.return_value = None

        agg: dict[str, int] = {
            "finalized": 0,
            "pending": 0,
            "failed": 0,
            "errors": 0,
        }

        with (
            patch("main.check_batches.get_batch_backend", return_value=backend),
            patch(
                "main.check_batches.retrieve_responses_from_batch",
                return_value=[{"custom_id": "doc-chunk-1", "response": {"ok": True}}],
            ),
            patch(
                "main.check_batches.build_unified_batch_output",
                return_value={"records": [{"custom_id": "doc-chunk-1"}]},
            ),
            patch("main.check_batches.get_schema_handler", return_value=MagicMock()),
        ):
            process_all_batches(
                root_folder=submission,
                processing_settings={"retain_temporary_jsonl": True},
                schema_name="BibliographicEntries",
                schema_config={"output": str(tmp_path / "schema_default")},
                ui=None,
                agg=agg,
            )

        assert (submission / "doc_output.json").exists(), (
            "a partial finalization must still persist the completed subset"
        )
        backend.cleanup.assert_not_called()
        assert agg["failed"] == 1 and agg["finalized"] == 0

    @pytest.mark.unit
    def test_schema_default_used_only_when_derivation_fails(self, tmp_path):
        """The schema default output directory is a last-resort fallback."""
        from main.check_batches import _resolve_group_output_dir

        schema_default = tmp_path / "schema_default"
        temp_file = tmp_path / "run" / "temp_jsonl" / "doc_temp.jsonl"

        # Normal case: derivation wins, schema default ignored.
        resolved = _resolve_group_output_dir(temp_file, {"output": str(schema_default)})
        assert resolved == tmp_path / "run"

        # Derivation failure: schema default is the fallback.
        with patch(
            "main.check_batches.derive_submission_output_dir",
            side_effect=OSError("pathological path"),
        ):
            resolved = _resolve_group_output_dir(
                temp_file, {"output": str(schema_default)}
            )
        assert resolved == schema_default

    @pytest.mark.unit
    def test_repair_writes_output_next_to_submission(self, tmp_path):
        """repair_extractions regenerates the final output in the submission
        directory (parent of temp_jsonl/), not inside temp_jsonl/."""
        from main.repair_extractions import _repair_temp_file
        from modules.batch import BatchStatus, BatchStatusInfo

        submission, temp_file = self._write_submission(tmp_path)

        backend = MagicMock()
        backend.get_status.return_value = BatchStatusInfo(
            status=BatchStatus.COMPLETED, results_available=True
        )

        candidate = {
            "temp_file": temp_file,
            "schema_name": "BibliographicEntries",
            "schema_config": {},
            "responses": [],
            "tracking": [{"batch_id": "batch-123", "provider": "openai"}],
        }

        with (
            patch("main.repair_extractions.get_batch_backend", return_value=backend),
            patch(
                "main.repair_extractions.retrieve_responses_from_batch",
                return_value=[{"custom_id": "req-0", "response": {"ok": True}}],
            ),
            patch(
                "main.repair_extractions.build_unified_batch_output",
                return_value={"records": [{"ok": True}]},
            ),
            patch(
                "main.repair_extractions.get_schema_handler",
                return_value=MagicMock(),
            ),
        ):
            _repair_temp_file(candidate, {}, ui=MagicMock())

        assert (submission / "doc_output.json").exists(), (
            "repaired output must land with the submission"
        )
        assert not (submission / "temp_jsonl" / "doc_output.json").exists(), (
            "repaired output must not be written inside temp_jsonl/"
        )
