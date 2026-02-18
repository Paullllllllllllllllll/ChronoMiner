"""
Coverage extension tests targeting previously under-tested modules.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ===========================================================================
# payload_builder.py
# ===========================================================================

class TestBuildStructuredTextFormat:
    @pytest.mark.unit
    def test_none_returns_none(self):
        from modules.operations.extraction.payload_builder import _build_structured_text_format
        assert _build_structured_text_format(None) is None  # type: ignore[arg-type]

    @pytest.mark.unit
    def test_empty_dict_returns_none(self):
        from modules.operations.extraction.payload_builder import _build_structured_text_format
        assert _build_structured_text_format({}) is None

    @pytest.mark.unit
    def test_wrapper_schema_object(self):
        from modules.operations.extraction.payload_builder import _build_structured_text_format
        result = _build_structured_text_format(
            {"name": "MySchema", "schema": {"type": "object", "properties": {}}, "strict": False}
        )
        assert result is not None
        assert result["type"] == "json_schema"
        assert result["name"] == "MySchema"
        assert result["strict"] is False

    @pytest.mark.unit
    def test_bare_json_schema_uses_default_name(self):
        from modules.operations.extraction.payload_builder import _build_structured_text_format
        bare = {"type": "object", "properties": {"a": {"type": "string"}}}
        result = _build_structured_text_format(bare, default_name="FallbackName")
        assert result is not None
        assert result["name"] == "FallbackName"
        assert result["schema"] == bare

    @pytest.mark.unit
    def test_wrapper_with_empty_schema_returns_none(self):
        from modules.operations.extraction.payload_builder import _build_structured_text_format
        assert _build_structured_text_format({"name": "X", "schema": {}}) is None

    @pytest.mark.unit
    def test_non_dict_returns_none(self):
        from modules.operations.extraction.payload_builder import _build_structured_text_format
        assert _build_structured_text_format("string") is None  # type: ignore[arg-type]


class TestPayloadBuilder:
    @pytest.mark.unit
    def test_basic_structure(self):
        from modules.operations.extraction.payload_builder import PayloadBuilder
        result = PayloadBuilder("S").build_payload(
            "text", "prompt",
            {"transcription_model": {"name": "gpt-4o", "max_output_tokens": 2048}},
            {}
        )
        assert result["method"] == "POST"
        assert result["body"]["model"] == "gpt-4o"

    @pytest.mark.unit
    def test_includes_schema_format(self):
        from modules.operations.extraction.payload_builder import PayloadBuilder
        schema = {"type": "object", "properties": {"title": {"type": "string"}}}
        result = PayloadBuilder("Bib").build_payload(
            "text", "prompt",
            {"transcription_model": {"name": "gpt-4o", "max_output_tokens": 512}},
            schema
        )
        assert result["body"]["text"]["format"]["name"] == "Bib"

    @pytest.mark.unit
    def test_sampler_controls_for_standard_model(self):
        from modules.operations.extraction.payload_builder import PayloadBuilder
        result = PayloadBuilder("S").build_payload(
            "text", "prompt",
            {"transcription_model": {
                "name": "gpt-4o", "max_output_tokens": 512,
                "temperature": 0.3, "top_p": 0.8,
                "frequency_penalty": 0.5, "presence_penalty": 0.2,
            }},
            {}
        )
        body = result["body"]
        assert body.get("temperature") == 0.3
        assert body.get("frequency_penalty") == 0.5

    @pytest.mark.unit
    def test_no_sampler_for_reasoning_model(self):
        from modules.operations.extraction.payload_builder import PayloadBuilder
        result = PayloadBuilder("S").build_payload(
            "text", "prompt",
            {"transcription_model": {"name": "o3-mini", "max_output_tokens": 16384, "temperature": 0.0}},
            {}
        )
        assert "temperature" not in result["body"]

    @pytest.mark.unit
    def test_build_json_schema_payload(self):
        from modules.operations.extraction.payload_builder import PayloadBuilder
        result = PayloadBuilder("MySchema")._build_json_schema_payload(
            "dev", {"transcription_model": {"name": "gpt-4o"}}, {"type": "object"}
        )
        assert result["name"] == "MySchema"
        assert result["strict"] is True


# ===========================================================================
# response_parser.py
# ===========================================================================

class TestResponseParser:
    @pytest.mark.unit
    def test_parse_valid_json(self):
        from modules.operations.extraction.response_parser import ResponseParser
        result = ResponseParser("S").parse_response('{"entries": [{"a": 1}]}')
        assert result == {"entries": [{"a": 1}]}

    @pytest.mark.unit
    def test_parse_invalid_json_returns_error(self):
        from modules.operations.extraction.response_parser import ResponseParser
        result = ResponseParser("S").parse_response("{bad}")
        assert "error" in result and "JSON decode error" in result["error"]

    @pytest.mark.unit
    def test_parse_none_returns_error(self):
        from modules.operations.extraction.response_parser import ResponseParser
        result = ResponseParser("S").parse_response(None)  # type: ignore[arg-type]
        assert "error" in result

    @pytest.mark.unit
    def test_validate_true_for_valid(self):
        from modules.operations.extraction.response_parser import ResponseParser
        assert ResponseParser("S").validate_response({"entries": []}) is True

    @pytest.mark.unit
    def test_validate_false_for_error_key(self):
        from modules.operations.extraction.response_parser import ResponseParser
        assert ResponseParser("S").validate_response({"error": "x"}) is False

    @pytest.mark.unit
    def test_validate_false_for_non_dict(self):
        from modules.operations.extraction.response_parser import ResponseParser
        assert ResponseParser("S").validate_response("str") is False  # type: ignore[arg-type]

    @pytest.mark.unit
    def test_extract_entries_standard(self):
        from modules.operations.extraction.response_parser import ResponseParser
        assert ResponseParser("S").extract_entries({"entries": [1, 2]}) == [1, 2]

    @pytest.mark.unit
    def test_extract_entries_no_key(self):
        from modules.operations.extraction.response_parser import ResponseParser
        assert ResponseParser("S").extract_entries({"data": []}) == []

    @pytest.mark.unit
    def test_extract_entries_on_error_response(self):
        from modules.operations.extraction.response_parser import ResponseParser
        assert ResponseParser("S").extract_entries({"error": "fail"}) == []

    @pytest.mark.unit
    def test_extract_entries_non_list_entries(self):
        from modules.operations.extraction.response_parser import ResponseParser
        assert ResponseParser("S").extract_entries({"entries": "not a list"}) == []


# ===========================================================================
# converter_base.py
# ===========================================================================

class TestResolveField:
    @pytest.mark.unit
    def test_flat_key(self):
        from modules.core.converter_base import resolve_field
        assert resolve_field({"a": 1}, "a") == 1

    @pytest.mark.unit
    def test_missing_flat_key_default(self):
        from modules.core.converter_base import resolve_field
        assert resolve_field({}, "x", default="N/A") == "N/A"

    @pytest.mark.unit
    def test_dotted_key(self):
        from modules.core.converter_base import resolve_field
        assert resolve_field({"address": {"street": "Main"}}, "address.street") == "Main"

    @pytest.mark.unit
    def test_dotted_key_outer_not_dict(self):
        from modules.core.converter_base import resolve_field
        assert resolve_field({"address": "flat"}, "address.street", default="N/A") == "N/A"


class TestBaseConverterHelpers:
    def _make(self, name="s"):
        from modules.core.converter_base import BaseConverter

        class Impl(BaseConverter):
            def convert(self, j, o): pass

        return Impl(name)

    @pytest.mark.unit
    def test_safe_str(self):
        impl = self._make()
        assert impl.safe_str(None) == ""
        assert impl.safe_str(42) == "42"

    @pytest.mark.unit
    def test_join_list(self):
        impl = self._make()
        assert impl.join_list(["a", "b"]) == "a, b"
        assert impl.join_list(["a", None, ""]) == "a"
        assert impl.join_list("not list") == ""
        assert impl.join_list(["x", "y"], separator="; ") == "x; y"

    @pytest.mark.unit
    def test_format_name_variants(self):
        impl = self._make()
        variants = [{"original": "Kochen", "modern_english": "Cooking"}, {"original": "Braten"}]
        result = impl.format_name_variants(variants)
        assert "Kochen (Cooking)" in result
        assert "Braten" in result

    @pytest.mark.unit
    def test_format_name_variants_non_list(self):
        assert self._make().format_name_variants(None) == ""

    @pytest.mark.unit
    def test_format_associations_string(self):
        impl = self._make()
        assocs = [{"target_type": "person", "target_label_modern_english": "John", "relationship": "author"}]
        result = impl.format_associations(assocs)
        assert "person" in result and "John" in result and "author" in result

    @pytest.mark.unit
    def test_format_associations_list(self):
        impl = self._make()
        result = impl.format_associations([{"target_type": "place", "target_label_original": "Paris"}], as_list=True)
        assert isinstance(result, list) and len(result) == 1

    @pytest.mark.unit
    def test_format_associations_non_list(self):
        impl = self._make()
        assert impl.format_associations(None) == ""
        assert impl.format_associations(None, as_list=True) == []

    @pytest.mark.unit
    def test_get_entries_filters_none(self, tmp_path):
        json_file = tmp_path / "data.json"
        json_file.write_text(json.dumps({"entries": [{"a": 1}, None, {"b": 2}]}))
        entries = self._make().get_entries(json_file)
        assert len(entries) == 2

    @pytest.mark.unit
    def test_get_entries_missing_file(self, tmp_path):
        assert self._make().get_entries(tmp_path / "missing.json") == []

    @pytest.mark.unit
    def test_get_converter(self):
        impl = self._make("myschema")
        fn = lambda x: x
        assert impl.get_converter({"myschema": fn}) is fn
        assert impl.get_converter({}) is None

    @pytest.mark.unit
    def test_schema_name_lowercased(self):
        assert self._make("MySchema").schema_name == "myschema"


# ===========================================================================
# workflow_utils.py
# ===========================================================================

class TestFilterTextFiles:
    @pytest.mark.unit
    def test_keeps_plain_txt(self, tmp_path):
        from modules.core.workflow_utils import filter_text_files
        f = tmp_path / "doc.txt"
        f.write_text("x")
        assert f in filter_text_files([f])

    @pytest.mark.unit
    def test_skips_non_txt(self, tmp_path):
        from modules.core.workflow_utils import filter_text_files
        f = tmp_path / "doc.pdf"
        f.write_text("x")
        assert filter_text_files([f]) == []

    @pytest.mark.unit
    def test_skips_context_suffix(self, tmp_path):
        from modules.core.workflow_utils import filter_text_files
        f = tmp_path / "doc_extraction.txt"
        f.write_text("x")
        assert filter_text_files([f]) == []

    @pytest.mark.unit
    def test_skips_line_ranges_suffix(self, tmp_path):
        from modules.core.workflow_utils import filter_text_files
        f = tmp_path / "doc_line_ranges.txt"
        f.write_text("x")
        assert filter_text_files([f]) == []

    @pytest.mark.unit
    def test_skips_directories(self, tmp_path):
        from modules.core.workflow_utils import filter_text_files
        d = tmp_path / "subdir"
        d.mkdir()
        assert filter_text_files([d]) == []


class TestCollectTextFiles:
    @pytest.mark.unit
    def test_collects_from_dir(self, tmp_path):
        from modules.core.workflow_utils import collect_text_files
        (tmp_path / "a.txt").write_text("x")
        (tmp_path / "b.txt").write_text("y")
        (tmp_path / "skip.pdf").write_text("z")
        names = {f.name for f in collect_text_files(tmp_path)}
        assert "a.txt" in names and "b.txt" in names and "skip.pdf" not in names

    @pytest.mark.unit
    def test_single_file(self, tmp_path):
        from modules.core.workflow_utils import collect_text_files
        f = tmp_path / "doc.txt"
        f.write_text("x")
        assert collect_text_files(f) == [f]

    @pytest.mark.unit
    def test_skips_auxiliary_files(self, tmp_path):
        from modules.core.workflow_utils import collect_text_files
        (tmp_path / "main.txt").write_text("x")
        (tmp_path / "main_line_ranges.txt").write_text("y")
        (tmp_path / "main_context.txt").write_text("z")
        result = collect_text_files(tmp_path)
        assert len(result) == 1 and result[0].name == "main.txt"


class TestValidateSchemaPaths:
    @pytest.mark.unit
    def test_valid_returns_true(self):
        from modules.core.workflow_utils import validate_schema_paths
        assert validate_schema_paths("S", {"S": {"input": "/in", "output": "/out"}}) is True

    @pytest.mark.unit
    def test_missing_schema_returns_false(self):
        from modules.core.workflow_utils import validate_schema_paths
        assert validate_schema_paths("Ghost", {}) is False

    @pytest.mark.unit
    def test_missing_output_returns_false(self):
        from modules.core.workflow_utils import validate_schema_paths
        assert validate_schema_paths("S", {"S": {"input": "/in", "output": ""}}) is False

    @pytest.mark.unit
    def test_missing_input_returns_false(self):
        from modules.core.workflow_utils import validate_schema_paths
        assert validate_schema_paths("S", {"S": {"input": None, "output": "/out"}}) is False

    @pytest.mark.unit
    def test_reports_via_ui(self):
        from modules.core.workflow_utils import validate_schema_paths
        ui = MagicMock()
        validate_schema_paths("Ghost", {}, ui=ui)
        ui.print_error.assert_called_once()


# ===========================================================================
# batch/backends/factory.py
# ===========================================================================

class TestGetBatchBackend:
    @pytest.mark.unit
    def test_openai_returned(self):
        from modules.llm.batch.backends.factory import get_batch_backend, clear_backend_cache
        clear_backend_cache()
        assert get_batch_backend("openai").provider_name == "openai"

    @pytest.mark.unit
    def test_anthropic_returned(self):
        from modules.llm.batch.backends.factory import get_batch_backend, clear_backend_cache
        clear_backend_cache()
        assert get_batch_backend("anthropic").provider_name == "anthropic"

    @pytest.mark.unit
    def test_google_returned(self):
        from modules.llm.batch.backends.factory import get_batch_backend, clear_backend_cache
        clear_backend_cache()
        assert get_batch_backend("google").provider_name == "google"

    @pytest.mark.unit
    def test_openrouter_raises(self):
        from modules.llm.batch.backends.factory import get_batch_backend, clear_backend_cache
        clear_backend_cache()
        with pytest.raises(ValueError, match="OpenRouter"):
            get_batch_backend("openrouter")

    @pytest.mark.unit
    def test_unknown_raises(self):
        from modules.llm.batch.backends.factory import get_batch_backend, clear_backend_cache
        clear_backend_cache()
        with pytest.raises(ValueError, match="Unknown provider"):
            get_batch_backend("fakecloud")

    @pytest.mark.unit
    def test_none_provider_auto_detects_from_config(self):
        """When provider=None, factory auto-detects from model config and returns a backend."""
        from modules.llm.batch.backends.factory import get_batch_backend, clear_backend_cache
        clear_backend_cache()
        # conftest injects gpt-4o which auto-detects as openai
        backend = get_batch_backend(None)
        assert backend.provider_name == "openai"

    @pytest.mark.unit
    def test_none_provider_raises_when_provider_unknown(self):
        """When model name resolves to unknown provider, ValueError is raised."""
        from modules.llm.batch.backends.factory import get_batch_backend, clear_backend_cache
        clear_backend_cache()
        with patch("modules.config.loader.get_config_loader") as mock_loader:
            mock_loader.return_value.get_model_config.return_value = {
                "transcription_model": {"name": ""}
            }
            with pytest.raises(ValueError):
                get_batch_backend(None)

    @pytest.mark.unit
    def test_cached_instance_reused(self):
        from modules.llm.batch.backends.factory import get_batch_backend, clear_backend_cache
        clear_backend_cache()
        b1 = get_batch_backend("openai")
        b2 = get_batch_backend("openai")
        assert b1 is b2

    @pytest.mark.unit
    def test_clear_cache_creates_new_instance(self):
        from modules.llm.batch.backends.factory import get_batch_backend, clear_backend_cache
        clear_backend_cache()
        b1 = get_batch_backend("openai")
        clear_backend_cache()
        b2 = get_batch_backend("openai")
        assert b1 is not b2

    @pytest.mark.unit
    def test_supports_batch(self):
        from modules.llm.batch.backends.factory import supports_batch
        assert supports_batch("openai") is True
        assert supports_batch("anthropic") is True
        assert supports_batch("google") is True
        assert supports_batch("openrouter") is False
        assert supports_batch("OpenAI") is True


# ===========================================================================
# model_capabilities.py — OpenRouter branch
# ===========================================================================

class TestDetectCapabilitiesOpenRouter:
    @pytest.mark.unit
    def test_deepseek_r1_is_reasoning(self):
        from modules.llm.model_capabilities import detect_capabilities
        caps = detect_capabilities("deepseek/deepseek-r1")
        assert caps.is_reasoning_model is True and caps.supports_reasoning_effort is True

    @pytest.mark.unit
    def test_deepseek_non_r1_not_reasoning(self):
        from modules.llm.model_capabilities import detect_capabilities
        assert detect_capabilities("deepseek/deepseek-chat").is_reasoning_model is False

    @pytest.mark.unit
    def test_gpt5_via_openrouter(self):
        from modules.llm.model_capabilities import detect_capabilities
        caps = detect_capabilities("openrouter/gpt-5")
        assert caps.is_reasoning_model is True and caps.supports_reasoning_effort is True

    @pytest.mark.unit
    def test_o3_via_openrouter(self):
        from modules.llm.model_capabilities import detect_capabilities
        assert detect_capabilities("openai/o3").is_reasoning_model is True

    @pytest.mark.unit
    def test_claude_via_openrouter(self):
        from modules.llm.model_capabilities import detect_capabilities
        caps = detect_capabilities("anthropic/claude-sonnet-4-5")
        assert caps.provider == "openrouter" and caps.is_reasoning_model is True

    @pytest.mark.unit
    def test_gemini_thinking_via_openrouter(self):
        from modules.llm.model_capabilities import detect_capabilities
        assert detect_capabilities("google/gemini-2.5-flash").is_reasoning_model is True

    @pytest.mark.unit
    def test_llama_via_openrouter(self):
        from modules.llm.model_capabilities import detect_capabilities
        caps = detect_capabilities("meta/llama-3.2-90b")
        assert caps.provider == "openrouter" and caps.family == "openrouter-llama"

    @pytest.mark.unit
    def test_mistral_detected_via_underlying_name(self):
        """openrouter/mistral-large matches the mistral branch."""
        from modules.llm.model_capabilities import detect_capabilities
        caps = detect_capabilities("openrouter/mistral-large")
        assert caps.family == "openrouter-mistral"

    @pytest.mark.unit
    def test_pixtral_image_support_via_mixtral_path(self):
        """Model with 'mixtral' in name and 'pixtral' in name gets image support."""
        from modules.llm.model_capabilities import detect_capabilities
        # mixtral in m → mistral branch; pixtral in m → supports_image_input=True
        caps = detect_capabilities("openrouter/mixtral-pixtral")
        assert caps.family == "openrouter-mistral"
        assert caps.supports_image_input is True

    @pytest.mark.unit
    def test_generic_openrouter_fallback(self):
        from modules.llm.model_capabilities import detect_capabilities
        caps = detect_capabilities("somevendor/some-model")
        assert caps.provider == "openrouter" and caps.family == "openrouter"


# ===========================================================================
# langchain_provider.py — OpenRouter and error paths
# ===========================================================================

class TestLangChainProviderAdditional:
    @pytest.mark.unit
    def test_openrouter_model_prefix_stripped(self):
        from modules.llm.langchain_provider import ProviderConfig, LangChainLLM
        config = ProviderConfig(
            provider="openrouter",
            model="openrouter/mistral/pixtral-large",
            api_key="test-key",
            extra_params={},
        )
        with patch("langchain_openai.ChatOpenAI") as Mock:
            Mock.return_value = MagicMock()
            LangChainLLM(config)._create_chat_model()
            assert not Mock.call_args[1]["model"].startswith("openrouter/")

    @pytest.mark.unit
    def test_openrouter_reasoning_payload_injected(self):
        from modules.llm.langchain_provider import ProviderConfig, LangChainLLM
        config = ProviderConfig(
            provider="openrouter",
            model="openrouter/deepseek-r1",
            api_key="test-key",
            max_tokens=10000,
            extra_params={"reasoning_config": {"effort": "medium"}, "reasoning_effort": "medium"},
        )
        with patch("langchain_openai.ChatOpenAI") as Mock:
            Mock.return_value = MagicMock()
            LangChainLLM(config)._create_chat_model()
            kwargs = Mock.call_args[1]
            assert "model_kwargs" in kwargs
            assert "reasoning" in kwargs["model_kwargs"]["extra_body"]

    @pytest.mark.unit
    def test_unsupported_provider_raises(self):
        from modules.llm.langchain_provider import ProviderConfig, LangChainLLM
        config = ProviderConfig(provider="unknown", model="m", api_key="k")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unsupported provider"):
            LangChainLLM(config)._create_chat_model()

    @pytest.mark.unit
    def test_provider_config_from_config_loads_text_config(self):
        from modules.llm.langchain_provider import ProviderConfig
        model_config = {
            "transcription_model": {
                "name": "gpt-5",
                "max_output_tokens": 4096,
                "text": {"verbosity": "high"},
            }
        }
        with patch.dict("os.environ", {"OPENAI_API_KEY": "k"}):
            config = ProviderConfig.from_config(model_config)
        assert config.extra_params.get("text_config") == {"verbosity": "high"}
