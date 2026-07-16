"""Tests for multi-provider batch backends."""

import json
from unittest.mock import MagicMock, patch

import pytest

from modules.batch import (
    BatchHandle,
    BatchRequest,
    BatchResultItem,
    BatchStatus,
    BatchStatusInfo,
    clear_backend_cache,
    get_batch_backend,
    supports_batch,
)
from modules.config.capabilities import detect_capabilities


class TestBatchDataClasses:
    """Test batch data classes."""

    def test_batch_request_is_visual_false_for_text(self):
        """BatchRequest without image_base64 reports is_visual=False."""
        req = BatchRequest(custom_id="req-1", text="some text")
        assert req.is_visual is False

    def test_batch_request_is_visual_true_when_image_set(self):
        """BatchRequest with image_base64 reports is_visual=True."""
        req = BatchRequest(
            custom_id="req-1",
            image_base64="abc123",
            mime_type="image/png",
            image_detail="low",
        )
        assert req.is_visual is True
        assert req.mime_type == "image/png"
        assert req.image_detail == "low"

    def test_batch_request_visual_fields_default_to_none(self):
        """Visual fields default to None for text requests."""
        req = BatchRequest(custom_id="req-1", text="text")
        assert req.image_base64 is None
        assert req.mime_type is None
        assert req.image_detail is None

    def test_batch_handle_serialization(self):
        """Test BatchHandle to_dict and from_dict."""
        handle = BatchHandle(
            provider="openai", batch_id="batch_123", metadata={"key": "value"}
        )

        data = handle.to_dict()
        assert data["provider"] == "openai"
        assert data["batch_id"] == "batch_123"
        assert data["metadata"] == {"key": "value"}

        restored = BatchHandle.from_dict(data)
        assert restored.provider == handle.provider
        assert restored.batch_id == handle.batch_id
        assert restored.metadata == handle.metadata

    def test_batch_request(self):
        """Test BatchRequest creation."""
        req = BatchRequest(
            custom_id="req-1",
            text="Sample text",
            order_index=1,
            metadata={"file": "test.txt"},
        )

        assert req.custom_id == "req-1"
        assert req.text == "Sample text"
        assert req.order_index == 1
        assert req.metadata["file"] == "test.txt"

    def test_batch_result_item_properties(self):
        """Test BatchResultItem computed properties."""
        # Result with entries
        result_with_entries = BatchResultItem(
            custom_id="req-1",
            success=True,
            parsed_output={"entries": [{"name": "Test"}]},
        )
        assert result_with_entries.has_entries is True
        assert result_with_entries.contains_no_content is False

        # Result with no content flag
        result_no_content = BatchResultItem(
            custom_id="req-2",
            success=True,
            parsed_output={"contains_no_content_of_requested_type": True},
        )
        assert result_no_content.has_entries is False
        assert result_no_content.contains_no_content is True

        # Empty result
        empty_result = BatchResultItem(custom_id="req-3")
        assert empty_result.has_entries is False
        assert empty_result.contains_no_content is False

    def test_batch_status_info(self):
        """Test BatchStatusInfo creation."""
        status = BatchStatusInfo(
            status=BatchStatus.COMPLETED,
            total_requests=10,
            completed_requests=10,
            failed_requests=0,
            results_available=True,
            output_file_id="file-123",
        )

        assert status.status == BatchStatus.COMPLETED
        assert status.total_requests == 10
        assert status.results_available is True


class TestBatchFactory:
    """Test batch backend factory functions."""

    def setup_method(self):
        """Clear backend cache before each test."""
        clear_backend_cache()

    def test_supports_batch(self):
        """Test supports_batch function."""
        assert supports_batch("openai") is True
        assert supports_batch("anthropic") is True
        assert supports_batch("google") is True
        assert supports_batch("openrouter") is False
        assert supports_batch("unknown") is False

        # Case insensitive
        assert supports_batch("OpenAI") is True
        assert supports_batch("ANTHROPIC") is True

    def test_get_batch_backend_openai(self):
        """Test getting OpenAI backend."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            backend = get_batch_backend("openai")
            assert backend.provider_name == "openai"
            assert backend.max_batch_size == 50000

    def test_get_batch_backend_anthropic(self):
        """Test getting Anthropic backend."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            backend = get_batch_backend("anthropic")
            assert backend.provider_name == "anthropic"
            assert backend.max_batch_size == 100000

    def test_get_batch_backend_google(self):
        """Test getting Google backend."""
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            backend = get_batch_backend("google")
            assert backend.provider_name == "google"
            assert backend.max_batch_size == 50000

    def test_get_batch_backend_openrouter_raises(self):
        """Test that OpenRouter raises ValueError."""
        with pytest.raises(ValueError, match="OpenRouter does not support batch"):
            get_batch_backend("openrouter")

    def test_get_batch_backend_unknown_raises(self):
        """Test that unknown provider raises ValueError."""
        with pytest.raises(ValueError, match="Unknown provider"):
            get_batch_backend("unknown_provider")

    def test_backend_caching(self):
        """Test that backends are cached."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            backend1 = get_batch_backend("openai")
            backend2 = get_batch_backend("openai")
            assert backend1 is backend2

    def test_clear_backend_cache(self):
        """Test clearing backend cache."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            backend1 = get_batch_backend("openai")
            clear_backend_cache()
            backend2 = get_batch_backend("openai")
            assert backend1 is not backend2


class TestOpenAIBackend:
    """Test OpenAI batch backend."""

    def setup_method(self):
        clear_backend_cache()

    @patch("openai.OpenAI")
    def test_submit_batch(self, mock_openai_class):
        """Test batch submission."""
        # Setup mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_file_response = MagicMock()
        mock_file_response.id = "file-123"
        mock_client.files.create.return_value = mock_file_response

        mock_batch_response = MagicMock()
        mock_batch_response.id = "batch-456"
        mock_client.batches.create.return_value = mock_batch_response

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            backend = get_batch_backend("openai")

            requests = [
                BatchRequest(custom_id="req-1", text="Sample text 1", order_index=1),
                BatchRequest(custom_id="req-2", text="Sample text 2", order_index=2),
            ]

            model_config = {
                "extraction_model": {
                    "name": "gpt-4o",
                    "max_output_tokens": 4096,
                }
            }

            handle = backend.submit_batch(
                requests,
                model_config,
                system_prompt="Extract data from text.",
                schema={"type": "object"},
                schema_name="TestSchema",
            )

            assert handle.provider == "openai"
            assert handle.batch_id == "batch-456"
            assert handle.metadata["input_file_id"] == "file-123"
            assert handle.metadata["request_count"] == 2

    @patch("openai.OpenAI")
    def test_get_status(self, mock_openai_class):
        """Test getting batch status."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_batch = MagicMock()
        mock_batch.status = "completed"
        mock_batch.request_counts = MagicMock()
        mock_batch.request_counts.total = 10
        mock_batch.request_counts.completed = 10
        mock_batch.request_counts.failed = 0
        mock_batch.output_file_id = "output-file-123"
        mock_client.batches.retrieve.return_value = mock_batch

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            backend = get_batch_backend("openai")
            handle = BatchHandle(provider="openai", batch_id="batch-123")

            status = backend.get_status(handle)

            assert status.status == BatchStatus.COMPLETED
            assert status.total_requests == 10
            assert status.completed_requests == 10
            assert status.results_available is True

    @patch("openai.OpenAI")
    def test_download_results_does_not_delete_but_cleanup_does(self, mock_openai_class):
        """Deletion moved out of download_results (Bug 7): iterating results must
        NOT delete remote files; cleanup(handle) deletes both input and output
        only after the caller has durably written the final output JSON."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_batch = MagicMock()
        mock_batch.output_file_id = "output-file-1"
        mock_client.batches.retrieve.return_value = mock_batch

        result_line = json.dumps(
            {
                "custom_id": "req-1",
                "response": {
                    "status_code": 200,
                    "body": {
                        "output": [
                            {
                                "type": "message",
                                "content": [
                                    {"type": "output_text", "text": '{"ok": true}'}
                                ],
                            }
                        ]
                    },
                },
            }
        )
        mock_stream = MagicMock()
        mock_stream.read.return_value = result_line.encode("utf-8")
        mock_client.files.content.return_value = mock_stream

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            backend = get_batch_backend("openai")
            handle = BatchHandle(
                provider="openai",
                batch_id="batch-123",
                metadata={"input_file_id": "input-file-1"},
            )
            results = list(backend.download_results(handle))
            # Nothing deleted during download.
            assert mock_client.files.delete.call_count == 0
            backend.cleanup(handle)

        assert len(results) == 1
        deleted = {call.args[0] for call in mock_client.files.delete.call_args_list}
        assert deleted == {"input-file-1", "output-file-1"}

    @patch("openai.OpenAI")
    def test_cleanup_survives_delete_failure(self, mock_openai_class):
        """Regression (A5): a delete failure during cleanup must not raise."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_batch = MagicMock()
        mock_batch.output_file_id = "output-file-1"
        mock_client.batches.retrieve.return_value = mock_batch

        mock_stream = MagicMock()
        mock_stream.read.return_value = json.dumps(
            {
                "custom_id": "req-1",
                "response": {
                    "status_code": 200,
                    "body": {
                        "output": [
                            {
                                "type": "message",
                                "content": [
                                    {"type": "output_text", "text": '{"ok": true}'}
                                ],
                            }
                        ]
                    },
                },
            }
        ).encode("utf-8")
        mock_client.files.content.return_value = mock_stream
        mock_client.files.delete.side_effect = RuntimeError("delete boom")

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            backend = get_batch_backend("openai")
            handle = BatchHandle(
                provider="openai",
                batch_id="batch-123",
                metadata={"input_file_id": "input-file-1"},
            )
            results = list(backend.download_results(handle))
            backend.cleanup(handle)  # must not raise

        assert len(results) == 1
        assert results[0].success is True


class TestAnthropicBackend:
    """Test Anthropic batch backend."""

    def setup_method(self):
        clear_backend_cache()

    @patch("anthropic.Anthropic")
    def test_get_status_in_progress(self, mock_anthropic_class):
        """Test getting in-progress batch status."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        mock_batch = MagicMock()
        mock_batch.processing_status = "in_progress"
        mock_batch.request_counts = MagicMock()
        mock_batch.request_counts.processing = 5
        mock_batch.request_counts.succeeded = 3
        mock_batch.request_counts.errored = 0
        mock_batch.request_counts.canceled = 0
        mock_batch.request_counts.expired = 0
        mock_batch.results_url = None
        mock_client.messages.batches.retrieve.return_value = mock_batch

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            backend = get_batch_backend("anthropic")
            handle = BatchHandle(provider="anthropic", batch_id="batch-123")

            status = backend.get_status(handle)

            assert status.status == BatchStatus.IN_PROGRESS
            assert status.completed_requests == 3
            assert status.pending_requests == 5

    @patch("anthropic.Anthropic")
    def test_get_status_ended_without_counts_is_unknown(self, mock_anthropic_class):
        """Regression (C6): an ``ended`` batch with absent/all-zero
        request_counts must map to UNKNOWN, not FAILED. With total == 0 the
        old `errored == total` check (0 == 0) mislabeled it FAILED."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        mock_batch = MagicMock()
        mock_batch.processing_status = "ended"
        mock_batch.request_counts = None
        mock_batch.results_url = None
        mock_client.messages.batches.retrieve.return_value = mock_batch

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            backend = get_batch_backend("anthropic")
            handle = BatchHandle(provider="anthropic", batch_id="batch-123")

            status = backend.get_status(handle)

            assert status.status == BatchStatus.UNKNOWN


class TestGoogleBackend:
    """Test Google batch backend."""

    def setup_method(self):
        clear_backend_cache()

    def test_get_status_completed(self):
        """Test getting completed batch status."""
        import sys

        # Create mock genai module
        mock_genai = MagicMock()
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client

        mock_batch_job = MagicMock()
        mock_batch_job.state = MagicMock()
        mock_batch_job.state.name = "JOB_STATE_SUCCEEDED"
        mock_batch_job.dest = MagicMock()
        mock_batch_job.dest.file_name = "output-file"
        mock_client.batches.get.return_value = mock_batch_job

        # Mock the google.genai import path
        mock_google = MagicMock()
        mock_google.genai = mock_genai

        with (
            patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}),
            patch.dict(
                sys.modules, {"google": mock_google, "google.genai": mock_genai}
            ),
        ):
            backend = get_batch_backend("google")
            handle = BatchHandle(
                provider="google",
                batch_id="batch-123",
                metadata={"request_count": 5},
            )

            status = backend.get_status(handle)

            assert status.status == BatchStatus.COMPLETED
            assert status.results_available is True

    def test_download_results_empty_candidate_marks_failure(self):
        """Regression (C7): a candidate with no text parts (e.g. a SAFETY or
        MAX_TOKENS block) must be reported as a failure, not a successful empty
        extraction that silently corrupts downstream aggregation."""
        import json
        import sys

        mock_genai = MagicMock()
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client

        mock_batch_job = MagicMock()
        mock_dest = MagicMock()
        mock_dest.file_name = "result-file"
        mock_batch_job.dest = mock_dest
        mock_client.batches.get.return_value = mock_batch_job

        result_line = json.dumps(
            {
                "key": "doc-chunk-1",
                "response": {
                    "candidates": [{"finishReason": "SAFETY", "content": {"parts": []}}]
                },
            }
        )
        mock_client.files.download.return_value = result_line.encode("utf-8")

        mock_google = MagicMock()
        mock_google.genai = mock_genai

        with (
            patch.dict("os.environ", {"GOOGLE_API_KEY": "test"}),
            patch.dict(
                sys.modules, {"google": mock_google, "google.genai": mock_genai}
            ),
        ):
            backend = get_batch_backend("google")
            handle = BatchHandle(provider="google", batch_id="batch-123")

            results = list(backend.download_results(handle))

        assert len(results) == 1
        assert results[0].success is False
        assert "SAFETY" in (results[0].error or "")

    def test_submit_batch_wires_sanitized_response_schema(self):
        """Regression (A4): the structured-output schema must reach Gemini's
        generation_config as response_schema/response_mime_type, with keys
        Gemini rejects (e.g. additionalProperties, $ref) stripped."""
        import sys

        mock_genai = MagicMock()
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client

        captured = {}

        def _capture_create(*_args, **kwargs):
            captured["src"] = kwargs.get("src")
            return MagicMock()

        mock_client.batches.create.side_effect = _capture_create

        mock_google = MagicMock()
        mock_google.genai = mock_genai

        schema = {
            "name": "TestSchema",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "ref_field": {"$ref": "#/$defs/Thing"},
                    "name": {"type": "string", "title": "Name"},
                },
                "$defs": {"Thing": {"type": "string"}},
            },
        }

        with (
            patch.dict("os.environ", {"GOOGLE_API_KEY": "test"}),
            patch.dict(
                sys.modules, {"google": mock_google, "google.genai": mock_genai}
            ),
        ):
            backend = get_batch_backend("google")
            backend.submit_batch(
                [BatchRequest(custom_id="req-1", text="hello", order_index=1)],
                {"extraction_model": {"name": "gemini-2.5-flash"}},
                system_prompt="Extract data.",
                schema=schema,
                schema_name="TestSchema",
            )

        # Inline requests carry generation params inside "config"
        # (GenerateContentConfig shape), with the system prompt folded in.
        gen_config = captured["src"][0]["config"]
        assert gen_config["response_mime_type"] == "application/json"
        assert gen_config["system_instruction"] == "Extract data."
        response_schema = gen_config["response_schema"]
        assert "additionalProperties" not in response_schema
        assert "$defs" not in response_schema
        assert "title" not in response_schema["properties"]["name"]
        # The forbidden $ref is stripped from the nested property.
        assert "$ref" not in response_schema["properties"]["ref_field"]

    def test_submit_batch_skips_schema_when_unsupported(self):
        """Regression (A4): when the model lacks structured-output support, no
        response_schema is set (Gemini falls back to prompt-guided JSON)."""
        import sys

        mock_genai = MagicMock()
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client

        captured = {}

        def _capture_create(*_args, **kwargs):
            captured["src"] = kwargs.get("src")
            return MagicMock()

        mock_client.batches.create.side_effect = _capture_create

        mock_google = MagicMock()
        mock_google.genai = mock_genai

        unsupported = MagicMock()
        unsupported.supports_structured_outputs = False

        with (
            patch.dict("os.environ", {"GOOGLE_API_KEY": "test"}),
            patch.dict(
                sys.modules, {"google": mock_google, "google.genai": mock_genai}
            ),
            patch(
                "modules.batch.backends.google_backend.detect_capabilities",
                return_value=unsupported,
            ),
        ):
            backend = get_batch_backend("google")
            backend.submit_batch(
                [BatchRequest(custom_id="req-1", text="hello", order_index=1)],
                {"extraction_model": {"name": "gemini-2.5-flash"}},
                system_prompt="Extract data.",
                schema={"schema": {"type": "object"}},
                schema_name="TestSchema",
            )

        gen_config = captured["src"][0]["config"]
        assert gen_config is None or "response_schema" not in gen_config

    def test_cleanup_deletes_remote_files(self):
        """Deletion moved out of download_results (Bug 7): iterating results must
        NOT delete remote files; cleanup(handle) deletes both input and result
        files only after the final output JSON is durably written."""
        import json
        import sys

        mock_genai = MagicMock()
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client

        mock_batch_job = MagicMock()
        mock_dest = MagicMock()
        mock_dest.file_name = "files/result-file"
        mock_batch_job.dest = mock_dest
        mock_client.batches.get.return_value = mock_batch_job

        result_line = json.dumps(
            {
                "key": "doc-chunk-1",
                "response": {
                    "candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}}]
                },
            }
        )
        mock_client.files.download.return_value = result_line.encode("utf-8")

        mock_google = MagicMock()
        mock_google.genai = mock_genai

        with (
            patch.dict("os.environ", {"GOOGLE_API_KEY": "test"}),
            patch.dict(
                sys.modules, {"google": mock_google, "google.genai": mock_genai}
            ),
        ):
            backend = get_batch_backend("google")
            handle = BatchHandle(
                provider="google",
                batch_id="batch-123",
                metadata={"input_file_name": "files/input-file"},
            )
            results = list(backend.download_results(handle))
            assert mock_client.files.delete.call_count == 0
            backend.cleanup(handle)

        assert len(results) == 1
        deleted = {
            call.kwargs.get("name") for call in mock_client.files.delete.call_args_list
        }
        assert deleted == {"files/input-file", "files/result-file"}

    def test_cleanup_survives_delete_failure(self):
        """Regression (A5): a failure deleting a remote file during cleanup must
        not crash the run; results are still yielded."""
        import json
        import sys

        mock_genai = MagicMock()
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client

        mock_batch_job = MagicMock()
        mock_dest = MagicMock()
        mock_dest.file_name = "files/result-file"
        mock_batch_job.dest = mock_dest
        mock_client.batches.get.return_value = mock_batch_job
        mock_client.files.download.return_value = json.dumps(
            {
                "key": "doc-chunk-1",
                "response": {
                    "candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}}]
                },
            }
        ).encode("utf-8")
        mock_client.files.delete.side_effect = RuntimeError("delete boom")

        mock_google = MagicMock()
        mock_google.genai = mock_genai

        with (
            patch.dict("os.environ", {"GOOGLE_API_KEY": "test"}),
            patch.dict(
                sys.modules, {"google": mock_google, "google.genai": mock_genai}
            ),
        ):
            backend = get_batch_backend("google")
            handle = BatchHandle(
                provider="google",
                batch_id="batch-123",
                metadata={"input_file_name": "files/input-file"},
            )
            results = list(backend.download_results(handle))
            backend.cleanup(handle)  # must not raise

        assert len(results) == 1
        assert results[0].success is True


class TestOpenAIImageResponsesBody:
    """Test _build_image_responses_body helper for OpenAI visual batch requests."""

    def test_builds_input_image_content_block(self):
        """Output body contains input_image block with correct data URL and detail."""
        from modules.batch.backends.openai_backend import _build_image_responses_body

        model_config = {
            "extraction_model": {"name": "gpt-4o", "max_output_tokens": 1024}
        }
        body = _build_image_responses_body(
            model_config=model_config,
            system_prompt="Extract recipes.",
            image_base64="AAAA",
            mime_type="image/png",
            image_detail="low",
        )

        assert body["model"] == "gpt-4o"
        # Check user turn contains input_image block
        user_turn = body["input"][1]
        assert user_turn["role"] == "user"
        assert len(user_turn["content"]) == 1
        img_block = user_turn["content"][0]
        assert img_block["type"] == "input_image"
        assert img_block["detail"] == "low"
        assert img_block["image_url"] == "data:image/png;base64,AAAA"

    def test_defaults_detail_to_auto(self):
        """When image_detail is None, body uses 'auto'."""
        from modules.batch.backends.openai_backend import _build_image_responses_body

        model_config = {
            "extraction_model": {"name": "gpt-4o", "max_output_tokens": 1024}
        }
        body = _build_image_responses_body(
            model_config=model_config,
            system_prompt="sys",
            image_base64="B64",
            mime_type="image/jpeg",
            image_detail=None,
        )

        img_block = body["input"][1]["content"][0]
        assert img_block["detail"] == "auto"

    def test_service_tier_flex_converted_to_auto(self):
        """service_tier='flex' is converted to 'auto' for batch API."""
        from modules.batch.backends.openai_backend import _build_image_responses_body

        model_config = {
            "extraction_model": {
                "name": "gpt-4o",
                "max_output_tokens": 512,
                "service_tier": "flex",
            }
        }
        body = _build_image_responses_body(
            model_config=model_config,
            system_prompt="sys",
            image_base64="X",
            mime_type="image/png",
        )

        assert body.get("service_tier") == "auto"


class TestOpenAIVisualBatchRouting:
    """Test that submit_batch routes visual requests through
    _build_image_responses_body."""

    def setup_method(self):
        clear_backend_cache()

    @patch("openai.OpenAI")
    def test_submit_visual_batch_uses_input_image(self, mock_openai_class):
        """Visual BatchRequests produce input_image content blocks in the JSONL."""
        import json

        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_file_response = MagicMock()
        mock_file_response.id = "file-vis"
        mock_client.files.create.return_value = mock_file_response

        mock_batch_response = MagicMock()
        mock_batch_response.id = "batch-vis"
        mock_client.batches.create.return_value = mock_batch_response

        captured_jsonl: list = []

        def _capture_file(file, purpose):
            content = file.read().decode("utf-8")
            captured_jsonl.extend(content.strip().split("\n"))
            return mock_file_response

        mock_client.files.create.side_effect = _capture_file

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test"}):
            backend = get_batch_backend("openai")
            requests = [
                BatchRequest(
                    custom_id="stem-page-1",
                    image_base64="IMGDATA",
                    mime_type="image/jpeg",
                    image_detail="high",
                    order_index=1,
                )
            ]
            model_config = {
                "extraction_model": {"name": "gpt-4o", "max_output_tokens": 512}
            }
            backend.submit_batch(
                requests,
                model_config,
                system_prompt="sys",
            )

        assert len(captured_jsonl) == 1
        line = json.loads(captured_jsonl[0])
        user_content = line["body"]["input"][1]["content"]
        assert user_content[0]["type"] == "input_image"
        assert "IMGDATA" in user_content[0]["image_url"]


class TestAnthropicVisualBatchRouting:
    """Test Anthropic backend routes visual requests with image source blocks."""

    def setup_method(self):
        clear_backend_cache()

    @patch("anthropic.Anthropic")
    def test_submit_visual_batch_uses_image_source_block(self, mock_anthropic_class):
        """Visual BatchRequests produce Anthropic image source blocks."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        mock_batch_response = MagicMock()
        mock_batch_response.id = "anthr-batch-1"
        mock_client.messages.batches.create.return_value = mock_batch_response

        captured_requests: list = []

        def _capture_create(requests):
            captured_requests.extend(requests)
            return mock_batch_response

        mock_client.messages.batches.create.side_effect = _capture_create

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}):
            backend = get_batch_backend("anthropic")
            requests = [
                BatchRequest(
                    custom_id="stem-page-1",
                    image_base64="ANTHRIMG",
                    mime_type="image/png",
                    order_index=1,
                )
            ]
            model_config = {
                "extraction_model": {
                    "name": "claude-sonnet-4-20250514",
                    "max_tokens": 512,
                }
            }
            backend.submit_batch(requests, model_config, system_prompt="sys")

        assert len(captured_requests) == 1
        user_content = captured_requests[0]["params"]["messages"][0]["content"]
        # First block is a text prompt, second is the image source block
        image_blocks = [b for b in user_content if b.get("type") == "image"]
        assert len(image_blocks) == 1
        source = image_blocks[0]["source"]
        assert source["type"] == "base64"
        assert source["media_type"] == "image/png"
        assert source["data"] == "ANTHRIMG"


class TestGoogleVisualBatchRouting:
    """Test Google backend routes visual requests with inline_data blocks."""

    def setup_method(self):
        clear_backend_cache()

    def test_submit_visual_batch_uses_inline_data(self):
        """Visual BatchRequests produce inline_data parts in the Google request."""
        import sys

        mock_genai = MagicMock()
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client

        mock_batch_job = MagicMock()
        mock_batch_job.name = "batches/google-1"
        mock_client.batches.create.return_value = mock_batch_job

        captured_src: list = []

        def _capture_create(model, src, config):
            captured_src.append(src)
            return mock_batch_job

        mock_client.batches.create.side_effect = _capture_create

        mock_google = MagicMock()
        mock_google.genai = mock_genai

        with (
            patch.dict("os.environ", {"GOOGLE_API_KEY": "test"}),
            patch.dict(
                sys.modules, {"google": mock_google, "google.genai": mock_genai}
            ),
        ):
            backend = get_batch_backend("google")
            requests = [
                BatchRequest(
                    custom_id="stem-page-1",
                    image_base64="GOOGLEIMG",
                    mime_type="image/png",
                    order_index=1,
                )
            ]
            model_config = {
                "extraction_model": {
                    "name": "gemini-2.5-flash",
                    "max_output_tokens": 512,
                }
            }
            backend.submit_batch(requests, model_config, system_prompt="sys")

        assert len(captured_src) == 1
        src = captured_src[0]
        # Inline submission: src is a list of request dicts
        assert isinstance(src, list)
        parts = src[0]["contents"][0]["parts"]
        inline_parts = [p for p in parts if "inline_data" in p]
        assert len(inline_parts) == 1
        assert inline_parts[0]["inline_data"]["mime_type"] == "image/png"
        assert inline_parts[0]["inline_data"]["data"] == "GOOGLEIMG"


class TestGoogleInlineRequestShape:
    """Regression: inline Google batch requests must use the SDK's
    InlinedRequest shape (model/contents/metadata/config), which forbids extra
    keys. system_instruction and generation params belong inside config."""

    def setup_method(self):
        clear_backend_cache()

    def test_inline_src_validates_against_batchjobsource(self):
        """The captured inline src must construct a real BatchJobSource without
        the 'Extra inputs are not permitted' ValidationError."""
        import sys

        # Real SDK types for validation; only the client is mocked.
        from google.genai import types as real_types

        mock_genai = MagicMock()
        mock_genai.types = real_types
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client

        mock_batch_job = MagicMock()
        mock_batch_job.name = "batches/google-1"
        mock_client.batches.create.return_value = mock_batch_job

        captured: dict = {}

        def _capture_create(model, src, config):
            captured["src"] = src
            return mock_batch_job

        mock_client.batches.create.side_effect = _capture_create

        mock_google = MagicMock()
        mock_google.genai = mock_genai

        with (
            patch.dict("os.environ", {"GOOGLE_API_KEY": "test"}),
            patch.dict(
                sys.modules, {"google": mock_google, "google.genai": mock_genai}
            ),
        ):
            backend = get_batch_backend("google")
            backend.submit_batch(
                [BatchRequest(custom_id="req-1", text="hello", order_index=1)],
                {"extraction_model": {"name": "gemini-2.5-flash", "temperature": 0.0}},
                system_prompt="Extract data.",
            )

        src = captured["src"]
        item = src[0]
        # No forbidden top-level keys.
        assert set(item.keys()) == {"contents", "config"}
        assert item["config"]["system_instruction"] == "Extract data."
        # Constructing BatchJobSource with the real model must not raise.
        source = real_types.BatchJobSource(inlined_requests=src)
        assert source.inlined_requests[0].config.system_instruction == "Extract data."


class TestAnthropicMaxTokensClamp:
    """Regression: submit_batch must clamp max_output_tokens to the model's
    registry cap, mirroring the sync path, or every batch request 400s."""

    def setup_method(self):
        clear_backend_cache()

    @patch("anthropic.Anthropic")
    def test_submit_clamps_max_tokens_to_cap(self, mock_anthropic_class):
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        mock_batch_response = MagicMock()
        mock_batch_response.id = "anthr-batch-clamp"

        captured: dict = {}

        def _capture_create(requests):
            captured["requests"] = requests
            return mock_batch_response

        mock_client.messages.batches.create.side_effect = _capture_create

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}):
            backend = get_batch_backend("anthropic")
            model_name = "claude-haiku-4-5"
            caps = detect_capabilities(model_name, provider="anthropic")
            cap = caps.max_output_tokens
            assert cap is not None  # guard: this model has a known cap
            backend.submit_batch(
                [BatchRequest(custom_id="req-1", text="hello", order_index=1)],
                {"extraction_model": {"name": model_name, "max_output_tokens": 128000}},
                system_prompt="sys",
            )

        sent_max = captured["requests"][0]["params"]["max_tokens"]
        assert sent_max == cap
        assert sent_max < 128000


class TestAnthropicErrorUnwrap:
    """Regression: an errored result carries an ErrorResponse wrapping the real
    ErrorObject; the backend must unwrap one level to surface message/type."""

    def setup_method(self):
        clear_backend_cache()

    @patch("anthropic.Anthropic")
    def test_errored_result_surfaces_real_message(self, mock_anthropic_class):
        from anthropic.types import ErrorResponse

        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        error_response = ErrorResponse(
            type="error",
            error={"type": "invalid_request_error", "message": "bad thing happened"},
        )
        fake_result_data = MagicMock()
        fake_result_data.type = "errored"
        fake_result_data.error = error_response

        fake_result = MagicMock()
        fake_result.custom_id = "req-1"
        fake_result.result = fake_result_data

        mock_client.messages.batches.results.return_value = iter([fake_result])

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}):
            backend = get_batch_backend("anthropic")
            handle = BatchHandle(provider="anthropic", batch_id="batch-1")
            results = list(backend.download_results(handle))

        assert len(results) == 1
        assert results[0].success is False
        assert results[0].error == "bad thing happened"
        assert results[0].error_code == "invalid_request_error"


class TestEmptyContentBatchResults:
    """Regression: a transport-level success with no extractable text must be
    reported as a failure in EVERY backend (the guard existed only on the
    Google file-results path). A success with empty content is written as a
    completed empty record, is skipped on resume, and its chunk is lost."""

    def setup_method(self):
        clear_backend_cache()

    @patch("openai.OpenAI")
    def test_openai_incomplete_body_without_text_marks_failure(self, mock_openai_class):
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_batch = MagicMock()
        mock_batch.output_file_id = "output-file-1"
        mock_client.batches.retrieve.return_value = mock_batch

        mock_stream = MagicMock()
        mock_stream.read.return_value = json.dumps(
            {
                "custom_id": "doc-chunk-1",
                "response": {
                    "status_code": 200,
                    "body": {
                        # Reasoning consumed max_output_tokens: no message item.
                        "status": "incomplete",
                        "output": [{"type": "reasoning"}],
                    },
                },
            }
        ).encode("utf-8")
        mock_client.files.content.return_value = mock_stream

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            backend = get_batch_backend("openai")
            handle = BatchHandle(provider="openai", batch_id="batch-123")
            results = list(backend.download_results(handle))

        assert len(results) == 1
        assert results[0].success is False
        assert "incomplete" in (results[0].error or "")

    @patch("anthropic.Anthropic")
    def test_anthropic_message_without_text_blocks_marks_failure(
        self, mock_anthropic_class
    ):
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        thinking_block = MagicMock()
        thinking_block.type = "thinking"
        message = MagicMock()
        message.content = [thinking_block]
        message.stop_reason = "max_tokens"
        message.usage = None

        fake_result_data = MagicMock()
        fake_result_data.type = "succeeded"
        fake_result_data.message = message

        fake_result = MagicMock()
        fake_result.custom_id = "doc-chunk-1"
        fake_result.result = fake_result_data

        mock_client.messages.batches.results.return_value = iter([fake_result])

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}):
            backend = get_batch_backend("anthropic")
            handle = BatchHandle(provider="anthropic", batch_id="batch-1")
            results = list(backend.download_results(handle))

        assert len(results) == 1
        assert results[0].success is False
        assert "max_tokens" in (results[0].error or "")

    @patch("anthropic.Anthropic")
    def test_anthropic_succeeded_without_message_marks_failure(
        self, mock_anthropic_class
    ):
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        fake_result_data = MagicMock()
        fake_result_data.type = "succeeded"
        fake_result_data.message = None

        fake_result = MagicMock()
        fake_result.custom_id = "doc-chunk-1"
        fake_result.result = fake_result_data

        mock_client.messages.batches.results.return_value = iter([fake_result])

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}):
            backend = get_batch_backend("anthropic")
            handle = BatchHandle(provider="anthropic", batch_id="batch-1")
            results = list(backend.download_results(handle))

        assert len(results) == 1
        assert results[0].success is False
        assert results[0].error is not None

    def test_google_inline_response_without_text_marks_failure(self):
        import sys

        mock_genai = MagicMock()
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client

        inline_response = MagicMock()
        inline_response.error = None
        mock_response = MagicMock()
        mock_response.text = ""
        inline_response.response = mock_response

        mock_dest = MagicMock(spec=["inlined_responses"])
        mock_dest.inlined_responses = [inline_response]
        mock_batch_job = MagicMock()
        mock_batch_job.dest = mock_dest
        mock_client.batches.get.return_value = mock_batch_job

        mock_google = MagicMock()
        mock_google.genai = mock_genai

        with (
            patch.dict("os.environ", {"GOOGLE_API_KEY": "test"}),
            patch.dict(
                sys.modules, {"google": mock_google, "google.genai": mock_genai}
            ),
        ):
            backend = get_batch_backend("google")
            handle = BatchHandle(
                provider="google",
                batch_id="batch-123",
                metadata={"custom_id_map": {"doc-chunk-1": 0}},
            )
            results = list(backend.download_results(handle))

        assert len(results) == 1
        assert results[0].success is False
        assert "No text content" in (results[0].error or "")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
