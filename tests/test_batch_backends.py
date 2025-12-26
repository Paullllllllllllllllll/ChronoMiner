"""Tests for multi-provider batch backends."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from modules.llm.batch import (
    BatchBackend,
    BatchHandle,
    BatchRequest,
    BatchResultItem,
    BatchStatus,
    BatchStatusInfo,
    get_batch_backend,
    supports_batch,
    clear_backend_cache,
)


class TestBatchDataClasses:
    """Test batch data classes."""
    
    def test_batch_handle_serialization(self):
        """Test BatchHandle to_dict and from_dict."""
        handle = BatchHandle(
            provider="openai",
            batch_id="batch_123",
            metadata={"key": "value"}
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
            metadata={"file": "test.txt"}
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
            parsed_output={"entries": [{"name": "Test"}]}
        )
        assert result_with_entries.has_entries is True
        assert result_with_entries.contains_no_content is False
        
        # Result with no content flag
        result_no_content = BatchResultItem(
            custom_id="req-2",
            success=True,
            parsed_output={"contains_no_content_of_requested_type": True}
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
            output_file_id="file-123"
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
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            backend = get_batch_backend("openai")
            assert backend.provider_name == "openai"
            assert backend.max_batch_size == 50000
    
    def test_get_batch_backend_anthropic(self):
        """Test getting Anthropic backend."""
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
            backend = get_batch_backend("anthropic")
            assert backend.provider_name == "anthropic"
            assert backend.max_batch_size == 100000
    
    def test_get_batch_backend_google(self):
        """Test getting Google backend."""
        with patch.dict('os.environ', {'GOOGLE_API_KEY': 'test-key'}):
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
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            backend1 = get_batch_backend("openai")
            backend2 = get_batch_backend("openai")
            assert backend1 is backend2
    
    def test_clear_backend_cache(self):
        """Test clearing backend cache."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            backend1 = get_batch_backend("openai")
            clear_backend_cache()
            backend2 = get_batch_backend("openai")
            assert backend1 is not backend2


class TestOpenAIBackend:
    """Test OpenAI batch backend."""
    
    def setup_method(self):
        clear_backend_cache()
    
    @patch('modules.llm.batch.backends.openai_backend.OpenAI')
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
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            backend = get_batch_backend("openai")
            
            requests = [
                BatchRequest(custom_id="req-1", text="Sample text 1", order_index=1),
                BatchRequest(custom_id="req-2", text="Sample text 2", order_index=2),
            ]
            
            model_config = {
                "transcription_model": {
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
    
    @patch('modules.llm.batch.backends.openai_backend.OpenAI')
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
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            backend = get_batch_backend("openai")
            handle = BatchHandle(provider="openai", batch_id="batch-123")
            
            status = backend.get_status(handle)
            
            assert status.status == BatchStatus.COMPLETED
            assert status.total_requests == 10
            assert status.completed_requests == 10
            assert status.results_available is True


class TestAnthropicBackend:
    """Test Anthropic batch backend."""
    
    def setup_method(self):
        clear_backend_cache()
    
    @patch('modules.llm.batch.backends.anthropic_backend.anthropic')
    def test_get_status_in_progress(self, mock_anthropic_module):
        """Test getting in-progress batch status."""
        mock_client = MagicMock()
        mock_anthropic_module.Anthropic.return_value = mock_client
        
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
        
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
            backend = get_batch_backend("anthropic")
            handle = BatchHandle(provider="anthropic", batch_id="batch-123")
            
            status = backend.get_status(handle)
            
            assert status.status == BatchStatus.IN_PROGRESS
            assert status.completed_requests == 3
            assert status.pending_requests == 5


class TestGoogleBackend:
    """Test Google batch backend."""
    
    def setup_method(self):
        clear_backend_cache()
    
    @patch('modules.llm.batch.backends.google_backend.genai', create=True)
    def test_get_status_completed(self, mock_genai_module):
        """Test getting completed batch status."""
        mock_client = MagicMock()
        mock_genai_module.Client.return_value = mock_client
        
        mock_batch_job = MagicMock()
        mock_batch_job.state = MagicMock()
        mock_batch_job.state.name = "JOB_STATE_SUCCEEDED"
        mock_batch_job.dest = MagicMock()
        mock_batch_job.dest.file_name = "output-file"
        mock_client.batches.get.return_value = mock_batch_job
        
        with patch.dict('os.environ', {'GOOGLE_API_KEY': 'test-key'}):
            # Patch the import inside the backend
            with patch('modules.llm.batch.backends.google_backend.genai', mock_genai_module):
                backend = get_batch_backend("google")
                handle = BatchHandle(
                    provider="google", 
                    batch_id="batch-123",
                    metadata={"request_count": 5}
                )
                
                status = backend.get_status(handle)
                
                assert status.status == BatchStatus.COMPLETED
                assert status.results_available is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
