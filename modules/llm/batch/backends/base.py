"""Base interface for batch processing backends.

Defines the abstract interface that all batch processing backends must implement.
Adapted for ChronoMiner's text extraction workflow (text chunks instead of images).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


class BatchStatus(Enum):
    """Status of a batch job."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    UNKNOWN = "unknown"


@dataclass
class BatchHandle:
    """Handle to track a submitted batch job.
    
    Attributes:
        provider: The provider name (openai, anthropic, google)
        batch_id: Provider-specific batch identifier
        metadata: Additional provider-specific metadata
    """
    provider: str
    batch_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for persistence."""
        return {
            "provider": self.provider,
            "batch_id": self.batch_id,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BatchHandle":
        """Deserialize from dictionary."""
        return cls(
            provider=data.get("provider", ""),
            batch_id=data.get("batch_id", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class BatchRequest:
    """A single request to include in a batch.
    
    Adapted for text extraction: carries text content instead of image data.
    
    Attributes:
        custom_id: Unique identifier for matching results to requests
        text: The text chunk to process
        order_index: Original order index for result sorting
        metadata: Additional metadata (e.g., file_path, chunk_range)
    """
    custom_id: str
    text: str = ""
    order_index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchResultItem:
    """Result of a single request in a batch.
    
    Attributes:
        custom_id: Matches the request's custom_id
        success: Whether the request succeeded
        content: The raw response text
        parsed_output: Parsed structured output (if schema was used)
        error: Error message (if failed)
        error_code: Provider-specific error code
        raw_response: The raw response from the provider
        input_tokens: Number of input tokens used
        output_tokens: Number of output tokens used
    """
    custom_id: str
    success: bool = True
    content: str = ""
    parsed_output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    raw_response: Dict[str, Any] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    
    @property
    def has_entries(self) -> bool:
        """Check if result contains entries."""
        if self.parsed_output and isinstance(self.parsed_output, dict):
            entries = self.parsed_output.get("entries", [])
            return isinstance(entries, list) and len(entries) > 0
        return False
    
    @property
    def contains_no_content(self) -> bool:
        """Check if result indicates no content of requested type."""
        if self.parsed_output and isinstance(self.parsed_output, dict):
            return self.parsed_output.get("contains_no_content_of_requested_type", False)
        return False


@dataclass
class BatchStatusInfo:
    """Detailed status information for a batch job.
    
    Attributes:
        status: The overall batch status
        total_requests: Total number of requests in the batch
        completed_requests: Number of completed requests
        failed_requests: Number of failed requests
        pending_requests: Number of pending requests
        error_message: Overall error message (if batch failed)
        results_available: Whether results can be downloaded
        output_file_id: Provider-specific output file identifier (if applicable)
    """
    status: BatchStatus
    total_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    pending_requests: int = 0
    error_message: Optional[str] = None
    results_available: bool = False
    output_file_id: Optional[str] = None


class BatchBackend(ABC):
    """Abstract base class for batch processing backends.
    
    Each provider (OpenAI, Anthropic, Google) implements this interface
    to provide batch processing capabilities for text extraction.
    """
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name (e.g., 'openai', 'anthropic', 'google')."""
        pass
    
    @property
    @abstractmethod
    def max_batch_size(self) -> int:
        """Return the maximum number of requests per batch."""
        pass
    
    @property
    @abstractmethod
    def max_batch_bytes(self) -> int:
        """Return the maximum batch size in bytes."""
        pass
    
    @abstractmethod
    def submit_batch(
        self,
        requests: List[BatchRequest],
        model_config: Dict[str, Any],
        *,
        system_prompt: str,
        schema: Optional[Dict[str, Any]] = None,
        schema_name: Optional[str] = None,
    ) -> BatchHandle:
        """Submit a batch of requests for processing.
        
        Args:
            requests: List of BatchRequest objects to process
            model_config: Model configuration dict (transcription_model settings)
            system_prompt: System prompt text (already rendered with schema)
            schema: JSON schema for structured output (optional)
            schema_name: Name of the schema for structured outputs
        
        Returns:
            BatchHandle for tracking the submitted batch
        
        Raises:
            Exception: If batch submission fails
        """
        pass
    
    @abstractmethod
    def get_status(self, handle: BatchHandle) -> BatchStatusInfo:
        """Get the current status of a batch job.
        
        Args:
            handle: The BatchHandle returned from submit_batch
        
        Returns:
            BatchStatusInfo with current status details
        """
        pass
    
    @abstractmethod
    def download_results(self, handle: BatchHandle) -> Iterator[BatchResultItem]:
        """Download and iterate over batch results.
        
        Args:
            handle: The BatchHandle returned from submit_batch
        
        Yields:
            BatchResultItem for each request in the batch
        
        Raises:
            Exception: If results cannot be downloaded
        """
        pass
    
    @abstractmethod
    def cancel(self, handle: BatchHandle) -> bool:
        """Cancel a batch job.
        
        Args:
            handle: The BatchHandle returned from submit_batch
        
        Returns:
            True if cancellation was successful
        """
        pass
    
    def diagnose_failure(self, handle: BatchHandle) -> str:
        """Get diagnostic information for a failed batch.
        
        Args:
            handle: The BatchHandle of the failed batch
        
        Returns:
            Human-readable diagnostic message
        """
        status_info = self.get_status(handle)
        if status_info.error_message:
            return status_info.error_message
        return f"Batch {handle.batch_id} has status {status_info.status.value}"
