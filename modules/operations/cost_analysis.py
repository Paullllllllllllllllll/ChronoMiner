# modules/operations/cost_analysis.py

"""
Cost analysis operations for token usage and pricing calculations.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from modules.core.logger import setup_logger

logger = setup_logger(__name__)


# Model pricing per million tokens (input, cached_input, output)
# Note: Use 0.0 for cached_input when not available
# Pricing last updated: December 2024
MODEL_PRICING = {
    # ============== OpenAI Models ==============
    # GPT-5 family
    "gpt-5": (1.25, 0.125, 10.00),
    "gpt-5-mini": (0.25, 0.025, 2.00),
    "gpt-5-nano": (0.05, 0.005, 0.40),
    "gpt-5-chat-latest": (1.25, 0.125, 10.00),
    "gpt-5-codex": (1.25, 0.125, 10.00),
    # GPT-4.1 family
    "gpt-4.1": (2.00, 0.50, 8.00),
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "gpt-4.1-nano": (0.10, 0.025, 0.40),
    # GPT-4o family
    "gpt-4o": (2.50, 1.25, 10.00),
    "gpt-4o-2024-05-13": (5.00, 0.00, 15.00),
    "gpt-4o-mini": (0.15, 0.075, 0.60),
    "gpt-4o-realtime-preview": (5.00, 2.50, 20.00),
    "gpt-4o-mini-realtime-preview": (0.60, 0.30, 2.40),
    "gpt-4o-audio-preview": (2.50, 0.00, 10.00),
    "gpt-4o-mini-audio-preview": (0.15, 0.00, 0.60),
    "gpt-4o-search-preview": (2.50, 0.00, 10.00),
    "gpt-4o-mini-search-preview": (0.15, 0.00, 0.60),
    # Audio models
    "gpt-audio": (2.50, 0.00, 10.00),
    # o-series models
    "o1": (15.00, 7.50, 60.00),
    "o1-pro": (150.00, 0.00, 600.00),
    "o1-mini": (1.10, 0.55, 4.40),
    "o3": (2.00, 0.50, 8.00),
    "o3-pro": (20.00, 0.00, 80.00),
    "o3-mini": (1.10, 0.55, 4.40),
    "o3-deep-research": (10.00, 2.50, 40.00),
    "o4-mini": (1.10, 0.275, 4.40),
    "o4-mini-deep-research": (2.00, 0.50, 8.00),
    # Other OpenAI models
    "codex-mini-latest": (1.50, 0.375, 6.00),
    "computer-use-preview": (3.00, 0.00, 12.00),
    "gpt-image-1": (5.00, 1.25, 0.00),  # Image generation (no output tokens)
    
    # ============== Anthropic Models ==============
    # Claude 4 family (Haiku 4.5: $1/$5 per million tokens)
    "claude-haiku-4-5": (1.00, 0.10, 5.00),
    "claude-haiku-4-5-20251101": (1.00, 0.10, 5.00),
    # Claude 3.5 family
    "claude-3-5-sonnet": (3.00, 0.30, 15.00),
    "claude-3-5-sonnet-20241022": (3.00, 0.30, 15.00),
    "claude-3-5-haiku": (1.00, 0.10, 5.00),
    "claude-3-5-haiku-20241022": (1.00, 0.10, 5.00),
    # Claude 3 family
    "claude-3-opus": (15.00, 1.50, 75.00),
    "claude-3-opus-20240229": (15.00, 1.50, 75.00),
    "claude-3-sonnet": (3.00, 0.30, 15.00),
    "claude-3-sonnet-20240229": (3.00, 0.30, 15.00),
    "claude-3-haiku": (0.25, 0.03, 1.25),
    "claude-3-haiku-20240307": (0.25, 0.03, 1.25),
    
    # ============== Google Gemini Models ==============
    # Gemini 2.5 family
    "gemini-2.5-pro": (1.25, 0.125, 10.00),
    "gemini-2.5-flash": (0.30, 0.03, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.01, 0.40),
    # Gemini 2.0 family
    "gemini-2.0-flash": (0.10, 0.01, 0.40),
    "gemini-2.0-flash-lite": (0.08, 0.008, 0.30),
    # Gemini 1.5 family (legacy)
    "gemini-1.5-pro": (1.25, 0.125, 5.00),
    "gemini-1.5-flash": (0.075, 0.0075, 0.30),
}


@dataclass
class TokenUsage:
    """Token usage statistics for a single chunk."""
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    model: str = ""


@dataclass
class FileStats:
    """Statistics for a single file."""
    file_path: Path
    model: str = ""
    total_chunks: int = 0
    successful_chunks: int = 0
    failed_chunks: int = 0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    cost_standard: float = 0.0
    cost_discounted: float = 0.0


@dataclass
class CostAnalysis:
    """Overall cost analysis results."""
    file_stats: List[FileStats] = field(default_factory=list)
    total_files: int = 0
    total_chunks: int = 0
    total_prompt_tokens: int = 0
    total_cached_tokens: int = 0
    total_completion_tokens: int = 0
    total_reasoning_tokens: int = 0
    total_tokens: int = 0
    total_cost_standard: float = 0.0
    total_cost_discounted: float = 0.0
    models_used: Dict[str, int] = field(default_factory=dict)


def normalize_model_name(model: str) -> str:
    """
    Normalize model name by removing version suffixes.
    
    Args:
        model: Full model name (e.g., 'gpt-5-mini-2025-08-07')
        
    Returns:
        Normalized model name (e.g., 'gpt-5-mini')
    """
    # Check for exact match first
    if model in MODEL_PRICING:
        return model
    
    # Try to match base model name by removing date suffixes
    for base_model in MODEL_PRICING.keys():
        if model.startswith(base_model):
            return base_model
    
    return model


def extract_token_usage_from_record(record: Dict[str, Any]) -> Optional[TokenUsage]:
    """
    Extract token usage from a single JSONL record.
    
    Handles both structures:
    - Direct: record["response_data"]
    - Nested: record["response"]["body"]["response_data"] (temp.jsonl from sync processing)
    
    Args:
        record: Dictionary containing the JSONL record
        
    Returns:
        TokenUsage object or None if no usage data found
    """
    usage = TokenUsage()
    
    # Try to get response_data from different locations
    # 1. Direct at record level (batch API responses)
    response_data = record.get("response_data", {})
    request_metadata = record.get("request_metadata", {})
    
    # 2. Nested under response.body (sync processing temp.jsonl)
    if not response_data:
        body = record.get("response", {}).get("body", {})
        if isinstance(body, dict):
            response_data = body.get("response_data", {})
            request_metadata = body.get("request_metadata", request_metadata)
    
    # Extract model
    if isinstance(response_data, dict):
        usage.model = response_data.get("model", "")
    if not usage.model and isinstance(request_metadata, dict):
        # Try from request_metadata
        usage.model = request_metadata.get("model", "")
        if not usage.model:
            payload = request_metadata.get("payload", {})
            if isinstance(payload, dict):
                usage.model = payload.get("model", "")
    
    # Extract usage data from response_data
    if isinstance(response_data, dict):
        usage_data = response_data.get("usage", {})
        if isinstance(usage_data, dict):
            # Try both naming conventions (OpenAI API uses input_tokens/output_tokens)
            usage.prompt_tokens = usage_data.get("input_tokens") or usage_data.get("prompt_tokens") or 0
            usage.completion_tokens = usage_data.get("output_tokens") or usage_data.get("completion_tokens") or 0
            usage.total_tokens = usage_data.get("total_tokens", 0)
            
            # Calculate total if not provided
            if usage.total_tokens == 0 and (usage.prompt_tokens > 0 or usage.completion_tokens > 0):
                usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
            
            # Extract cached tokens from input_tokens_details or prompt_tokens_details
            input_details = usage_data.get("input_tokens_details", usage_data.get("prompt_tokens_details", {}))
            if isinstance(input_details, dict):
                usage.cached_tokens = input_details.get("cached_tokens", 0)
            
            # Extract reasoning tokens from output_tokens_details or completion_tokens_details
            output_details = usage_data.get("output_tokens_details", usage_data.get("completion_tokens_details", {}))
            if isinstance(output_details, dict):
                usage.reasoning_tokens = output_details.get("reasoning_tokens", 0)
            
            return usage
    
    return None


def calculate_cost(
    prompt_tokens: int,
    cached_tokens: int,
    completion_tokens: int,
    model: str,
    discount: float = 0.0
) -> float:
    """
    Calculate cost for token usage.
    
    Args:
        prompt_tokens: Number of prompt tokens (excluding cached)
        cached_tokens: Number of cached prompt tokens
        completion_tokens: Number of completion tokens
        model: Model name
        discount: Discount percentage (0.0 to 1.0)
        
    Returns:
        Total cost in dollars
    """
    # Normalize model name
    normalized_model = normalize_model_name(model)
    
    if normalized_model not in MODEL_PRICING:
        logger.warning(f"Unknown model '{model}' (normalized: '{normalized_model}'), cannot calculate cost")
        return 0.0
    
    input_price, cached_price, output_price = MODEL_PRICING[normalized_model]
    
    # Apply discount
    if discount > 0:
        input_price *= (1 - discount)
        cached_price *= (1 - discount)
        output_price *= (1 - discount)
    
    # Calculate cost per million tokens
    uncached_prompt = prompt_tokens - cached_tokens
    cost = (
        (uncached_prompt * input_price / 1_000_000) +
        (cached_tokens * cached_price / 1_000_000) +
        (completion_tokens * output_price / 1_000_000)
    )
    
    return cost


def analyze_jsonl_file(jsonl_path: Path) -> FileStats:
    """
    Analyze a single temporary .jsonl file.
    
    Args:
        jsonl_path: Path to the .jsonl file
        
    Returns:
        FileStats object with analysis results
    """
    stats = FileStats(file_path=jsonl_path)
    
    try:
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    record = json.loads(line)
                    
                    # Skip batch tracking records
                    if "batch_tracking" in record:
                        continue
                    
                    stats.total_chunks += 1
                    
                    # Check status - consider records with response data as successful
                    status = record.get("status", "")
                    has_response = bool(record.get("response", {}).get("body"))
                    if status == "success" or (not status and has_response):
                        stats.successful_chunks += 1
                    else:
                        stats.failed_chunks += 1
                    
                    # Extract token usage
                    usage = extract_token_usage_from_record(record)
                    if usage:
                        if not stats.model:
                            stats.model = usage.model
                        
                        stats.prompt_tokens += usage.prompt_tokens
                        stats.cached_tokens += usage.cached_tokens
                        stats.completion_tokens += usage.completion_tokens
                        stats.reasoning_tokens += usage.reasoning_tokens
                        stats.total_tokens += usage.total_tokens
                
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line in {jsonl_path.name}: {e}")
                    continue
        
        # Calculate costs
        if stats.model:
            stats.cost_standard = calculate_cost(
                stats.prompt_tokens,
                stats.cached_tokens,
                stats.completion_tokens,
                stats.model,
                discount=0.0
            )
            stats.cost_discounted = calculate_cost(
                stats.prompt_tokens,
                stats.cached_tokens,
                stats.completion_tokens,
                stats.model,
                discount=0.5
            )
    
    except Exception as e:
        logger.error(f"Error analyzing {jsonl_path}: {e}")
    
    return stats


def find_jsonl_files(paths_config: Dict, schemas_paths: Dict) -> List[Path]:
    """
    Find all temporary .jsonl files based on configuration.
    
    Args:
        paths_config: Paths configuration dictionary
        schemas_paths: Schema-specific paths dictionary
        
    Returns:
        List of Path objects for .jsonl files
    """
    jsonl_files: list[Path] = []
    input_is_output = paths_config.get("general", {}).get("input_paths_is_output_path", True)
    
    # Scan schema-specific directories
    for schema_name, schema_config in schemas_paths.items():
        if input_is_output:
            # Files are in input directory
            input_dir = schema_config.get("input")
            if input_dir:
                input_path = Path(input_dir)
                if input_path.exists():
                    jsonl_files.extend(input_path.glob("*_temp.jsonl"))
        else:
            # Files are in output directory
            output_dir = schema_config.get("output")
            if output_dir:
                output_path = Path(output_dir)
                if output_path.exists():
                    # Check both root and temp_jsonl subfolder
                    jsonl_files.extend(output_path.glob("*_temp.jsonl"))
                    temp_folder = output_path / "temp_jsonl"
                    if temp_folder.exists():
                        jsonl_files.extend(temp_folder.glob("*_temp.jsonl"))
    
    return sorted(set(jsonl_files))


def perform_cost_analysis(jsonl_files: List[Path]) -> CostAnalysis:
    """
    Perform cost analysis on all .jsonl files.
    
    Args:
        jsonl_files: List of .jsonl file paths
        
    Returns:
        CostAnalysis object with results
    """
    analysis = CostAnalysis()
    analysis.total_files = len(jsonl_files)
    
    for jsonl_path in jsonl_files:
        file_stats = analyze_jsonl_file(jsonl_path)
        analysis.file_stats.append(file_stats)
        
        # Aggregate totals
        analysis.total_chunks += file_stats.total_chunks
        analysis.total_prompt_tokens += file_stats.prompt_tokens
        analysis.total_cached_tokens += file_stats.cached_tokens
        analysis.total_completion_tokens += file_stats.completion_tokens
        analysis.total_reasoning_tokens += file_stats.reasoning_tokens
        analysis.total_tokens += file_stats.total_tokens
        analysis.total_cost_standard += file_stats.cost_standard
        analysis.total_cost_discounted += file_stats.cost_discounted
        
        # Track models used
        if file_stats.model:
            analysis.models_used[file_stats.model] = analysis.models_used.get(file_stats.model, 0) + 1
    
    return analysis


def save_analysis_to_csv(analysis: CostAnalysis, output_path: Path) -> None:
    """
    Save cost analysis to CSV file.
    
    Args:
        analysis: CostAnalysis object
        output_path: Path to save CSV file
    """
    import csv
    
    try:
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                "File",
                "Model",
                "Total Chunks",
                "Successful Chunks",
                "Failed Chunks",
                "Prompt Tokens",
                "Cached Tokens",
                "Completion Tokens",
                "Reasoning Tokens",
                "Total Tokens",
                "Cost (Standard)",
                "Cost (50% Discount)"
            ])
            
            # Write file stats
            for stats in analysis.file_stats:
                writer.writerow([
                    stats.file_path.name,
                    stats.model or "Unknown",
                    stats.total_chunks,
                    stats.successful_chunks,
                    stats.failed_chunks,
                    stats.prompt_tokens,
                    stats.cached_tokens,
                    stats.completion_tokens,
                    stats.reasoning_tokens,
                    stats.total_tokens,
                    f"{stats.cost_standard:.4f}",
                    f"{stats.cost_discounted:.4f}"
                ])
            
            # Write summary row
            writer.writerow([])
            writer.writerow([
                "TOTAL",
                ", ".join(analysis.models_used.keys()),
                analysis.total_chunks,
                "",
                "",
                analysis.total_prompt_tokens,
                analysis.total_cached_tokens,
                analysis.total_completion_tokens,
                analysis.total_reasoning_tokens,
                analysis.total_tokens,
                f"{analysis.total_cost_standard:.4f}",
                f"{analysis.total_cost_discounted:.4f}"
            ])
        
        logger.info(f"Saved cost analysis to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save CSV: {e}")
        raise
